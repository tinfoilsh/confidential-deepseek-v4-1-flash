"""Bind a measured kernel bundle to the actual inference runtime."""

import csv
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

from b200_bundle import digest, identity_key, install

SITE = Path("/usr/local/lib/python3.12/dist-packages")
BUILD = Path("/opt/tinfoil/b200-runtime-build.json")
BUNDLE = Path("/opt/tinfoil/b200-bundle")
CACHE_ROOT = Path("/root")
MODEL_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
MODEL_PACK = "mpk-159977c3efb94d1eba09025f71af1d1d55d6a58c58e4165b4587ec83166fa385"
QUALIFIED_POLICY = {"timer": "globaltimer", "timer_boundary": "captured_kernel_v1",
                    "warmup": 3, "repeat": 10, "l2_cache_policy": "cold", "cuda_graph_profile_replays": 1}
PACKAGE_NAMES = {"vllm", "torch", "triton", "transformers", "flashinfer-python", "apache-tvm-ffi"}
SOURCE_NAMES = (
    "flashinfer/autotuner/autotuner.py", "flashinfer/utils.py", "flashinfer/jit/core.py",
    "flashinfer/jit/cute_dsl_core.py", "flashinfer/gemm/gemm_base.py",
    "flashinfer/gemm/gemm_mm_fp4_cute_dsl.py",
    "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py",
    "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_common.py",
    "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py",
    "vllm/model_executor/warmup/flashinfer_autotune_cache.py",
)
ENVIRONMENT_NAMES = (
    "FLASHINFER_AUTOTUNE_TIMER", "FLASHINFER_CUTE_DSL_DISABLE_CACHE", "CUTE_DSL_ARCH",
    "FLASHINFER_CUDA_ARCH_LIST", "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_ALLREDUCE_USE_FLASHINFER",
    "VLLM_DISABLE_UVA", "VLLM_TINFOIL_CC_OPTIMIZATIONS", "VLLM_TINFOIL_DEEPSEEK_TARGET_TACTIC",
    "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
)
COMMAND_TIMEOUT_SECONDS = 15
LOGGER = logging.getLogger("b200_runtime")


def option(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def capture(command):
    return subprocess.check_output(command, text=True, timeout=COMMAND_TIMEOUT_SECONDS).strip()


def describe(command, *, gpu_mode="cc"):
    if gpu_mode not in ("cc", "devtools"):
        raise ValueError("Use an explicit GPU measurement mode")
    model = Path(option(command, "--model"))
    if model.name != MODEL_PACK:
        raise ValueError("Qualify a new model pack before packaging its kernels")
    config = json.loads((model / "config.json").read_text())
    gpu = list(csv.reader(capture(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                                   "--format=csv,noheader,nounits"]).splitlines(), skipinitialspace=True))
    cc = capture(["nvidia-smi", "conf-compute", "-q"])
    devtools = capture(["nvidia-smi", "conf-compute", "--get-devtools-mode"])
    cc_state = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                for line in cc.splitlines() if ":" in line}
    if (len(gpu) != int(option(command, "--tensor-parallel-size"))
            or cc_state.get("CC State") != "ON"
            or devtools != ("DevTools Mode: ON" if gpu_mode == "devtools" else "DevTools Mode: OFF")):
        raise ValueError("The GPU allocation or confidential-computing state differs")
    distributions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                     if dist.metadata["Name"] in PACKAGE_NAMES
                     or dist.metadata["Name"].startswith(("nvidia-", "flashinfer-"))}
    if not PACKAGE_NAMES <= distributions.keys():
        raise ValueError("A required runtime package is missing")
    timer = os.environ.get("FLASHINFER_AUTOTUNE_TIMER")
    if timer != "globaltimer" or os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") != "0":
        raise ValueError("The qualified timing and native compiled-cache paths must be enabled")
    return {
        "gpu": {"devices": gpu, "cc": cc_state, "devtools": devtools},
        "model": {"revision": MODEL_REVISION, "pack": model.name,
                  "config_sha256": digest(model / "config.json"),
                  "weight_index_sha256": digest(model / "model.safetensors.index.json")},
        "configuration": {"command": command, "environment": {name: os.environ.get(name) for name in ENVIRONMENT_NAMES}},
        "precision": {"quantization": config.get("quantization_config"),
                      "model_dtype": config.get("dtype", config.get("torch_dtype")),
                      "activation_dtype": option(command, "--dtype", "auto"),
                      "kv_dtype": option(command, "--kv-cache-dtype", "auto")},
        "software": {"build": json.loads(BUILD.read_text()), "packages": distributions,
                     "cpu_architecture": platform.machine(), "python": platform.python_version(),
                     "sources": {name: digest(SITE / name) for name in SOURCE_NAMES},
                     "loader_sha256": digest(Path(__file__)),
                     "bundle_code_sha256": digest(Path(__file__).with_name("b200_bundle.py")),
                     "timer_library_sha256": digest(Path(os.environ["FLASHINFER_GLOBALTIMER_LIBRARY"]))},
        "measurement_policy": {"timer": timer, "boundary": "captured_kernel_v1",
                               "profiles": [QUALIFIED_POLICY],
                               "per_shape_policy": "flashinfer.profiling_policy.v1",
                               "autotuner_source_sha256": digest(SITE / SOURCE_NAMES[0])},
    }


def main():
    logging.basicConfig(level=logging.INFO)
    if sys.argv[1:3] != ["vllm", "serve"]:
        raise ValueError("Use the declared vLLM serve entrypoint")
    identity = describe(sys.argv[3:])
    key = identity_key(identity)
    tuning_root = CACHE_ROOT / ".cache/vllm/b200-qualified" / key
    mode = os.environ.get("TINFOIL_TUNING_MODE", "require")
    if mode == "tune":
        tuning_root.mkdir(parents=True, exist_ok=True)
        (tuning_root / "runtime-identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        LOGGER.info("Explicit tuning enabled for configuration %s", key)
    elif mode == "require":
        install(BUNDLE, CACHE_ROOT, identity)
        LOGGER.info("Measured shape choices required for configuration %s; startup profiling is disabled", key)
    else:
        raise ValueError("TINFOIL_TUNING_MODE must be tune or require")
    os.environ["FLASHINFER_AUTOTUNE_REQUIRE_CACHE"] = "1" if mode == "require" else "0"
    os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = str(tuning_root)
    os.execv("/opt/tinfoil/entrypoint.sh", ["/opt/tinfoil/entrypoint.sh", *sys.argv[1:]])


if __name__ == "__main__":
    main()
