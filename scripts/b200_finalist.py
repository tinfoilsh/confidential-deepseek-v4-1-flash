"""Load the selected B200 kernels without experiment or profiling controls."""

import hashlib
import json
import logging
import os
from pathlib import Path
import sys

from b200_bundle import digest, identity_key, install, verify
from b200_runtime import CACHE_ROOT, SITE, describe as describe_runtime, option

ROOT = Path("/opt/tinfoil")
BUILD = ROOT / "b200-finalist-build.json"
BUNDLE = ROOT / "b200-finalist-bundle"
SOURCE_PACKAGES = ("vllm", "flashinfer")
ENVIRONMENT_PREFIXES = ("VLLM_", "TINFOIL_", "FLASHINFER_", "CUTE_", "CUDA_",
                        "NCCL_", "TORCH_", "PYTORCH_", "OMP_", "TRITON_")
REQUIRED_ENVIRONMENT = {
    "TINFOIL_TUNING_MODE": "require", "FLASHINFER_DISABLE_JIT": "1",
    "TINFOIL_FLASHINFER_REPLAY_ONLY": "1", "TINFOIL_TARGET_PREPARATION": "1",
    "TINFOIL_TARGET_VERIFICATION": "1", "TINFOIL_TARGET_FILTERING": "1",
    "TINFOIL_PROCESSOR_MIRRORS": "1", "TINFOIL_PREFILL_GRAPHS": "1",
    "TINFOIL_PREFILL_CAPTURE_SET": "compact", "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
    "VLLM_TINFOIL_CC_OPTIMIZATIONS": "1", "VLLM_DISABLE_UVA": "1",
}
TINFOIL_ENVIRONMENT = {name for name in REQUIRED_ENVIRONMENT if name.startswith("TINFOIL_")}
LOGGER = logging.getLogger("b200_finalist")


def describe(command):
    build = json.loads(BUILD.read_text())
    if any(os.environ.get(name) != value for name, value in REQUIRED_ENVIRONMENT.items()):
        raise ValueError("The selected inference and cache policy differs")
    if any(name.startswith("TINFOIL_") and name not in TINFOIL_ENVIRONMENT for name in os.environ):
        raise ValueError("Experiment controls are unsupported by the finalist runtime")
    if os.environ.get("LD_PRELOAD") or os.environ.get("NSYS_PROFILING_SESSION_ID"):
        raise ValueError("The finalist runtime requires uninjected execution")
    identity = describe_runtime(command, gpu_mode=build["gpu_mode"])
    identity["configuration"]["environment"] = {
        name: value for name, value in os.environ.items() if name.startswith(ENVIRONMENT_PREFIXES)}
    config = json.loads((Path(option(command, "--model")) / "config.json").read_text())
    text = config.get("text_config", config)
    identity["precision"].update(
        text_quantization=text.get("quantization_config"),
        text_dtype=text.get("dtype", text.get("torch_dtype")))
    identity["software"].update(
        finalist_build=build, finalist_loader_sha256=digest(Path(__file__)),
        entrypoint_sha256=digest(ROOT / "entrypoint.sh"),
        sources={str(path.relative_to(SITE)): digest(path)
                 for package in SOURCE_PACKAGES for path in sorted((SITE / package).rglob("*.py"))})
    identity["measurement_policy"] = build["measurement_policy"]
    return identity


def main():
    logging.basicConfig(level=logging.INFO)
    describe_only = sys.argv[1:2] == ["--describe"]
    verify_only = sys.argv[1:2] == ["--verify"]
    arguments = sys.argv[2:] if describe_only or verify_only else sys.argv[1:]
    if arguments[:2] != ["vllm", "serve"]:
        raise ValueError("Use the declared vLLM serve entrypoint")
    identity = describe(arguments[2:])
    if describe_only:
        sys.stdout.write(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        return
    manifest = verify(BUNDLE, identity)
    tuning = CACHE_ROOT / manifest["tuning_path"]
    build = identity["software"]["finalist_build"]
    if (tuning.parent.name != build["native_cache_key"]
            or manifest["files"][manifest["tuning_path"]]["sha256"] != build["selected_cache_sha256"]):
        raise ValueError("The packaged shape choices differ from the selected candidate")
    if verify_only:
        sys.stdout.write(json.dumps({"identity_key": identity_key(identity),
                                    "tuning_entries": manifest["tuning_entries"]}) + "\n")
        return
    install(BUNDLE, CACHE_ROOT, identity)
    cache = json.loads(tuning.read_text())
    canonical = json.dumps(cache, sort_keys=True, separators=(",", ":")).encode()
    os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = str(tuning.parent.parent)
    os.environ["FLASHINFER_AUTOTUNE_REQUIRE_CACHE"] = "1"
    os.environ["TINFOIL_FLASHINFER_CACHE_KEY"] = tuning.parent.name
    os.environ["TINFOIL_FLASHINFER_CACHE_SHA256"] = hashlib.sha256(canonical).hexdigest()
    LOGGER.info("B200 finalist bundle %s; %s recorded choices, profiling disabled",
                identity_key(identity), manifest["tuning_entries"])
    os.execv(str(ROOT / "entrypoint.sh"), [str(ROOT / "entrypoint.sh"), *arguments])


if __name__ == "__main__":
    main()
