"""Build the declared B200 timer and CUDA cache helper without a GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig

SCHEMA = "tinfoil-b200-helpers-v1"
CUDA_ARCHITECTURES = "10.0"
DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "b200/kernels/helpers.json"
DEFAULT_JOBS = 2
TIMER_MODULE = "flashinfer_globaltimer"
DRIVER_MODULE = "cuda_utils"
CUDA_SOURCES = {"bin/nvcc", "bin/ptxas", "include/cuda_runtime.h"}
SOURCES = {
    "flashinfer/utils.py", "torch/utils/cpp_extension.py", "triton/runtime/build.py",
    "triton/runtime/cache.py", "triton/backends/nvidia/driver.c",
    "triton/backends/nvidia/driver.py",
}
LOGGER = logging.getLogger("b200_compile_helpers")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_recipe(path):
    recipe = json.loads(path.read_text())
    if recipe["schema"] != SCHEMA or recipe["cuda_architectures"] != CUDA_ARCHITECTURES:
        raise ValueError("Unsupported B200 helper recipe")
    if set(recipe["sources"]) != SOURCES or set(recipe["cuda_sources"]) != CUDA_SOURCES:
        raise ValueError("The helper source inventory is incomplete")
    return recipe


def verify_files(root, sources):
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Helper source paths must be contained relative paths")
        if digest(root / path) != expected:
            raise ValueError("The declared helper source differs: " + relative)


def compiler_identity(variable, default):
    name = os.environ.get(variable, default)
    path = shutil.which(name)
    if path is None:
        raise ValueError("The host compiler is unavailable: " + name)
    return {"path": path, "sha256": digest(Path(path)),
            "version": subprocess.check_output([path, "--version"], text=True).splitlines()[0]}


def build(recipe_path, output, cuda_home, cuda_stubs, jobs):
    recipe = load_recipe(recipe_path)
    if jobs <= 0:
        raise ValueError("The build job count must be positive")
    site = Path(sysconfig.get_path("purelib"))
    verify_files(site, recipe["sources"])
    verify_files(cuda_home, recipe["cuda_sources"])
    if digest(cuda_stubs / "libcuda.so") != recipe["cuda_stub_sha256"]:
        raise ValueError("The declared CUDA link stub differs")
    if importlib.metadata.version("triton") != recipe["triton_version"]:
        raise ValueError("The declared Triton version differs")
    compilers = {"cc": compiler_identity("CC", "gcc"),
                 "cxx": compiler_identity("CXX", "c++")}
    os.environ.update(CUDA_VISIBLE_DEVICES="", CUDA_HOME=str(cuda_home.resolve()),
                      TORCH_CUDA_ARCH_LIST=recipe["cuda_architectures"],
                      TORCH_EXTENSIONS_DIR=str(output.resolve() / "torch-extensions"),
                      TRITON_CACHE_DIR=str(output.resolve() / "triton"), MAX_JOBS=str(jobs),
                      CC=compilers["cc"]["path"], CXX=compilers["cxx"]["path"])
    os.environ.pop("FLASHINFER_GLOBALTIMER_LIBRARY", None)

    import torch
    if torch.__version__ != recipe["torch_version"]:
        raise ValueError("The declared PyTorch runtime differs")
    from flashinfer.utils import get_globaltimer_kernel
    from triton.backends.nvidia import driver
    from triton.runtime.build import _build, platform_key
    from triton.runtime.cache import get_cache_manager

    output.mkdir(parents=True, exist_ok=False)
    get_globaltimer_kernel.cache_clear()
    # Building imports the host extension; calling the returned function would launch a kernel.
    if get_globaltimer_kernel() is None:
        raise RuntimeError("The required timer extension did not build")
    timer = output / "torch-extensions" / TIMER_MODULE / (TIMER_MODULE + ".so")
    compiled = output / "compiled"
    compiled.mkdir()
    timer = Path(shutil.copy2(timer, compiled / timer.name))

    source = Path(driver.__file__).with_name("driver.c")
    helper = Path(_build(DRIVER_MODULE, str(source), str(compiled),
                         [str(cuda_stubs.resolve())], driver.include_dirs, ["cuda"], []))
    # Use Triton's native cache key, but do not load its CUDA driver extension.
    cache_key = hashlib.sha256((source.read_text() + platform_key()).encode("utf-8")).hexdigest()
    cached_helper = Path(get_cache_manager(cache_key).put(helper.read_bytes(), helper.name))
    files = {str(path.relative_to(output.resolve())):
             {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in (timer.resolve(), cached_helper.resolve())}
    manifest = {"schema": SCHEMA, "recipe_sha256": digest(recipe_path), "files": files,
                "host": {"python": platform.python_version(), "platform": platform_key(),
                         "compilers": compilers}}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Built both B200 helper libraries without GPU execution")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cuda-stubs", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(arguments.recipe, arguments.output, arguments.cuda_home,
          arguments.cuda_stubs, arguments.jobs)


if __name__ == "__main__":
    main()
