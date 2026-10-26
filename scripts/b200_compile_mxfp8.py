"""Build declared B200 MXFP8 kernels from pinned sources without a GPU."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
from pathlib import Path
import sysconfig

SCHEMA = "tinfoil-b200-mxfp8-v1"
ARCHITECTURE = "sm_100a"
SCALE_VECTOR_SIZE = 32
DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "b200/kernels/mxfp8.json"
DEFAULT_WORKERS = 4
MODULE_NAME = "mxfp8_gemm"
INPUT_ALIGNMENT = 16
DYNAMIC_SCALE_DIMENSION = 1
VARIANT_FIELDS = {"tile", "cluster", "swap_ab", "prefetch", "split_k_slices",
                  "batch_size", "max_active_clusters"}
SOURCE_FILES = {
    "flashinfer/gemm/gemm_base.py", "flashinfer/gemm/gemm_mm_fp4_cute_dsl.py",
    "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py",
    "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_common.py",
    "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py",
    "flashinfer/jit/cute_dsl_core.py",
}
COMPILER_PACKAGES = {
    "flashinfer-python", "apache-tvm-ffi", "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12", "nvidia-cutlass-dsl-libs-cu13",
}
LOGGER = logging.getLogger("b200_compile_mxfp8")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_recipe(path):
    recipe = json.loads(path.read_text())
    if recipe["schema"] != SCHEMA or recipe["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported B200 compilation recipe")
    if set(recipe["sources"]) != SOURCE_FILES or set(recipe["packages"]) != COMPILER_PACKAGES:
        raise ValueError("The source or compiler dependency inventory is incomplete")
    if recipe["output_dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("Unsupported output precision")
    if (recipe["scale_vector_size"] != SCALE_VECTOR_SIZE
            or type(recipe["programmatic_launch"]) is not bool):
        raise ValueError("Unsupported MXFP8 configuration")
    variants = recipe["variants"]
    if not variants:
        raise ValueError("The compilation inventory is empty")
    seen = set()
    for variant in variants:
        if set(variant) != VARIANT_FIELDS:
            raise ValueError("A kernel specialization has missing or unknown fields")
        for name in ("tile", "cluster"):
            dimensions = variant[name]
            if (not isinstance(dimensions, list) or len(dimensions) != 2
                    or any(type(value) is not int or value <= 0 for value in dimensions)):
                raise ValueError("Kernel dimensions must be pairs of positive integers")
        for name in ("split_k_slices", "batch_size", "max_active_clusters"):
            if type(variant[name]) is not int or variant[name] <= 0:
                raise ValueError("Kernel capacities must be positive integers")
        if any(type(variant[name]) is not bool for name in ("swap_ab", "prefetch")):
            raise ValueError("Kernel flags must be booleans")
        if variant["split_k_slices"] > 1 and (
                variant["cluster"] != [1, 1] or not variant["swap_ab"] or variant["prefetch"]):
            raise ValueError("Invalid split-K specialization")
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            raise ValueError("Duplicate kernel specialization")
        seen.add(key)
    return recipe


def verify_sources(recipe, site):
    for relative, expected in recipe["sources"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Source paths must be relative to the installed packages")
        if digest(site / path) != expected:
            raise ValueError("The declared kernel source differs: " + relative)


def verify_environment(recipe):
    for name, expected in recipe["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("The declared compiler package differs: " + name)
    import torch
    if torch.__version__ != recipe["torch_version"]:
        raise ValueError("The declared PyTorch runtime differs")
    verify_sources(recipe, Path(sysconfig.get_path("purelib")))


def compile_variant(arguments):
    recipe, variant, output = arguments
    import cutlass
    import torch
    from flashinfer.gemm.gemm_base import _mxfp8_kernel_disk_name
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _make_blockscaled_gemm_compile_fn
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk import (
        Sm100BlockScaledSplitKGemmKernel,
    )

    tile, cluster = tuple(variant["tile"]), tuple(variant["cluster"])
    scale, launch = recipe["scale_vector_size"], recipe["programmatic_launch"]
    split = variant["split_k_slices"]
    dtype = getattr(torch, recipe["output_dtype"])
    name = _mxfp8_kernel_disk_name(
        (scale, tile, cluster, variant["swap_ab"], variant["prefetch"], launch, dtype, split),
        variant["batch_size"], variant["max_active_clusters"],
    )
    if split == 1:
        kernel = Sm100BlockScaledPersistentDenseGemmKernel(
            scale, tile, cluster, variant["prefetch"], launch)
    else:
        kernel = Sm100BlockScaledSplitKGemmKernel(scale, tile, split, launch)
    output_types = {"bfloat16": cutlass.BFloat16, "float16": cutlass.Float16}
    compile_kernel = _make_blockscaled_gemm_compile_fn(
        kernel, ab_cutlass_dtype=cutlass.Float8E4M3FN, sf_dtype=cutlass.Float8E8M0FNU,
        c_cutlass_dtype=output_types[recipe["output_dtype"]], ab_assumed_align=INPUT_ALIGNMENT,
        swap_ab=variant["swap_ab"], sf_m=DYNAMIC_SCALE_DIMENSION,
        sf_n=DYNAMIC_SCALE_DIMENSION, sf_k=DYNAMIC_SCALE_DIMENSION,
        batch_size=variant["batch_size"], max_active_clusters=variant["max_active_clusters"],
    )
    path = output / (name + ".o")
    # Export directly: the serving cache helper also loads the module onto a GPU.
    compile_kernel().export_to_c(str(path), function_name=MODULE_NAME + "_" + name)
    return path.name, {"sha256": digest(path), "bytes": path.stat().st_size}


def build(recipe_path, output, workers):
    recipe = load_recipe(recipe_path)
    if workers <= 0:
        raise ValueError("The worker count must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUTE_DSL_ARCH"] = recipe["architecture"]
    verify_environment(recipe)
    from flashinfer.gemm.gemm_base import _mxfp8_kernel_cache_files
    from flashinfer.jit.cute_dsl_core import _get_cute_dsl_version, _hash_source_files

    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        files = dict(pool.map(compile_variant,
                              ((recipe, variant, output) for variant in recipe["variants"])))
    metadata = {"arch": recipe["architecture"].replace("_", ""),
                "cute_dsl_version": _get_cute_dsl_version(),
                "source_sha256": _hash_source_files(_mxfp8_kernel_cache_files())}
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {"schema": SCHEMA, "recipe_sha256": digest(recipe_path),
                "files": files, "metadata": metadata}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Compiled %d declared MXFP8 objects without GPU execution", len(files))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(arguments.recipe, arguments.output, arguments.workers)


if __name__ == "__main__":
    main()
