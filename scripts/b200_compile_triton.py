"""Compile explicitly declared Triton source specializations without a GPU."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sysconfig

SCHEMA = "tinfoil-b200-triton-v1"
TARGET = {"backend": "cuda", "arch": 100, "warp_size": 32}
COMPILER_SOURCES = {
    "triton/compiler/compiler.py", "triton/compiler/code_generator.py",
    "triton/backends/nvidia/compiler.py", "triton/backends/nvidia/lib/libdevice.10.bc",
}
DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "b200/kernels/triton-example.json"
LOGGER = logging.getLogger("b200_compile_triton")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_recipe(recipe_path):
    recipe = json.loads(recipe_path.read_text())
    if recipe["schema"] != SCHEMA:
        raise ValueError("Unsupported Triton source recipe")
    if recipe["target"] != TARGET:
        raise ValueError("The declared B200 compiler target differs")
    if not COMPILER_SOURCES <= recipe["sources"].keys() or not recipe["specializations"]:
        raise ValueError("The declared source inventory is incomplete")
    for specialization in recipe["specializations"]:
        module_name, function_name = specialization["entrypoint"].split(":")
        if (module_name.replace(".", "/") + ".py" not in recipe["sources"]
                or not function_name.isidentifier()):
            raise ValueError("The kernel entrypoint has no reviewed source pin")
        if specialization["signature"].keys() & specialization["constexprs"].keys():
            raise ValueError("Runtime and constant kernel arguments must be distinct")
    return recipe


def build(recipe_path, output, ptxas):
    recipe = load_recipe(recipe_path)
    if digest(ptxas) != recipe["ptxas_sha256"]:
        raise ValueError("The declared NVIDIA assembler differs")
    if importlib.metadata.version("triton") != recipe["triton_version"]:
        raise ValueError("The declared Triton version differs")
    site = Path(sysconfig.get_path("purelib"))
    for relative, expected in recipe["sources"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Kernel source paths must be contained relative paths")
        if digest(site / path) != expected:
            raise ValueError("The declared Triton kernel source differs: " + relative)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(ptxas.resolve())
    os.environ["TRITON_CACHE_DIR"] = str(output.resolve())

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend

    target = GPUTarget(**recipe["target"])
    if digest(Path(get_ptxas(target.arch).path)) != recipe["ptxas_sha256"]:
        raise ValueError("Triton resolved a different NVIDIA assembler")
    backend = make_backend(target)
    builds = []
    for specialization in recipe["specializations"]:
        module_name, function_name = specialization["entrypoint"].split(":")
        function = getattr(importlib.import_module(module_name), function_name)
        if not isinstance(function, triton.JITFunction):
            raise ValueError("The entrypoint must be an importable Triton JIT function")
        constants = specialization["constexprs"]
        signature = {**specialization["signature"], **dict.fromkeys(constants, "constexpr")}
        if set(signature) != set(function.arg_names):
            raise ValueError("The declared kernel arguments differ from the source")
        attributes = {
            (function.arg_names.index(name),): backend.parse_attr(descriptor)
            for name, descriptor in specialization["attributes"].items()
        }
        source = ASTSource(function, signature, constants, attributes)
        compiled = triton.compile(source, target=target, options=specialization["options"])
        metadata_path, = (Path(path) for name, path in compiled.metadata_group.items()
                          if name.endswith(".json"))
        group_path = metadata_path.with_name("__grp__" + metadata_path.name)
        files = {str(Path(path).relative_to(output.resolve())):
                 {"sha256": digest(Path(path)), "bytes": Path(path).stat().st_size}
                 for path in [*compiled.metadata_group.values(), group_path]}
        builds.append({"entrypoint": specialization["entrypoint"],
                       "compiler_key": compiled.hash, "files": files})
    manifest = {"schema": SCHEMA, "recipe_sha256": digest(recipe_path),
                "ptxas_sha256": digest(ptxas), "builds": builds}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Compiled %d declared Triton specializations without GPU execution", len(builds))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(arguments.recipe, arguments.output, arguments.ptxas)


if __name__ == "__main__":
    main()
