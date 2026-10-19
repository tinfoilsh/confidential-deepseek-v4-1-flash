# Declarative B200 kernel inputs

This is a CPU build prototype. The production Dockerfile still consumes the
qualified `b200-kernels-v4` archive. These outputs have not been executed on a
GPU and are not a qualified replacement for that archive.

`choices.json` is the original 198-entry FlashInfer selection, byte-identical
to the qualified cache. Its hash matches `selected_cache_sha256` in
`../build.json`. The 196 inherited records retain their incomplete measurement
metadata; recording them in Git does not supply the missing timer provenance.

`mxfp8.json` declares all 264 cached CuTe MXFP8 specializations, the B200 target,
source hashes, and compiler package versions. It records active-cluster
capacities explicitly instead of querying a build host's GPU. The builder uses
the pinned FlashInfer compilation helper with synthetic tensors and exports
objects directly, without loading or executing them on a GPU.

After applying the repository's shared and B200 patches to the pinned runtime:

```sh
python scripts/b200_compile_mxfp8.py --output /tmp/mxfp8-build
```

Use the pinned image or a virtual environment with the declared compiler
versions. The output directory must be new. Source or compiler mismatches fail
before compilation. The output includes objects, native cache metadata, and a
build manifest with the input recipe hash and each generated file's hash.

The initial CPU experiment rebuilt all 264 specializations. Their complete
`.lrodata` sections, including embedded GPU programs, matched the archived
objects. The surrounding host objects differed; whole-object reproducibility
and GPU correctness have not been established.

`triton-example.json` and `b200_compile_triton.py` demonstrate the corresponding
source-based build for one real logit-bias specialization. The recipe declares
its argument types, constants, alignment attributes, compiler options, source
hashes, and exact NVIDIA assembler. It builds through Triton's `ASTSource` API
with an explicit target and no device initialization or kernel launch:

```sh
python scripts/b200_compile_triton.py --output /tmp/triton-build \
  --ptxas /usr/local/cuda/bin/ptxas
```

This requires the pinned CUDA 13.0.88 assembler. The example's GPU instruction
section matches the archived kernel; debugging information contains build-path
differences. Native Triton cache group files contain absolute paths, so release
assembly must build at the intended cache location or explicitly relocate those
records when installing the cache.

The CPU investigation also rebuilt all 274 archived Triton kernels from their
text IR with matching GPU instruction sections. This diagnoses compiler support;
it does not provide the missing Python specialization arguments. The existing
archive does not record those arguments. Triton's specialization serializer and
compilation hooks can record them during future development, with explicit
factory inputs for functions created inside other functions.

The timer extension and Triton driver helper were also compiled from their
existing sources on the CPU host. GPU execution, whole-object reproducibility,
runtime cache hits, and a complete image rebuild remain unverified.

The remaining release work is to complete the Triton source inventory, assemble
the full cache from source build outputs, and qualify that exact image. The 198
FlashInfer selections alone do not describe the separate Triton inventory.
Archived IR is not a substitute for a source-based recipe.
