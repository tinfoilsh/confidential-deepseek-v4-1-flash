# Declarative B200 kernel inputs

This is a CPU build prototype. The production Dockerfile still consumes the
qualified `b200-kernels-v4` archive. These outputs have not been executed on a
GPU and are not a qualified replacement for that archive.

The optional build target compiles all declared inputs inside the pinned vLLM
base image and exports artifacts to a new local directory:

```sh
docker build --platform linux/amd64 -f b200/kernels/Dockerfile \
  --output type=local,dest=/tmp/b200-source-build .
```

Compilation steps run without network access or GPU devices. This exports 264
CuTe kernels, one Triton example, and both helper libraries. It does not build a
complete inference image. The production release workflow does not use it.

With `pyelftools==0.33` installed in a virtual environment, compare the exported
payloads against the existing public `b200-kernels-v4` archive:

```sh
python scripts/b200_compare_kernels.py \
  --reference /tmp/b200-normal-cc-kernels-r4.tar.gz \
  --artifacts /tmp/b200-source-build --report /tmp/b200-comparison.json
```

The comparison checks the reference archive's checksum and all CuTe objects.
For Triton, it compares each supplied kernel against reference kernels with the
same name and reports both the supplied and total reference counts. A matching
GPU payload does not verify runtime argument specialization, host wrapper
behavior, cache reuse, or correctness on hardware.

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

The CPU experiments rebuilt all 264 specializations, including inside the
pinned release environment. Their complete
`.lrodata` sections, including embedded GPU programs, matched the archived
objects. The surrounding host objects differed; whole-object reproducibility
and GPU correctness have not been established. The native cache metadata
matched the reference when built at the image's installed package paths.

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

`helpers.json` and `b200_compile_helpers.py` build the timer extension and Triton
driver helper from their existing sources. They require a coherent CUDA 13.0
development toolkit, including its link-time driver stub and `libcudart.so`
linker name. The stub is used only for linking; the driver helper is not loaded.

```sh
python scripts/b200_compile_helpers.py --output /tmp/helpers-build \
  --cuda-home /usr/local/cuda-13.0 \
  --cuda-stubs /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs
```

The helper manifest records the host compiler identities and both library
hashes. The Triton helper uses its native cache layout. All three builders are
standalone commands intended for fresh processes and new output directories.
Source hashes and selected compiler versions are checked, but these recipes are
not a complete dependency lock; release assembly must also pin the build image.
CuTe's host wrapper export uses the build CPU. The pinned compiler rejects an
explicit host target with TVM-FFI export, so pinning a container alone does not
establish portable or reproducible host wrappers.
GPU execution, whole-object reproducibility, runtime cache hits, and a complete
image rebuild remain unverified.

The remaining release work is to complete the Triton source inventory, assemble
the full cache from source build outputs, and qualify that exact image. The 198
FlashInfer selections alone do not describe the separate Triton inventory.
Archived IR is not a substitute for a source-based recipe.
