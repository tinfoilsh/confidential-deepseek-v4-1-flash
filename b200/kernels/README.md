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

The remaining release work is to declare the Triton source specializations,
build the supporting libraries, assemble the complete cache from those inputs,
and qualify the resulting image. The 198 FlashInfer selections alone do not
describe the separate Triton compilation inventory. Archived Triton IR can be
used for compiler diagnostics, but is not a substitute for a source-based recipe.
