# confidential-deepseek-v4-1-flash

vLLM 0.31.0 image for DeepSeek-V4.1-Flash-1017 on 4x NVIDIA B300, 1M-token context,
text and image input.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
(DSpark speculative decoding enabled). Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10` and served
  from a verified model pack under the `deepseek-v4-1-flash` API model name.
- Python OpenAI frontend (the recipe's `VLLM_USE_RUST_FRONTEND=1` is not set).
- Engram tables kept on GPU (`--engram-config` with `cpu_offload: false`; the
  default offloads them to host memory).
- TP4 with 56 CPUs and 768 GiB RAM, FP8 MLA KV cache, and static five-token
  DSpark verification. CPU KV offloading is disabled.
- Explicit CPU/GPU transfers for CC, private text-input staging, and deferred
  reply readback. Unsupported request paths use the native implementation.
- Native expert tactics selected for B300 target batches of 6–24 rows;
  draft batches and larger target batches keep native selection.
- `--max-num-seqs 64` and `--override-generation-config` with the model
  card's recommended `temperature` / `top_p` (the checkpoint ships no
  `generation_config.json`).
- FlashInfer cubins verified and baked at build time (the container runs offline).
- Released reasoning-effort mappings retained across the vLLM upgrade.

The build fetches the measured FlashInfer preset from a pinned commit and
verifies its SHA-256. The image seeds its 198 choices into the matching native
configuration namespace on an empty cache. Existing tuning is preserved.
FlashInfer validates GPU and library metadata before loading the preset, and
configuration changes select a different namespace. Cold-cache profiling policy
is retained so startup can reuse these choices under CC.

The configuration prioritizes interactive coding at concurrency 1–8. Static-five
verification trades some throughput at concurrency 32 and short-prompt
time-to-first-token for those gains.

The B200/TDX variant uses `Dockerfile.b200` and `tinfoil-config.b200.yml`, retaining
the qualified `DeepSeek-V4.1-Flash@dba1be0a` checkpoint. It
reuses compact native prefill graphs, captured target preparation and sampling,
and processor metadata mirrors. Its 198 measured kernel choices and compiled
artifacts come from a checksum-pinned [public build artifact](https://github.com/tinfoilsh/confidential-deepseek-v4-1-flash/releases/tag/b200-kernels-v2)
in this repository; generated binaries remain outside Git. Runtime source and
build inputs are maintained here. Benchmarks, experiment validation, and their
results live in the private CC lab.
The [B200 patch review guide](b200/patches/README.md) describes the ordered
inference changes and their dependencies.
The runtime requires the qualified TP4 configuration, normal CC mode, driver
595.71.05, model revision, precision, software, and measurement policy. A
compatibility mismatch fails before loading weights instead of retuning at boot.

Build the variant directly; the Dockerfile downloads and verifies its bundle:

```sh
docker build -f Dockerfile.b200 -t deepseek-b200 .
```

Run the **Tinfoil Release** workflow from the source branch with a base `version` and a `variant`
(`b200` or `b300`). For example, `v0.0.14` with `b200` publishes GitHub release
`v0.0.14+b200` and container tag `0.0.14-b200`; `v0.0.15-rc.1` produces
`v0.0.15-rc.1+b200` as a prerelease. The workflow selects the variant's Dockerfile
and pins its image digest into the tagged `tinfoil-config.yml` before measurement
and attestation. Both variants use the same source branch and can be released
independently. Published tags are never replaced, and releases do not promote
GitHub latest or publish shared container aliases. Freshness renewal selects
the full release tag, including its variant.

Both variants build from public inputs. The release workflow uses the standard
repository token; no private-lab credential is required. Kernel checksums and
compatibility checks remain enforced by the build and runtime.
