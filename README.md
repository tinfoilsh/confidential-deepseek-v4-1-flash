# confidential-deepseek-v4-1-flash

vLLM 0.31.0 image for DeepSeek-V4.1-Flash on 4x NVIDIA B300, 1M-token context,
text and image input.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
(DSpark speculative decoding enabled). Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a` and served
  from a verified model pack.
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

The image seeds 198 measured FlashInfer choices into the matching native
configuration namespace on an empty cache. Existing tuning is preserved.
FlashInfer validates GPU and library metadata before loading the preset, and
configuration changes select a different namespace. Cold-cache profiling policy
is retained so startup can reuse these choices under CC.

The configuration prioritizes interactive coding at concurrency 1–8. Static-five
verification trades some throughput at concurrency 32 and short-prompt
time-to-first-token for those gains.

Run the transfer and tactic checks on a B300 with `VLLM_DISABLE_UVA=1`,
`VLLM_TINFOIL_CC_OPTIMIZATIONS=1`, `VLLM_TINFOIL_DEEPSEEK_TARGET_TACTIC=1`, and
`DEEPSEEK_CC_TRANSFER_POLICY=B` in a virtual environment with access to the
image's Python packages: `python -m pytest tests/`.
