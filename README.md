# confidential-deepseek-v4-1-flash

vLLM image for DeepSeek-V4.1-Flash on 8x NVIDIA Blackwell, 1M-token context,
text and image input.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
(tensor-parallel strategy, DSpark speculative decoding enabled). Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a` and served
  from a verified model pack.
- Python OpenAI frontend (the recipe's `VLLM_USE_RUST_FRONTEND=1` is not set).
- Engram tables kept on GPU (`--engram-config` with `cpu_offload: false`; the
  default offloads them to host memory).
- `--max-num-seqs 64` and `--override-generation-config` with the model
  card's recommended `temperature` / `top_p` (the checkpoint ships no
  `generation_config.json`).
- FlashInfer cubins baked at build time (the container runs offline).
- Patches in `patches/`, one line each in the header of the patch file.
