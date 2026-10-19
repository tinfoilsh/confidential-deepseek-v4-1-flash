# syntax=docker/dockerfile:1.6
FROM vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b AS compile

ENV CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
COPY patches/ /recipe/shared-patches/
COPY b200/patches/ /recipe/b200-patches/
RUN --network=none set -eu; \
    test "$VLLM_BUILD_COMMIT" = "db9527a46873454610df6dbedf79a36d6bf1a7f6"; \
    cd /; \
    for patch_file in /recipe/shared-patches/*.patch /recipe/b200-patches/*.patch; do \
        patch -p1 --batch --forward --no-backup-if-mismatch --fuzz=0 < "$patch_file"; \
    done

COPY b200/kernels/ /recipe/b200/kernels/
COPY scripts/b200_compile_mxfp8.py scripts/b200_compile_triton.py scripts/b200_compile_helpers.py /recipe/scripts/
RUN --network=none python3 /recipe/scripts/b200_compile_mxfp8.py --output /output/mxfp8 && \
    python3 /recipe/scripts/b200_compile_triton.py --output /output/triton \
        --ptxas /usr/local/cuda-13.0/bin/ptxas && \
    python3 /recipe/scripts/b200_compile_helpers.py --output /output/helpers \
        --cuda-home /usr/local/cuda-13.0 \
        --cuda-stubs /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs

FROM scratch AS artifacts
COPY --from=compile /output/ /
