#!/usr/bin/env bash
# RTX 4090 preset: freeze WavLM/codebook through epoch 30, then jointly train.
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Set before path.sh or Python imports initialize the CUDA allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "RTX 4090 AMP: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" >&2

exec bash "${recipe_dir}/run_e2e.sh" \
    --ngpu 1 --e2e_mode full --warmup_epochs 30 \
    --asr_config conf/train_e2e_gumbel_kmeans_wavlm_large_4090_amp_1200k.yaml \
    "$@"
