#!/usr/bin/env bash
# Evaluate the saved best-valid-accuracy model without restarting training.
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${recipe_dir}"

experiment=exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup0_ep90_en
test -f "${experiment}/18epoch.pth"
test -f "${experiment}/config.yaml"

./run_e2e_4090_amp.sh \
    --stage 12 --stop_stage 13 \
    --skip_data_prep true --skip_train true \
    --asr_exp "${experiment}" \
    --inference_asr_model 18epoch.pth \
    --gpu_inference true --inference_nj 2 \
    --test_sets "org/dev test1 test2" \
    "$@" 2>&1 | tee "${experiment}/eval_18epoch.log"
