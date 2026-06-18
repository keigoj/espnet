#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="dys+ctl_trim/train"
valid_set="dys+ctl_trim/dev"
test_sets="dys+ctl_trim/test"

asr_config=conf/train_asr_hubert_large_10h_finetuning.yaml
inference_config=conf/decode.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --token_type char \
    --max_wav_duration 20 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --feats_normalize none \
    --inference_asr_model "valid.loss.ave.pth" "$@"
