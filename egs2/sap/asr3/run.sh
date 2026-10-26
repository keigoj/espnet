#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# Differentiable tokenizer initialized from SAP K-means centroids, with
# healthy-speech centroids kept separately for optional OT regularization.
nclusters=2000
ssl_config=conf/train_ssl_kmeans_wavlm_large.yaml
# ssl_tag=wavlm_large_l21_gumbel_km${nclusters}_ot_uniform
ssl_tag=wavlm_large_l21_gumbel_km${nclusters}_sap_init_div05
src_lang=${ssl_tag}
tgt_lang=en

train_set=train
valid_set=dev
test_sets="test1 test2"

src_case=rm  # rm: deduplicate consecutive units, ts: keep every frame unit
tgt_case=ts
src_nbpe=6000
tgt_nbpe=5000

asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml

./asr3.sh \
    --ngpu 1 \
    --extract_ngpu 1 \
    --nj 4 \
    --nclusters "${nclusters}" \
    --ssl_config "${ssl_config}" \
    --ssl_tag "${ssl_tag}" \
    --src_lang "${src_lang}" \
    --tgt_lang "${tgt_lang}" \
    --src_case "${src_case}" \
    --tgt_case "${tgt_case}" \
    --src_token_type bpe \
    --src_nbpe "${src_nbpe}" \
    --tgt_token_type bpe \
    --tgt_nbpe "${tgt_nbpe}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --asr_tag "${src_lang}_${tgt_lang}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --gpu_inference false \
    "$@"
