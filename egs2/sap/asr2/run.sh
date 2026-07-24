#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="hubert_asr/24"  # use model_type/layer_index eg. hubert_asr/12, wavlm_large/21
nclusters=2000
kmeans_method="phone-based" # base / phone-based / anchore-based
lambda_anchor=1.0

lambda_tag=
if [ -n "${lambda_anchor}" ]; then
    lambda_tag="_lam${lambda_anchor//./p}"
fi

src_lang=$(echo "${kmeans_feature}_${kmeans_method}${lambda_tag}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set="train"
train_dev="dev"
test_sets="test1 test2"

asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2.sh \
    --kmeans_opts "--batch_bins 3200000 --nj 4" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "dump/raw/${train_set}_sp/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "dump/raw/${train_set}_sp/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "dump/raw/${train_set}_sp/text.${tgt_case}.${tgt_lang}" \
    --portion 0.05 \
    --gpu_inference false \
    --kmeans_method "${kmeans_method}" \
    ${lambda_anchor:+--lambda_anchor "${lambda_anchor}"} "$@"
