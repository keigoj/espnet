#!/bin/sh

weights=(0.1 0.3 0.5 0.7 0.9 1.1)
for add in ${weights[@]}
do
    ./run.sh \
    --stage 12 --stop-stage 13 \
    --asr_exp "exp/asr_asr_transformer_aps" \
    --asr_exp_2 "exp/asr_asr_transformer_sps" \
    --asr_exp_3 "exp/asr_transformer_cv" \
    --lm_exp "exp/lm_transformer_cejc" \
    --inference_tag "hojo/cejc/lm_transformer/aps_sps_cv/sf/cejc"$add"" \
    --inference_args "--lm_weight $add" \
    --test_sets "cejc_rm/test_1h"
done
