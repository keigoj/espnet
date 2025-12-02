#!/bin/sh

weights=(0.1 0.3 0.5 0.7 0.9 1.1)
for add in ${weights[@]}
do
    ./run.sh \
    --stage 12 --stop-stage 13 \
    --asr_exp "exp/asr_transformer_cv" \
    --lm_exp "exp/lm_transformer_cejc" \
    --inference_tag "cejc/sf/cejc"$add"" \
    --inference_args "--lm_weight $add"  \
    --test_sets "cejc_rm/test_1h"
done

for add in ${weights[@]}
do
    ./run.sh \
    --stage 12 --stop-stage 13 \
    --asr_exp "exp/asr_transformer_aps+sps+cv" \
    --lm_exp "exp/lm_transformer_cejc" \
    --inference_tag "cejc/sf/cejc"$add"" \
    --inference_args "--lm_weight $add"  \
    --test_sets "cejc_rm/test_1h"
done
