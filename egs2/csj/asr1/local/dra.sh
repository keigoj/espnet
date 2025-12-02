#!/bin/sh

weights_add=(0.1 0.3 0.5 0.7)
weights_sub=(0.1 0.3 0.5)

for add in ${weights_add[@]}
do
    for sub in ${weights_sub[@]}
    do
        ./run.sh \
        --stage 12 --stop-stage 13 \
        --asr_exp "exp/asr_asr_transformer_aps+sps" \
        --lm_exp "exp/lm_transformer_cejc" \
        --lm_exp_sub "exp/lm_transformer_aps+sps_2" \
        --inference_tag "hojo/cejc/lm_transformer/dra/cejc"$add"_aps+sps"$sub"" \
        --inference_args "--lm_weight $add --lm_weight_sub $sub" \
        --test_sets "cejc_rm/test_1h"
    done
done

