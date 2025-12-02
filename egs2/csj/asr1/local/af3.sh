#!/bin/sh

add_w=(0.5 0.7 0.9 1.1 1.3)
weights=(0.1 0.3)
for add in ${add_w[@]}
do
    for sub in ${weights[@]}
    do
        ./run.sh \
        --stage 12 --stop-stage 13 \
        --asr_exp "exp/asr_asr_transformer_aps" \
        --asr_exp_2 "exp/asr_asr_transformer_sps" \
        --asr_exp_3 "exp/asr_transformer_cv" \
        --lm_exp "exp/lm_transformer_cejc" \
        --lm_exp_sub "exp/lm_transformer_aps" \
        --lm_exp_sub_2 "exp/lm_transformer_sps" \
        --lm_exp_sub_3 "exp/lm_transformer_cv" \
        --inference_tag "hojo/lm_transformer/cejc/aps_sps_cv/af3/cejc"$add"_aps0.1_sps0.1_cv"$sub"" \
        --inference_args "--lm_weight $add --lm_weight_sub 0.1 --lm_weight_sub2 0.1 --lm_weight_sub3 $sub" \
        --test_sets "cejc_rm/test_1h"
    done
done

