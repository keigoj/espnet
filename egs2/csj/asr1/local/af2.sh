#!/bin/sh

weights_add=(0.9)  
weights=(0.7 0.9)
for add in ${weights_add[@]}
do
    for sub1 in ${weights[@]}
    do
        for sub2 in ${weights[@]}
        do
            ./run.sh \
            --stage 12 --stop-stage 13 \
            --asr_exp "exp/asr_asr_transformer_aps" \
            --asr_exp_2 "exp/asr_asr_transformer_sps" \
            --lm_exp "exp/lm_transformer_cejc" \
            --lm_exp_sub "exp/lm_transformer_aps" \
            --lm_exp_sub_2 "exp/lm_transformer_sps" \
            --inference_tag "hojo/lm_transformer/cejc/aps_sps/af2/cejc"$add"_aps"$sub1"_sps"$sub2"" \
            --inference_args "--lm_weight $add --lm_weight_sub $sub1 --lm_weight_sub2 $sub2" \
            --test_sets "cejc_rm/test_1h"
        done
    done
done
