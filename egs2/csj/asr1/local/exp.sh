#!/bin/sh


./run.sh --stage 12 --stop-stage 13 \
    --asr_exp "exp/asr_asr_transformer_aps" --asr_exp_2 "exp/asr_asr_transformer_sps" --asr_exp_3 "exp/asr_transformer_cv" \
    --lm_exp "exp/lm_transformer_cejc" --lm_exp_sub "exp/lm_transformer_aps" --lm_exp_sub_2 "exp/lm_transformer_sps" --lm_exp_sub_3 "exp/lm_transformer_cv" \
    --inference_tag "hojo/lm_transformer/cejc/aps_sps_cv/af3/cejc1.1_aps0.5_sps0.1_cv0.1" --inference_args "--lm_weight 1.1 --lm_weight_sub 0.5 --lm_weight_sub2 0.1 --lm_weight_sub3 0.1" \
    --test_sets "cejc_rm/test_1h_2"



# ./run.sh --stage 12 --stop-stage 13 \
#     --asr_exp "exp/asr_asr_transformer_aps" --asr_exp_2 "exp/asr_asr_transformer_sps" --asr_exp_3 "exp/asr_transformer_cv" \
#     --lm_exp "exp/lm_transformer_mainiti" --lm_exp_sub "exp/lm_transformer_aps" --lm_exp_sub_2 "exp/lm_transformer_sps" --lm_exp_sub_3 "exp/lm_transformer_cv" \
#     --inference_tag "hojo/jnas/aps_sps_cv/af3/ms1.7_aps0.5_sps0.1_cv0.3" --inference_args "--lm_weight 1.7 --lm_weight_sub 0.5 --lm_weight_sub2 0.1 --lm_weight_sub3 0.3" \
#     --test_sets "jnas/test"

