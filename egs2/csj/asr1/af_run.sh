#!/bin/sh

msw=(0.3 0.5 0.7 0.9 1.1)
weights=(0.1 0.3 0.5 0.7 0.9 1.1)
for ms in ${msw[@]}
do
    for aps in ${weights[@]}
    do
        for sps in ${weights[@]}
        do
            ./run.sh \
            --stage 11 --stop-stage 11 \
            --asr_exp "exp/asr_rnn_sps" \
            --asr_exp_add "exp/asr_rnn_aps" \
            --lm_exp "exp/lm_mainiti" \
            --lm_exp_sub "exp/lm_sps" \
            --lm_exp_sub_2 "exp/lm_aps" \
            --inference_tag "ms"$ms"_aps"$aps"_sps"$sps"" \
            --inference_args "--lm_weight $ms --lm_add_weight $sps --lm_add_2_weight $aps"
        done
    done
done


# ./run.sh \
#     --stage 11 --stop-stage 12 \
#     --asr_exp "exp/asr_rnn_aps" \
#     --asr_exp_add "exp/asr_rnn_sps" \
#     --lm_exp "exp/lm_mainiti" \
#     --lm_exp_sub "exp/lm_aps" \
#     --lm_exp_sub_2 "exp/lm_sps" \
#     --inference_tag "test1-5" \
#     --inference_args "--lm_weight 1.1 --lm_add_weight 0.5 --lm_add_2_weight 0.3"

# ./run.sh \
# --stage 12 --stop-stage 12 \
# --asr_exp "exp/asr_rnn_aps" \
# --asr_exp_add "exp/asr_rnn_sps" \
# --lm_exp "exp/lm_mainiti" \
# --lm_exp_sub "exp/lm_aps" \
# --lm_exp_sub_2 "exp/lm_sps" 

#./run.sh --stage 11 --stop-stage 12 --asr_exp "exp/asr_rnn_aps" --asr_exp_add "exp/asr_rnn_sps" --lm_exp "exp/lm_mainiti" --lm_exp_sub "exp/lm_aps" --lm_exp_sub_2 "exp/lm_sps" --inference_tag "dra" --inference_args "--lm_weight 0.9 --lm_add_weight 0.9 --lm_add_2_weight 0.0"
#./run.sh --stage 11 --stop-stage 12 --asr_exp "exp/asr_rnn_aps" --asr_exp_add "exp/asr_rnn_sps" --lm_exp "exp/lm_mainiti" --lm_exp_sub "exp/lm_aps" --lm_exp_sub_2 "exp/lm_sps" 

#&& send-slack-msg "finish" || send-slack-msg "failure"