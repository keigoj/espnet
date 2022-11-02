#!/bin/sh

weights=(0.7 0.9 1.1)
for add in ${weights[@]}
do
    ./run.sh \
    --stage 11 --stop-stage 12 \
    --asr_exp "exp/asr_rnn_aps" \
    --asr_exp_add "exp/asr_rnn_sps" \
    --lm_exp "exp/lm_mainiti" \
    --lm_exp_sub "exp/lm_aps" \
    --lm_exp_sub_2 "exp/lm_sps" \
    --inference_tag "aps_sps_ctc0.5/ms"$add"" \
    --inference_args "--lm_weight $add --lm_add_weight 0.0 --lm_add_2_weight 0.0"
done

# file="result_pre_ctc0.5.csv"
# weights=(0.0 0.1 0.3 0.5)
# for add in ${weights[@]}
# do
#     cer=`cat "exp/asr_rnn_aps/aps_sps_ctc0.3/ms"$add"/Jnas/test/score_cer/result.txt" | \
#     sed -n 57p | \
#     awk '{print $10}'`
#     dir="ms"$add""
#     echo -n "$cer, " >> $file
# done