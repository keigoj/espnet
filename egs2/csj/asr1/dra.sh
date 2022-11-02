#!/bin/sh

weights=(0.1 0.3 0.5 0.7 0.9 1.1)
for add in ${weights[@]}
do
    for sub in ${weights[@]}
    do
        ./run.sh \
        --stage 11 --stop-stage 12 \
        --asr_exp "exp/asr_rnn_aps" \
        --asr_exp_add "exp/asr_rnn_sps" \
        --lm_exp "exp/lm_sps" \
        --lm_exp_sub "exp/lm_aps" \
        --lm_exp_sub_2 "exp/lm_aps" \
        --inference_tag "mori/dra_aps/aps"$sub"_sps"$add"" \
        --inference_args "--lm_weight $add --lm_add_weight $sub --lm_add_2_weight 0.0"
    done
done

file="mori_result/dra_asp.csv"
for add in ${weights[@]}
do
    for sub in ${weights[@]}
    do
        cer=`cat "exp/asr_rnn_aps/dra_aps/aps"$sub"_sps"$add"/dev2/test/score_cer/result.txt" | \
        sed -n 57p | \
        awk '{print $10}'`
        dir="aps"$sub"_sps"$add""
        echo -n "$cer, " >> $file
    done
    echo "" >> $file
done