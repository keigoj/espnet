#!/bin/sh

file="result_ctc0.5.csv"
weights=(0.1 0.3 0.5 0.7 0.9 1.1)
for ms in ${weights[@]}
do
    for aps in ${weights[@]}
    do
        for sps in ${weights[@]}
        do
            cer=`cat "exp/asr_rnn_sps/ms"$ms"_aps"$aps"_sps"$sps"/Jnas/test/score_cer/result.txt" | \
            sed -n 57p | \
            awk '{print $10}'`
            dir="ms"$ms"_aps"$aps"_sps"$sps""
            echo -n "$cer, " >> $file
        done
        echo "" >> $file
    done
    echo "" >> $file
done

file="result_aps_sps_sf_ctc0.3_.csv"
weights=(0.0 0.1 0.3 0.5 0.7 0.9 1.1)
for add in ${weights[@]}
do
    cer=`cat "exp/asr_rnn_aps/aps_sps_ctc0.3/ms"$add"/Jnas/test/score_cer/result.txt" | \
    sed -n 57p | \
    awk '{print $10}'`
    dir="ms"$add""
    echo -n "$cer, " >> $file
done