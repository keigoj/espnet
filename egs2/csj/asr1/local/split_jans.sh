#!/bin/bash

# utt_id 抜き出し
awk '{print $1}' /home/hojo/espnet_multi2/egs2/csj/asr1/exp/asr_rnn_aps/proposed/ms1.3_aps0.3_sps0.9/Jnas/dev2/text > dev2.utts
awk '{print $1}' /home/hojo/espnet_multi2/egs2/csj/asr1/exp/asr_rnn_aps/proposed/ms1.3_aps0.3_sps0.9/Jnas/test/text > test.utts

# 出力先
mkdir -p data/jnas/dev2 data/jnas/test

# dev2
utils/filter_scp.pl dev2.utts /home/hojo/dataset/Jnas_nodev/text    > data/jnas/dev2/text
utils/filter_scp.pl dev2.utts /home/hojo/dataset/Jnas_nodev/wav.scp > data/jnas/dev2/wav.scp
utils/filter_scp.pl dev2.utts /home/hojo/dataset/Jnas_nodev/utt2spk > data/jnas/dev2/utt2spk
utils/utt2spk_to_spk2utt.pl data/jnas/dev2/utt2spk > data/jnas/dev2/spk2utt

# test
utils/filter_scp.pl test.utts /home/hojo/dataset/Jnas_nodev/text    > data/jnas/test/text
utils/filter_scp.pl test.utts /home/hojo/dataset/Jnas_nodev/wav.scp > data/jnas/test/wav.scp
utils/filter_scp.pl test.utts /home/hojo/dataset/Jnas_nodev/utt2spk > data/jnas/test/utt2spk
utils/utt2spk_to_spk2utt.pl data/jnas/test/utt2spk > data/jnas/test/spk2utt
