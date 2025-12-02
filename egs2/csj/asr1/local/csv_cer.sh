# ===== デフォルト値 =====
file=""          # 出力CSVファイル ex) results/test.csv
method="af"      # sf | dra | af2 | af3
exp_dir=""       # 実験ディレクトリ ex) exp/asr_rnn_cejc/cejc/dra
test_path=""     # テストディレクトリへの相対パス ex) cejc_rm/test_1h
add=""
sub=""
sub2=""
sub3=""
result_line=

# ===== 引数処理 =====
usage() {
  cat <<'USAGE'
Usage: run.sh [-m METHOD] [-e EXP_DIR] [-t TEST_PATH] [-a ADD] [-s SUB] [-2 SUB2] [-3 SUB3]
              [-o OUT_CSV] [-l LINE]
  -m  手法: sf | dra | af2 (default: dra)
  -e  実験ディレクトリ (default: exp/asr_rnn_cejc/cejc/dra)
  -t  テストディレクトリへの相対パス (default: cejc_rm/test_1h)
  -a  加算側プレフィックス名 (default: cejc)
  -s  減算側1プレフィックス名 (default: cejc_sub)
  -2  減算側2プレフィックス名（af2用）(default: sps)
  -3  減算側3プレフィックス（必要なら）(default: cejc)
  -o  出力CSVファイル (default: results/test.csv)
  -l  CERが載っている行番号 (default: 351)
USAGE
}

while getopts "m:e:t:a:s:2:3:o:l:h" opt; do
  case "$opt" in
    m) method="$OPTARG" ;;
    e) exp_dir="$OPTARG" ;;
    t) test_path="$OPTARG" ;;
    a) add="$OPTARG" ;;
    s) sub="$OPTARG" ;;
    2) sub2="$OPTARG" ;;
    3) sub3="$OPTARG" ;;
    o) file="$OPTARG" ;;
    l) result_line="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

min=100.0
best_add=""
best_sub=""
best_sub2=""
best_sub3=""

mkdir -p $(dirname "$file")

echo "Data: $test_path" >> "$file"
echo "Method: $method" >> "$file"
echo "Experiment Directory: $exp_dir" >> "$file"

if [ "$method" == "sf" ]; then
    weights=(0.1 0.3 0.5 0.7 0.9 1.1)
    for add_w in ${weights[@]}
    do
        cer=`cat "$exp_dir/"$add""$add_w"/$test_path/score_cer/result.txt" | \
        sed -n "${result_line}p" | \
        awk '{print $(NF-2)}'`
        echo -n "$cer, " >> $file
        if [ `echo "$min > $cer" | bc` == 1 ] ; then
            min=$cer
            best_add=$add_w
        fi
    done
elif [ "$method" == "dra" ]; then
    weights=(0.1 0.3 0.5 0.7 0.9 1.1)
    for add_w in ${weights[@]}
    do
        for sub_w in ${weights[@]}
        do
            cer=`cat "$exp_dir/$add"$add_w"_$sub"$sub_w"/$test_path/score_cer/result.txt" | \
            sed -n "${result_line}p" | \
            awk '{print $(NF-2)}'`
            echo -n "$cer, " >> $file
            if [ `echo "$min > $cer" | bc` == 1 ] ; then
                min=$cer
                best_add=$add_w
                best_sub=$sub_w
            fi
        done
        echo "" >> $file
    done
elif [ "$method" == "af2" ]; then
    weights_add=(0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7)
    weights=(0.1 0.3 0.5 0.7 0.9 1.1)
    for add_w in ${weights_add[@]}
    do
        for sub_w in ${weights[@]}
        do
            for sub_w2 in ${weights[@]}
            do
                cer=`cat "$exp_dir/$add"$add_w"_$sub"$sub_w"_$sub2"$sub_w2"/$test_path/score_cer/result.txt" | \
                sed -n "${result_line}p" | \
                awk '{print $(NF-2)}'`
                echo -n "$cer, " >> $file
                if [ `echo "$min > $cer" | bc` == 1 ] ; then
                    min=$cer
                    best_add=$add_w
                    best_sub=$sub_w
                    best_sub2=$sub_w2
                fi
            done
            echo "" >> $file
        done
        echo "" >> $file
    done
elif [ "$method" == "af3" ]; then
    weights=(0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5)
    for add_w in ${weights[@]}
    do
        for sub_w in ${weights[@]}
        do
            cer=`cat "$exp_dir/$add"$add_w"_aps0.1_sps0.1_$sub"$sub_w"/$test_path/score_cer/result.txt" | \
            sed -n "${result_line}p" | \
            awk '{print $(NF-2)}'`
            echo -n "$cer, " >> $file
            if [ `echo "$min > $cer" | bc` == 1 ] ; then
                min=$cer
                best_add=$add_w
                best_sub=$sub_w
            fi
        done
        echo "" >> $file
    done
else
    echo "Unknown method: $method"
    exit 1
fi

echo "" >> $file
echo "" >> $file

case "$method" in
  sf)
    echo "minimum : ${min} (add_w=${best_add})" >> "$file"
    ;;
  dra)
    echo "minimum : ${min} (add_w=${best_add}, sub_w=${best_sub})" >> "$file"
    ;;
  af2)
    echo "minimum : ${min} (add_w=${best_add}, sub1_w=${best_sub}, sub2_w=${best_sub2})" >> "$file"
    ;;
  af3)
    echo "minimum : ${min} (add_w=${best_add}, sub_w=${best_sub})" >> "$file"
    ;;
esac


