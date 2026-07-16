# SAP 文単位分割 Kaldi 作成処理

このディレクトリには、SAP processed データから Kaldi 形式を作成し、30秒以上の長い発話を分割するためのスクリプトを置いている。

処理の方針は次の通り。

1. 元データから通常の Kaldi 形式と MFA 用 corpus を作る
2. 元発話全体に Montreal Forced Aligner を実行する
3. `*.origin.wrd` の句読点を使って文単位に分割する
4. 文単位にしても30秒以上の発話だけ `mfa segment` でさらに分割する
5. 分割後の発話を再度 MFA align し、音素ラベルを取得する
6. 最終 Kaldi と 50Hz の `utt2phones_50hz` を作る

## 実行コマンド

通常は以下だけを実行すればよい。

```bash
cd /home/hojo/espnet_dsu/egs2/sap/asr2
local/run_sap_sentence_split_kaldi.sh --splits train dev test1 test2
```

特定の split だけ処理する場合:

```bash
local/run_sap_sentence_split_kaldi.sh --splits test1 test2
```

別の SAP processed データを使う場合:

```bash
local/run_sap_sentence_split_kaldi.sh \
  --base /path/to/SAP0430_processed \
  --splits train dev test1 test2
```

既に元発話の MFA align が済んでいて、`mfa/aligned/<split>_textgrid` を再利用したい場合:

```bash
local/run_sap_sentence_split_kaldi.sh \
  --splits test1 test2 \
  --skip-original-align
```

## 入力として必要なファイル

`--base` で指定するディレクトリには、少なくとも以下が必要。

```text
manifest/<split>.tsv
manifest/<split>.origin.wrd
manifest/<split>.wrd.without.parentheses
data/processed/<split>/*.wav
```

ある場合は、話者IDの取得に以下も使う。

```text
manifest/<split>.transcripts.tsv
```

`*.transcripts.tsv` がない場合は、発話IDから話者IDを推定する。

## 出力

最終出力は以下に作られる。

```text
<base>/kaldi_sentence_first_mfa_with_unaligned_short/<split>/
```

主なファイル:

```text
wav.scp
segments
text
utt2spk
spk2utt
utt2phones_50hz
utt2num_frames_50hz
missing_phone_labels
```

`utt2phones_50hz` は 20ms ごと、つまり 50Hz の音素ラベル列である。MFA で音素ラベルが取れなかった発話は、発話IDだけの行にする。

例:

```text
utt_id AH0 N SIL ...
utt_without_phone_label
```

## 中間出力

処理中に以下の中間データも作成される。

```text
<base>/kaldi/<split>/
<base>/mfa/corpus/<split>/
<base>/mfa/aligned/<split>_textgrid/
<base>/kaldi_sentence_first_base/<split>/
<base>/mfa/corpus_sentence_first_long_for_segment/<split>/
<base>/mfa/corpus_sentence_first_segmented_for_align/<split>/
<base>/mfa/aligned/sentence_first_segmented_long/<split>_textgrid/
<base>/mfa/analysis_sentence_first/
```

`kaldi_sentence_first_base` は、文単位分割後、かつ30秒超の文を `mfa segment` する前の中間 Kaldi である。学習に使う最終版は `kaldi_sentence_first_mfa_with_unaligned_short`。

## 各スクリプトの役割

```text
run_sap_sentence_split_kaldi.sh
```

一連の処理をまとめて実行する wrapper。

```text
prepare_original_kaldi_and_mfa_corpus.py
```

`manifest` と wav から元の `kaldi/<split>` と `mfa/corpus/<split>` を作る。

```text
sap_sentence_split_utils.py
```

文分割、TextGrid 読み取り、単語境界との対応付けなどの共通処理。

```text
build_sentence_split_base_and_long_corpus.py
```

MFA align 済み TextGrid を使って文単位に分割する。30秒超の文は wav clip として切り出し、`mfa segment` 用 corpus に出す。

```text
export_mfa_segment_db_to_align_corpus.py
```

`mfa segment` の DB から分割結果を取り出し、再 align 用の TextGrid corpus を作る。

```text
build_final_segmented_kaldi_with_phones.py
```

文単位 Kaldi と `mfa segment` 結果を結合し、最終 Kaldi を作る。MFA の phone tier から `utt2phones_50hz` も作る。

## MFA 設定

デフォルトでは以下を使う。

```text
conda env: aligner
dictionary: english_us_arpa
acoustic model: english_us_arpa
g2p model: english_us_mfa
```

変更する場合:

```bash
local/run_sap_sentence_split_kaldi.sh \
  --splits test1 test2 \
  --conda-env aligner \
  --dictionary english_us_arpa \
  --acoustic-model english_us_arpa \
  --g2p-model english_us_mfa
```

## 確認方法

30秒超の発話が残っていないか確認する例:

```bash
base=/home/hojo/dataset/SAP0430_processed
for s in train dev test1 test2; do
  awk -v s="$s" '{d=$4-$3; if(d>m){m=d; u=$1}; if(d>30)c++}
    END{printf "%s max=%.3f over30=%d %s\n", s, m, c+0, u}' \
    "$base/kaldi_sentence_first_mfa_with_unaligned_short/$s/segments"
done
```

Kaldi の主要ファイルの行数確認:

```bash
base=/home/hojo/dataset/SAP0430_processed
for s in train dev test1 test2; do
  for f in text segments utt2spk utt2phones_50hz utt2num_frames_50hz; do
    wc -l "$base/kaldi_sentence_first_mfa_with_unaligned_short/$s/$f"
  done
done
```

集計は以下にも出る。

```text
<base>/mfa/analysis_sentence_first/sentence_first_final_summary.tsv
<base>/mfa/analysis_sentence_first/sentence_first_final_source_summary.tsv
```

## 注意点

- 文境界は `*.origin.wrd` の句読点を使う。
- Kaldi の `text` に入れる語列は `*.wrd.without.parentheses` を使う。
- 元発話の MFA align に失敗した短い発話も、30秒以下なら最終 Kaldi に残す。
- MFA で phone tier が得られない発話は、`utt2phones_50hz` に発話IDだけを書く。
- `--skip-original-align` は、既存の `mfa/aligned/<split>_textgrid` が正しい場合だけ使う。
