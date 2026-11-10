# SAP の軽度話者を参照する SIR 学習

`data/doc/<speaker>.json` の明瞭度評価から参照話者を選び、同内容の発話ペアを
保存して `WavLM -> Gumbel K-means -> ASR` を学習する。
これは健常音声を使う原論文 §IV・式 (8) を、軽度音声を参照に拡張する実験である。

## 話者の選定基準

- JSON の `Files[].Ratings[]` の `Dimension Description` が明瞭度の項目を使う。
  実データの `Intelligbility` という綴り、末尾空白、`(overall)` も扱う。
- `Level` の有効な値は 1--7。SAP の尺度は1が典型的・影響なし、7が著しく影響あり。
  出典は [SAP担当者の説明資料、スライド24--25](https://publish.illinois.edu/hkim17/files/2024/12/Handout_ASHA2024-1.pdf#page=24)。
- 同一発話に複数の明瞭度評価があれば発話内で平均し、その後話者内で平均する。
  Naturalness や別の評価次元は混ぜない。空値、範囲外、非有限値は除外する。
- **話者平均が2以下**を軽度参照とする。これは本実験の選定基準であり、
  SAP が配布する公式の軽度カテゴリではない。
- 評価欠損を軽度として扱わない。参照は学習話者からのみ選び、dev/testを混ぜない。
  話者平均は評価済み発話からの代理指標で、未評価の個々の発話の重症度は保証しない。

## データ作成

`asr3/local` は `asr2/local` へのリンクなので、実装は
[`../asr2/local/prepare_sir_mild.py`](../asr2/local/prepare_sir_mild.py) にある。
既存の音声整形・長さフィルタ後のデータを入力する。外部の元データは変更しない。

```bash
cd egs2/sap/asr3
python local/prepare_sir_mild.py \
  --doc-dir /mnt/nvme_data02/hojo/datasets/SAP0430_processed/data/doc \
  --train-dir dump/e2e/raw/train_sp \
  --valid-dir dump/e2e/raw/dev \
  --audio-root . \
  --output-dir data/sir_mild_le2 \
  --max-mean-intelligibility 2 --seed 0
```

既存手順の既定パス `/home/hojo/dataset/SAP0430_processed` は現在存在しないため、
上記では実データの配置を明示している。
入力は utterance 単位の `wav.scp`、`text`、`utt2spk`、任意の `utt2uniq`。
`segments` 付きの録音単位入力は、先に既存レシピで音声整形する。
再実行で既存の出力を上書きしない。条件を変える場合は別の出力ディレクトリを使う。

保存されるファイル:

| ファイル | 内容 |
|---|---|
| `speaker_scores.tsv` | train/dev各話者の平均、評価発話数、最小・最大、欠損理由、元docパス |
| `mild_speakers.txt` | 選定された学習話者IDの一覧 |
| `train/healthy.scp`, `valid/healthy.scp` | 各入力発話に対応する参照音声パス |
| `train/sir_pair_mask`, `valid/sir_pair_mask` | 対応ペアあり=1、SIR適用外=0 |
| `train/pairs.tsv`, `valid/pairs.tsv` | 入力・参照の発話ID、話者ID、評価値、転記ハッシュ、適用外の理由 |
| `summary.json` | 選定基準、seed、入力パス、話者数とペア数 |

ペアの主側は、平均が2を超える評価済み話者の発話とする。軽度話者と評価欠損話者も
全件ASR学習に残すが、それらが主側になったときはSIRを適用しない。
主側と参照は、大文字小文字と空白のみを正規化したASR正解語列が一致するものを使う。
疑問文のプロンプトIDだけで対応付けたり、異なる文を似ているという理由で使ったりしない。
ASR用の正解語列は既存の前処理で非流暢性を除いているため、音声が逐語的に完全一致する
保証はない。既存の転記・実験条件をこの準備処理で変更することはない。

参照候補は速度変換前の学習発話から作る。転記ごとに候補話者を一様に選び、
その話者内で候補発話を一様に選ぶ。seed付きの固定ペアであり、epochごとの再抽選はしない。
同じ主発話の速度変換版には同じ参照を使う。ペアなしの行にも短い実音声のパスを置くが、
マスク0なので参照側のfrontendやSIRの計算には入らない。

## 学習

```bash
bash run_e2e_sir.sh
```

今回の実行先はtmuxの `dsu` セッション、`sir-mild-le2` ウィンドウ。
`tmux attach -t dsu` で確認できる。既存ウィンドウの作業には干渉せず、専用ウィンドウを使う。

既存の `run_e2e.sh` とSIR用YAML
[`conf/train_e2e_gumbel_kmeans_wavlm_large_sir.yaml`](conf/train_e2e_gumbel_kmeans_wavlm_large_sir.yaml)
を使う。重みはこのYAMLの **`model_conf.sir_weight: 0.1`** で変更する。
`--sir_weight` を明示した場合だけYAMLの値を上書きする。
SIR導入時に追加した学習条件はこの重みとSIR用入力ストリームだけ。
メモリ調整前の基準設定はA6000×4、batch_bins=12000000、accum_grad=3。
90 epoch、最初の30 epochはfrontendとcodebookを凍結するスケジュールを
維持する。**SIRからtokenizerへの更新も、凍結解除後のepoch31から始まる。**
凍結中もSIR値は計測する。

stage5で5000語BPEを準備し、stage10で統計収集、stage11で学習する。
既存のデータ整形をやり直さず、別の統計・実験ディレクトリへ出力する。

```bash
# 統計収集まで実施済みの場合
bash run_e2e_sir.sh --stage 11

# 統計収集のみ
bash run_e2e_sir.sh --stage 10 --stop_stage 10
```

## GPUメモリとバッチ上限の調整

`run_e2e_sir.sh` は、可変長バッチのメモリ断片化を抑えるため、
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を既定で設定する。
利用者が環境変数を指定済みの場合はその値を使う。

今回の実験の調整処理は、tmux `dsu` の `sir-memory-tune` ウィンドウで実行した。
エポック末のチェックポイント保存後に元の学習を停止し、実データの長い主音声、
長いSIR参照、多数のペアを含むバッチで、凍結中・凍結解除後をそれぞれ測定する。
メモリ上限に収まる候補の処理速度を比較し、4 GPU DDPで学習・検証・保存まで確認した
候補を採用する。モデル・optimizer・schedulerの状態を引き継いで再開し、
最初の30 epochとepoch31以降で必要に応じて別の `batch_bins` を使う。

結果は実験ディレクトリの `memory_tuning_20261108/selected.json`、
進行状況は同ディレクトリの `status.json` と `controller.log` に保存する。
採用値はSIR用YAMLの `batch_bins` に反映される。
勾配累積は3回を維持するため、バッチ上限の変更で実効バッチサイズとepochあたりの
更新回数は変わる。検証の `valid_batch_bins` は基準値12000000を維持する。

2026-11-08の採用値は次のとおり。15 epochのチェックポイントから再開した。

| 学習区間 | `batch_bins` | 1 epochのバッチ数 |
|---|---:|---:|
| 調整前 | 12000000 | 7127 |
| epoch 16--30（frontend/codebook凍結） | 56000000 | 1723 |
| epoch 31--90（凍結解除後） | 12000000 | 7127 |

4 GPU・num_workers=4で、長い発話などを含む試験セット全体の学習、検証、注意重み描画、
チェックポイント保存を確認済み。試験時のGPU全体の最大使用量は、凍結中が約27.6 GiB、
凍結解除後が約41.6 GiBだった。さらに大きい候補も比較し、速度面で有力だった値を採用した。
これは試験セットによる比較で、epoch全体の実時間は本学習のログで確認する。

今回の再開ジョブはまず `max_epoch=30` で実行し、終了後に調整処理がYAMLの
`batch_bins`を12000000へ戻して、同じcheckpointからepoch31--90を自動再開する。
一連の実行を管理するプロセスはtmux `dsu` 内に残す。

## 損失とログ

主音声のASR/設定済みclustering損失は全件、SIRはペアあり発話だけで平均する。
ペアなしバッチのSIRは0。DDP学習時は全workerのペア数で正規化するため、ペアの偏りや
ペアなしworkerがあっても平均の分母に未対応発話を含めない。
`loss_sir` と `sir_num_pairs` を監視する。DDP学習時のペア数は全workerの合計。
評価時はローカルのペア平均・ペア数を返し、モデル内ではGPU間通信を行わない。
注意重みの描画はrank 0だけで実行されるためである。
分散検証の集計はESPnet標準の主音声バッチ数による重み付き平均に従う。
epoch集計の `sir_num_pairs` はバッチごとの平均で、epoch全体の累積数ではない。

UASpeechで健常参照を使う場合は同じモデルAPIを使用できる。
全件ペアがあれば `sir_pair_mask` を省略し、健常参照の `healthy.scp` を渡す。
