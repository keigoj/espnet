# SAP ASR3 モデル・評価結果一覧

確認日: 2026-11-17。対象: このレシピの `exp/` 直下25ディレクトリ。
保存済み `config.yaml`、`run.sh`、`train*.log`、`RESULTS.md`、
`decode*/{test1,test2}/score_{wer,cer}/result.txt` を確認した。
新たな学習・デコードは実施していない。

## ASRの認識精度

値はスコアファイルの **Sum/Avg 行の Err（%）**。WERは単語誤り率、CERは文字誤り率で、
いずれも低いほどよい。話者別誤り率の単純平均（Mean）ではない。
WERは挿入誤りを含むため100%を超えることがある。

| ID | モデル・方式 | test1 WER (%) | test2 WER (%) | test1 CER (%) | test2 CER (%) | 評価checkpoint |
|---|---|---:|---:|---:|---:|---|
| A | 固定WavLM＋SAP K-means → オフライン離散ASR | **18.7** | **19.7** | **12.4** | **13.3** | `valid.acc.ave.pth`（上位10個平均） |
| B | SSL＋微分可能K-meansを先に学習 → オフライン離散ASR | 100.4 | 102.2 | 82.3 | 83.5 | `valid.acc.ave.pth`（上位10個平均） |
| C | E2E、初期凍結なし（warmup0） | 19.7 | 21.0 | 13.3 | 14.6 | `18epoch.pth`（単一） |
| D | E2E、最初30 epoch凍結（warmup30） | 19.9 | 21.0 | 13.1 | 14.1 | `valid.acc.ave.pth`（上位10個平均） |

各モデルで参照文の件数・単語数は一致している。

| 評価集合 | 発話数 | 参照単語数 | 参照文字数 |
|---|---:|---:|---:|
| test1 | 24,981 | 212,718 | 1,077,475 |
| test2 | 22,273 | 186,270 | 947,663 |

保存された結果ではAが最良。Aに対するWERの差は、Cがtest1で+1.0、test2で+1.3ポイント、
Dがtest1で+1.2、test2で+1.3ポイント。ただし後述の構成・学習・checkpoint選択の差があり、
これだけで初期凍結や共同学習の効果を切り分けることはできない。

## 各ASRモデルの説明と保存先

4モデルともWavLM-largeの21層目、クラスタ数2000を使用する。
後段は12ブロックのE-Branchformer（出力256次元）と6ブロックのTransformer decoder。
ASR損失はCTC 0.3＋attention 0.7、出力は英語BPE 5000。
学習には `train_sp`、検証には `dev` を使用し、速度摂動は0.9/1.0/1.1。

| ID | `exp/` 内のディレクトリ | 内容・学習状況 | 設定・結果 |
|---|---|---|---|
| A | `asr_wavlm_large_l21_sap_km2000_fixed_en` | 事前学習済みWavLMとSAP centroidを固定してunitを抽出。連続重複除去→入力BPE 6000→512次元embedding→ASRを別学習。ASRの損失はtokenizerへ戻らない。70/70 epoch完了、単一bestは53 epoch。 | [設定](exp/asr_wavlm_large_l21_sap_km2000_fixed_en/config.yaml)、[結果](exp/asr_wavlm_large_l21_sap_km2000_fixed_en/RESULTS.md) |
| B | `asr_wavlm_large_l21_gumbel_km2000_sap_init_div05_en` | SAP centroid初期化でWavLM＋codebookをクラスタリング損失により先に学習し、unitを固定。その後のASR構成はAと同じ。ASR損失によるtokenizerの共同最適化は行わない。70/70 epoch完了、単一bestは7 epoch。 | [設定](exp/asr_wavlm_large_l21_gumbel_km2000_sap_init_div05_en/config.yaml)、[結果](exp/asr_wavlm_large_l21_gumbel_km2000_sap_init_div05_en/RESULTS.md) |
| C | `asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup0_ep90_en` | 波形→WavLM→Gumbel K-means→1024次元centroid特徴→ASR。初期凍結なしでASR損失からWavLM・codebookを更新する旧E2E構成。最新保存は44 epoch、設定上の上限は90。ログ末尾は `KeyboardInterrupt`。評価は18 epoch。 | [設定](exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup0_ep90_en/config.yaml)、[結果](exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup0_ep90_en/RESULTS.md) |
| D | `asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup30_ep90_en` | Cと同じ旧E2E接続。WavLM・codebookを30 epochまで凍結し、31 epochから解除。90/90 epoch完了、単一bestは28 epoch。評価用平均の選択元はすべて凍結期間内（下記）。 | [設定](exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup30_ep90_en/config.yaml)、[結果](exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup30_ep90_en/RESULTS.md) |

A・BのASR本体は約40.36Mパラメータ（別途tokenizerが必要）。C・DはWavLMを含め約355.18M。
C・DではWavLMのCNN feature extractorを全期間で固定し、K-means・commitment・diversity・
OTの補助損失重みはすべて0。保存設定にSIRの有効化はない。

### 比較時に必要な条件の違い

| 条件 | A・B | C: warmup0 | D: warmup30 |
|---|---|---|---|
| `model` | `discrete_asr` | `gumbel_kmeans` | `gumbel_kmeans` |
| ASR入力 | unit重複除去＋BPE＋512次元embedding | フレームごとの1024次元centroid | フレームごとの1024次元centroid |
| SpecAug | Time Maskあり（比率0–0.05、10個） | なし | Time Maskあり（比率0–0.05、10個） |
| optimizer / 設定LR | Adam / 0.006 | AdamW / 0.0001 | AdamW / 0.0001 |
| LR warmup steps | 15000 | 5000 | 5000 |
| `batch_bins` / `accum_grad` | 70000000 / 8 | 1200000 / 12 | 1200000 / 12 |
| quantizer `logit_scale` | Bのtokenizerは1/1024 | 1/1024 | 1.0 |
| quantizer温度（初期/下限/減衰率） | Bのtokenizerは2.0 / 0.5 / 0.999995 | 2.0 / 0.5 / 0.999990843068 | 2.0 / 0.5 / 0.999990843068 |
| checkpoint選択 | 検証 `acc` 上位10個平均 | 18 epoch単一 | 検証 `acc` 上位10個平均 |

`batch_bins` は入力表現が異なるので同じ単位のバッチ量として比較できない。
検証ログの `acc` はcheckpoint選択用の指標であり、上表のデコードWERとは異なる。
デコード設定は両方式ともbeam size 20、CTC weight 0.3、LM weight 0、外部LMなし。
根拠: [E2Eデコード設定](conf/decode_e2e_ctc0.3.yaml)、[離散ASRデコード設定](../asr2/conf/decode_ctc0.3.yaml)。

Dの[保存済みcheckpoint診断](exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup30_ep90_en/analysis/checkpoint_diagnostics.json)
では、平均に選ばれたepochは **20, 21, 22, 23, 24, 25, 27, 28, 29, 30**。
学習ログでも凍結解除は31 epochであり、評価されたモデルは共同学習後の重みを含まない。
`90epoch.pth` は存在するが、そのcheckpoint単独のtest評価結果は見つからなかった。
Cには18 epochでのdev評価もあり、WER 15.1%、CER 11.3%。他モデルのdevデコード結果は見つからなかった。

## tokenizer単体の実験

`ssl_*` は文字列を出力するASRモデルではなく、WavLM＋Gumbel K-meansの学習結果。
これら単体のWER/CERはない。Bに接続されたもの以外は、この `exp/` 内に対応する後段ASR評価がない。
下表のloss・perplexityは `valid.loss.best.pth` が指す単一epochの検証ログ値であり、
unit抽出で使う平均checkpointのスコアではない。

| `exp/` 内のディレクトリ | 初期centroid・主な保存設定 | 完了/上限epoch | best epoch | valid loss | valid perplexity |
|---|---|---:|---:|---:|---:|
| `ssl_wavlm_large_l21_gumbel_km2000` | LibriSpeech初期化。保存YAMLはKM=1、diversity=0.1、variance=0、入力正規化なし。ただし学習時ログとの不一致あり（下記）。 | 20/20 | 10 | 0.0001545 | 1.003 |
| `ssl_wavlm_large_l21_gumbel_km2000_div01` | LibriSpeech初期化。KM=1、diversity=0.1、variance=0、入力正規化なし。 | 20/20 | 7 | 0.755 | 1.088 |
| `ssl_wavlm_large_l21_gumbel_km2000_div05` | LibriSpeech初期化。KM=0.5、diversity=0.5、variance=0、入力正規化あり。 | 10/10 | 10 | 9.752 | 1941 |
| `ssl_wavlm_large_l21_gumbel_km2000_sap_init_div05` | SAP初期化。KM=0.1、diversity=0.5、variance=10、入力正規化なし、`logit_scale=1/1024`。Bの前段。 | 10/10 | 9 | 2.058 | 573.642 |
| `ssl_debug_cpu_wavlm_large_l21_gumbel_km2000_ot_uniform` | CPU・AMPなし、少量データの動作確認。KM=1、OT=0.01。 | 1/1 | 1 | 66.573 | 112.156 |
| `ssl_debug_gpu_wavlm_large_l21_gumbel_km2000_ot_uniform` | GPU・AMPあり、少量データの動作確認。KM=1、OT=0.01。 | 1/1 | 1 | 63.301 | 112.363 |

KM等は損失の係数。上の通常4実験はOT=0、すべてcommitment=0。
損失係数・入力正規化・実装時期が異なるため、lossの大小をそのままモデルの優劣と解釈できない。
debugのOTログ値は0であり、この結果からOTの認識性能への効果は評価できない。

### 抽出済みunitの確認

各 `units_*/dev/pseudo_labels.txt` を全行集計した。全6ディレクトリとも24,848発話。
種類数はdev全体で実際に現れたhard unit IDの数で、クラスタ数2000とは区別する。
トークン数は保存ファイル中の数であり、元の音声フレーム数ではない。

| `exp/` 内のディレクトリ | devトークン総数 | devのunit種類数 | 確認内容 |
|---|---:|---:|---|
| `units_wavlm_large_l21_gumbel_km2000` | 24,848 | 1 | 全発話がID 693のみ。 |
| `units_wavlm_large_l21_gumbel_km2000_div01` | 74,542 | 3 | 利用IDが3種類に限定。 |
| `units_wavlm_large_l21_gumbel_km2000_div05` | 24,848 | 1 | 全発話がID 222のみ。 |
| `units_wavlm_large_l21_gumbel_km2000_reference_init` | 4,658,770 | 1,989 | `dev/` のみ存在。名前以外の抽出条件を裏付ける `run.sh` はない。 |
| `units_wavlm_large_l21_gumbel_km2000_sap_init_div05` | 4,297,537 | 443 | Bの入力unit。 |
| `units_wavlm_large_l21_sap_km2000_fixed` | 5,052,032 | 1,984 | Aの入力unit。固定抽出の `run.sh` あり。 |

初期3実験はhard unitの利用が極端に偏っている。特に `div05` は検証perplexityが高くても
抽出IDは1種類だった。現行実装のperplexityはsoftな割当分布から計算されるため、
hard IDの利用種類数と同一視できない。Bの高いWERの原因は、この集計だけでは断定しない。

## その他のディレクトリ（モデルの認識精度を持たない生成物）

| `exp/` 内のディレクトリ | 用途 |
|---|---|
| `asr_stats_e2e_raw_en_bpe5000_sp` | E2E ASRの統計・shape収集。 |
| `asr_stats_raw_rm_wavlm_large_l21_gumbel_km2000_sap_init_div05_bpe6000_bpe5000_sp` | Bの後段ASRの統計・shape収集。 |
| `asr_stats_raw_rm_wavlm_large_l21_sap_km2000_fixed_bpe6000_bpe5000_sp` | Aの後段ASRの統計・shape収集。 |
| `ssl_stats_debug_cpu_wavlm_large_l21_gumbel_km2000_ot_uniform` | debug用tokenizerの統計・shape収集。 |
| `ssl_stats_wavlm_large_l21_gumbel_km2000` | 初期tokenizerの統計・shape収集。 |
| `ssl_stats_wavlm_large_l21_gumbel_km2000_div01` | diversity 0.1実験の統計・shape収集。 |
| `ssl_stats_wavlm_large_l21_gumbel_km2000_div05` | diversity 0.5実験の統計・shape収集。 |
| `ssl_stats_wavlm_large_l21_gumbel_km2000_sap_init_div05` | SAP初期化tokenizerの統計・shape収集。 |
| `e2e_training_check_20261110` | 設定監査・GPU smoke test・上流実装との比較。学習済みASRのtest精度ではない。 |

## 現行ドキュメント・コードと保存実験の相違

1. 現行READMEの新規E2E経路は `model: tokenized_asr` の独立512次元embedding方式。
   C・Dの保存設定は `model: gumbel_kmeans` で、コードもcentroid特徴を直接返す。
   現行READMEにも旧クラス互換として区別されているため、新方式の実験結果として扱わない。
2. 現行の新規E2E設定は温度下限0.1・減衰率0.999995。C・Dの保存値は下限0.5・
   減衰率0.999990843068。Cは距離係数も現行の1.0と異なる1/1024。
3. 現行READMEの「凍結中は決定的な最近傍割当」は新クラスの動作。
   Dの旧クラスは `quantizer_sample_gumbel: null` で、コードではtrain/eval状態に従うため、
   パラメータ凍結だけを根拠に「凍結中の学習もnoiseなし」と説明できない。
4. 現行オフライン設定の説明はdistortion・diversity・varianceの併用だが、過去の
   `ssl_*` 全実験がその条件ではない。上表の各保存設定を区別する。
5. `ssl_wavlm_large_l21_gumbel_km2000/config.yaml` はdiversity=0.1だが、元の学習ログ
   `train.1.log` のbest epochはdistortion=0.0001545、diversity=7.598、total loss=0.0001545。
   現行コードの加重和でdiversity=0.1なら約0.760となり、記録値と一致しない。
   `train.log` は20 epoch完了後の再開で設定を再保存し、追加学習せず終了した記録。
   保存YAMLを元の全学習期間の条件と断定しない。

以上は新旧実験・保存記録の差であり、今回確認した範囲では、現行READMEが区別している
新旧クラスの仕様と現行コードの間に新たな矛盾は見つからなかった。

## 確認方法

4モデル×2評価集合×2指標の16個のスコアを `RESULTS.md` と照合し、
評価集合の発話数・参照数、設定、checkpointリンク、学習終了記録も確認した。
unitの集計は保存テキストに対する集計。コード・実験条件・重みは変更していない。
文書追加のみのため、モデルの再推論、pytest、テンソルshape・デバイスの実行検証は行っていない。
