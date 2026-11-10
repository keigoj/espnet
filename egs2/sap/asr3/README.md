# SAP ASR3：微分可能な離散音声トークンを用いた学習レシピ

文書に基づく軽度話者の選定と、一部の発話にのみ参照音声があるSIR学習については、
[SAP軽度話者参照SIR](README_sir_mild.md)を参照してください。

詳細な仕様、実装の対応関係、生成物、検証項目は
[実装仕様書](IMPLEMENTATION_JA.md)に記載しています。

このレシピには、次の2種類の実験経路があります。

1. **オフライントークン方式**：tokenizerを学習してから離散トークンを抽出し、
   既存のdiscrete ASRを学習します。
2. **E2E方式**：tokenizerをASRの計算グラフに組み込み、ASR損失で共同最適化します。

## オフライントークン方式の概要

従来のオフライントークン方式は、次の2段階で実験します。

1. SAPのK-means centroidで初期化したGumbel-Softmax K-meansのcodebookと、
   WavLMの21層目までを共同学習します。必要に応じて健常音声参照のOTを使用できます。
2. 決定的なhard unitを抽出し、既存のSAP discrete ASRモデルを学習します。

実行コマンドは次のとおりです。

```bash
cd egs2/sap/asr3
./run.sh
```

## E2E方式

E2E方式では、トークン化をASRの計算グラフ内で行います。

```text
波形 -> WavLM -> K-meansのクラスタ割当 -> token embedding -> E-Branchformer
```

固定tokenizerのベースラインは、次のコマンドで実行します。

```bash
./run_e2e.sh --e2e_mode fixed
```

クラスタ数に応じて、SAP K-meansの初期centroidの既定パスと実験ディレクトリ名が
決まります。

```bash
./run_e2e.sh --e2e_mode fixed --nclusters 1000
# 初期centroid: ../asr2/exp/kmeans/wavlm_large_21_base_1000clusters/km_1000.npy
# 実験ディレクトリ: exp/asr_e2e_wavlm_large_l21_gumbel_km1000_tokembed512_fixed_ep90_en
```

この配置と異なるcentroidファイルを使う場合は`--initial_centroid_path`を指定します。
実験ディレクトリを変更する場合は`--asr_exp`を指定します。起動スクリプトは、
centroidファイルの行数が`nclusters`と一致することを実行前に確認します。

### 学習モードとクラスタ割当

同じモデル構成とデータ経路で、次の3条件を比較できます。

| `e2e_mode` | 1〜30 epochの学習対象 | 31〜90 epochの学習対象 |
|---|---|---|
| `fixed` | embedding＋ASR | embedding＋ASR |
| `quantizer` | embedding＋ASR | codebook＋embedding＋ASR |
| `full` | embedding＋ASR | 学習対象のWavLM層＋codebook＋embedding＋ASR |

クラスタ割当を選択するのは、WavLMの特徴抽出後、ASR用token embeddingの前にある
quantizer内部です。現行設定は、
[ESPnet commit 0c7942d1のtokenizer](https://github.com/espnet/espnet/blob/0c7942d1e0db7828f8adb2154561c28542ae7806/espnet2/tok/tokenizer.py)
の処理に対応しています。

| 段階 | クラスタ割当 | 学習対象 |
|---|---|---|
| 凍結中のウォームアップ | Gumbel noiseなしの決定的な最近傍割当 | token embedding＋ASR |
| 共同学習 | hardなStraight-Through Gumbel sampling | 指定されたtokenizer部分＋token embedding＋ASR |
| 検証・デコード | Gumbel noiseなしの決定的な最近傍割当 | なし |

`fixed`では、全学習期間で凍結中と同じ割当規則を使います。ASRへの入力は
`assignment @ embedding.weight`です。centroidと独立した学習可能な512次元embeddingと、
位置符号化（dropout 0.1）を使います。

新規設定では、この経路専用のクラスを`model: tokenized_asr`で選択します。
既存の`model: gumbel_kmeans`クラスは、旧checkpointとの互換性のため、従来の
centroid入力を維持します。新構成の凍結中のsampling切り替えに、追加フラグは不要です。

元実装との対応、残る相違点、checkpointの互換性については
[元実装との対応表](UPSTREAM_ALIGNMENT_JA.md)を参照してください。
GPUでの学習テストと論文の実験条件との比較は
[学習検証結果](TRAINING_VALIDATION_JA.md)に記載しています。

### 凍結と共同学習のスケジュール

既定スケジュールは、恩田らの論文の単一層実験に合わせています。WavLMとcodebookを
30 epochまで凍結し、同じ学習プロセス内で31 epochの開始時に対象部分の凍結を解除します。

```bash
./run_e2e.sh --e2e_mode full --warmup_epochs 30 --max_epoch 90
```

切り替え時にoptimizerとschedulerは作り直しません。1 epoch目から対象部分を更新する
場合は`--warmup_epochs 0`を指定します。既定の実験名には、凍結期間と総epoch数の
両方が含まれます。例えば、
`asr_e2e_wavlm_large_l21_gumbel_km2000_tokembed512_full_warmup30_ep90_en`
となり、異なるスケジュールのcheckpointから誤って再開することを防ぎます。

codebookの凍結中は温度の減衰も停止します。既定の30 epoch凍結では、30 epochまで
温度を2.0に保ち、31 epochも2.0から開始します。その後、学習時のforwardごとに
減衰が進みます。温度の更新counterはcheckpointに保存されます。
現行設定では、`full`と`quantizer`のウォームアップ中にGumbel samplingを無効にします。
学習率schedulerは、両段階を通じてリセットせずに継続します。

`run_e2e_4090_amp.sh`も、既定で最初の30 epochを凍結します。このスクリプトが選ぶ
1200k設定はTime Maskありです。base設定と別の1600k AMP設定は`specaug: null`です。
RTX 4090で新規実験を行う場合は、次のように実行します。

```bash
./run_e2e_4090_amp.sh --stage 11 --stop_stage 11 \
  --asr_exp exp/asr_e2e_wavlm_large_l21_gumbel_km2000_tokembed512_full_warmup30_ep90_en
```

現行設定は、`freeze_param`によりWavLMの畳み込み特徴抽出器を全期間で凍結します。
`full`では、それ以外のWavLMパラメータの凍結を31 epochで解除します。
障害者音声認識の実験で、SSLとcodebookの初期凍結期間を設けずに学習する場合は、
次のように指定します。畳み込み特徴抽出器は固定したまま、WavLM Transformerと
codebookを1 epoch目から更新します。

```bash
./run_e2e.sh \
  --e2e_mode full \
  --warmup_epochs 0
```

この条件には`full_warmup0_ep90`という実験名のタグが付きます。
段階的に凍結を解除する実験から、誤って再開することを防ぎます。

### 損失関数と温度

3モードとも、同じオフラインSAP K-meansのcentroidから開始します。既定のE2E目的関数は
ASR損失のみ（`0.3 * CTC + 0.7 * attention`）です。K-means、commitment、diversity、
OTの損失重みは0です。共同学習ではhardなStraight-Through Gumbel割当を使い、
ウォームアップ中と評価時は決定的な割当を使います。

E2Eの温度設定は、参照先quantizerの既定値に合わせています。初期値2.0、下限0.1、
学習可能なquantizerのforwardごとの減衰率0.999995です。これは実装の既定値との
対応であり、2025年論文の実験configそのものとの一致を確認したものではありません。
バッチ構成によって、epochあたりの温度減衰の速さは変わります。

E2E設定の`quantizer_conf.logit_scale: 1.0`は、論文の二乗距離の係数に対応します。
`hard: true`では、forwardはone-hot割当、backwardはsoft割当の勾配を使います。
`quantizer_sample_gumbel: false`で決定的な最近傍割当を選ぶ場合も、学習時には
同様に勾配を流せます。noiseを無効にすること自体は、SSLへの勾配を切断しません。
ただし、ウォームアップ中はtokenizerを明示的に凍結するため、その部分は更新しません。

### 分岐実験用の30 epoch checkpoint

後から条件を変えて実験するために、最初の30 epochを凍結したcheckpointを作るには、
次のように実行します。

```bash
./run_e2e_4090_amp.sh --stage 11 --stop_stage 11 \
  --warmup_epochs 30 --max_epoch 30
```

生成される`checkpoint.pth`には、モデルに加えてoptimizer、scheduler、scaler、epochの
状態が含まれます。この30 epochでtoken embeddingとASRが学習されます。

既存のcentroid入力モデルの学習を再開する場合は、その実験に保存されている
`exp/.../config.yaml`を使ってください。旧モデルの重みを新しいembedding構成の
初期値として使うには、重みの一部だけを読み込む方法を明示的に決める必要があります。

### E2E方式のstage番号

`run_e2e.sh`は、標準ASR1のstage番号を使います。SAPの`data/`ディレクトリは既に
存在し、`local/data.sh`は空なので、stage 2から開始します。LMとn-gramモデルは
学習しないため、stage 6〜9はスキップします。stage 10は統計収集、11はASR学習、
12はデコード、13はスコア計算です。

## オフライントークン方式のstageと設定

以下は`run.sh`から呼び出す`asr3.sh`独自のstage番号です。
上記のE2E方式のstage番号とは異なります。

| stage | 処理 |
|---:|---|
| 1 | SAPデータの準備、速度摂動、波形の形式変換 |
| 2 | SSL学習用の統計収集 |
| 3 | WavLMとGumbel K-meansの共同学習 |
| 4 | hard unitの抽出 |
| 5 | CJK文字への変換と、後段ASR用データの準備 |
| 6 | ASR2のトークンリスト作成、ASR学習、デコード、スコア計算 |

例えば、先にtokenizerだけを学習して検証指標を確認し、その後、トークン抽出とASRへ
進む場合は次のように実行します。

```bash
./run.sh --stop_stage 3
./run.sh --stage 4
```

健常音声の参照には、既定で`loss[0].conf.reference_centroid_path`に指定された
LibriSpeechのWavLM 21層目のK-meansファイルを使います。形状は`(2000, 1024)`である
必要があります。WavLM特徴を同じ1024次元空間のままK-meansへ渡し、線形変換や
Transformerによる射影は行いません。

Sinkhorn divergenceには、POTのPyTorchバックエンドを使います。OTを有効にする前に
レシピの依存パッケージをインストールするか、
`python -m pip install "POT>=0.9.7,<0.10"`を実行してください。
stage 3では、学習開始前にインストール済みのPOTのバージョンを確認します。

`ot_mass_mode: uniform`は、将来、centroidごとに異なる重みを扱うために用意した
設定項目です。低レベルのOT APIは、SAPと健常音声それぞれのクラスタ質量テンソルを
受け取れます。ただし、現行レシピはどちらも渡さず、一様な重みを使います。

現行のオフライントークン設定は、K-meansのdistortion、diversity、variance損失を
使用し、OTとcommitmentは無効にしています。`exp/ssl_*/train.log`で確認する主な
指標は、`gumbel_kmeans_distortion`、`gumbel_kmeans_diversity`、
`gumbel_kmeans_variance`、`gumbel_kmeans_perplexity`です。
perplexityが1付近に留まる場合、ほぼすべてのフレームが1つのクラスタを使っています。

`run.sh`の既定クラスタ数は2000です。変更する場合は、`run.sh`の`nclusters`と、
SSL設定の`loss[0].conf.num_clusters`の両方を変更してください。
WavLMをESPnetのS3PRL frontend経由で読み込むため、S3PRLのインストールが必要です。
