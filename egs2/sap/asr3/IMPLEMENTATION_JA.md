# SAP ASR3: 微分可能K-means 実装仕様・動作確認手順

## 1. この文書の目的

この文書は、以下の実装について、仕様とコードの対応関係を確認するためのものです。

- Gumbel-Softmaxを用いた微分可能K-means
- SSLエンコーダとK-meansコードブックの同時学習
- 健常音声の固定centroidとの一様質量Sinkhorn OT正則化
- 学習済みモデルからの決定的な離散unit抽出
- 抽出unitを入力とした後段ASRモデルの別学習
- SSL、K-means、ASRを一つのモデルで学習するためのコア実装

SAPで最初に行う実験は、次の二段階構成です。

```text
第1段階:
waveform
  -> WavLM large layer 21
  -> Identity encoder (1024 dimを変更せず通過)
  -> Gumbel K-means (2000 clusters, 1024 dim)
     <-> fixed healthy-speech centroids (2000 clusters, 1024 dim)
  -> hard unit sequence

第2段階:
hard unit sequence
  -> CJK表現
  -> source BPE
  -> discrete ASR E-Branchformer
  -> transcription
```

第1段階と第2段階の間ではunitをファイルへ固定しているため、後段ASRの損失は
WavLMやK-meansへ逆伝播しません。これは、tokenizer単体の性質を先に評価しやすく
するための意図した仕様です。

## 2. 実装済み範囲

| 項目 | 状態 | 備考 |
|---|---|---|
| 微分可能K-means層 | 実装済み | Gumbel-Softmax、hard straight-through対応 |
| SSL + K-means | 実装済み | `SSLTask`のlossとして登録 |
| 健常centroidとのOT loss | 実装済み | 一様質量、log-domain Sinkhorn divergence |
| HuBERT loss + K-means loss | 実装済み | 複数lossとして同時利用可能。HuBERT用ラベルは別途必要 |
| hard unit抽出 | 実装済み | SSL/ASRチェックポイントの両方に対応 |
| SAP `asr3` tokenizerレシピ | 実装済み | WavLM large layer 21、2000 clustersが初期設定 |
| 抽出unitによる後段ASR2学習 | 実装済み | SAP `asr2`のstage 7以降を再利用 |
| SSL + K-means + ASRモデル | コア実装済み | ASR損失をSSL frontendまで逆伝播可能 |
| SAP用の完全end-to-end ASRレシピ | 実装済み | `run_e2e.sh`で`fixed`・`quantizer`・`full`を比較 |

注意点として、現在の`asr3`でいう「SSL + K-means」は、事前学習済みWavLMを
クラスタリング損失でfine-tuneする構成です。HuBERTのmasked predictionなど、
独立した予測型SSL損失はデフォルト設定には含めていません。

## 3. 微分可能K-meansの仕様

### 3.1 入出力

入力を `h`、コードブック中心を `c` とします。

```text
h: (batch, time, feature_dim)
c: (num_clusters, feature_dim)
```

出力は以下です。

| 出力 | 内容 |
|---|---|
| `quantized` | コードブック中心から構成した量子化特徴 |
| `indices` | 各フレームのクラスタID。padding位置は`-1` |
| `probabilities` | クラスタ割当確率。padding位置は0 |
| `distortion_loss` | K-means距離損失 |
| `commitment_loss` | encoder特徴を割当中心へ近づける損失 |
| `diversity_loss` | クラスタ利用分布を一様分布へ近づける損失 |
| `codebook_perplexity` | 実効的に利用されているクラスタ数 |
| `temperature` | 現在のGumbel-Softmax温度 |

### 3.2 距離と割当

フレーム `t` とクラスタ `k` の二乗ユークリッド距離を使います。

```text
d(t, k) = ||h_t - c_k||^2
logit(t, k) = -d(t, k)
```

学習中は以下の割当です。

```text
p_t = gumbel_softmax(logit_t, temperature, hard=true)
q_t = sum_k p(t, k) c_k
```

`hard: true`の場合、forwardではone-hotですが、backwardではsoft確率の勾配を
使うstraight-through estimatorになります。そのため、K-means損失から
コードブックと上流encoderの両方へ勾配が流れます。

評価時と明示的なunit抽出時はGumbel noiseを使わず、最近傍中心を決定的に
選びます。同一チェックポイントと同一音声に対する抽出結果は再現可能です。

### 3.3 温度スケジュール

温度は更新回数に応じて次式で減衰します。

```text
temperature = max(
    min_temperature,
    initial_temperature * temperature_decay ^ num_updates
)
```

デフォルト設定は以下です。

```yaml
temperature: 2.0
min_temperature: 0.5
temperature_decay: 0.999995
```

更新回数はbufferとしてチェックポイントに保存されます。

### 3.4 損失

有効フレーム数を `N`、特徴次元を `D`、クラスタ数を `K` とします。

K-means distortionは特徴次元に依存して損失スケールが過大にならないよう、
`N * D`で正規化しています。

```text
q_t = sum_k p(t, k) c_k
L_distortion = (1 / ND) sum_t ||h_t - q_t||^2
```

これは論文 "Differentiable K-means for Fully-optimized Discrete Token-based
ASR" の式(3)に対応します。`hard: true`ではforwardの`p(t, k)`はone-hot、
backwardはsoft Gumbel-Softmaxの勾配を使います。

commitment lossも同じスケールにしています。`sg`はstop-gradientです。

```text
L_commitment = (1 / ND) sum_t ||h_t - sg(q_t)||^2
```

soft割当のバッチ平均を `p_bar` とすると、diversity lossは次の形です。

```text
L_diversity = log(K) - H(p_bar)
```

全クラスタが均等に利用される場合は0へ近づきます。

クラスタリング損失だけでは、全特徴と全中心が同じ値になるcollapsed solutionが
存在します。その緩和のため、SSL loss側に特徴分散正則化を追加しています。

```text
L_variance = mean_d ReLU(
    variance_target - sqrt(Var(h[:, d]) + 1e-4)
)
```

SSL + K-meansで使う合計損失は以下です。

```text
L = lambda_km         * L_distortion
  + lambda_commit     * L_commitment
  + lambda_diversity  * L_diversity
  + lambda_variance   * L_variance
  + lambda_ot         * L_OT
```

`L_OT`は学習中のSAP codebookと固定した健常音声centroidを、それぞれ一様質量の
離散分布として比較するSinkhorn divergenceです。centroid間のコストは特徴次元で
正規化した二乗Euclidean距離です。

```text
C(i, j) = ||c_sap_i - c_healthy_j||^2 / D
```

健常音声centroidはPyTorch bufferとしてcheckpointに保存されますが、optimizerの
更新対象にはなりません。SAP codebookは`nn.Parameter`なのでOT lossから更新されます。
K-means distortionを併用することで、SAP特徴もcodebookへ近づく方向に更新されます。
輸送問題の求解にはPOT 0.9.7系のPyTorch backendを使います。POTのlog-domain
Sinkhorn solverとenvelope gradientを使い、固定健常者側のself-transport costだけを
bufferへcacheします。

ここで`ot_weight`と各clusterの質量は別の値です。

- `ot_weight`: 合計lossへ掛ける係数 `lambda_ot`
- `ot_mass_mode`: centroidごとの確率質量を決める方式

現在の`ot_mass_mode`は`uniform`だけを実装しています。将来の拡張用にsolverと
`GumbelKMeans.optimal_transport_loss`は、`(K,)`形状の`source_mass`と
`reference_mass`を任意引数として受け取れます。入力値は正値か検査した後、総和1へ
正規化されます。たとえば将来、SAP側に平滑化したcluster使用頻度、健常者側に事前
K-means時のcluster頻度を接続できます。明示した参照質量ではuniform用self-cost cacheを
再利用しません。未実装のmode名は黙ってuniformにせず`NotImplementedError`にします。

現在のSAP設定値は次のとおりです。

```yaml
kmeans_weight: 1.0
commitment_weight: 0.0
diversity_weight: 0.0
variance_weight: 0.0
ot_weight: 0.01
variance_target: 1.0
reference_centroid_path: ../../librispeech_100/asr2/exp/kmeans/wavlm_large_21_base_2000clusters/km_2000.npy
initialize_from_reference: true
ot_epsilon: 1.0
ot_iterations: 20
ot_mass_mode: uniform
```

初回実験は効果を解釈しやすくするため、`L_distortion + 0.01 * L_OT`だけを最適化
します。commitment・diversity・varianceの値自体はstatsへ出ますが、合計lossには
寄与しません。perplexityが1付近へcollapseする場合は、まず`diversity_weight`だけを
有効化して比較し、その後に必要ならvariance、commitmentの順で検討します。

paddingフレームは上記の損失、平均利用率、perplexityの計算対象から除外されます。
AMP利用時も距離計算はfloat32で行います。

## 4. SAP tokenizerモデルの仕様

設定ファイルは `conf/train_ssl_kmeans_wavlm_large.yaml` です。

| 部分 | 設定 |
|---|---|
| SSL frontend | S3PRL `wavlm_large` |
| SSL layer | 21 |
| WavLM出力次元 | 1024 |
| projection | なし |
| 追加encoder | Identity（特徴を変更しない） |
| K-means | 2000 clusters、1024次元 |
| 健常centroid | LibriSpeech、WavLM layer 21、2000 clusters、1024次元 |
| OT | POT、一様質量Sinkhorn divergence、epsilon `1.0`、最大20 iterations |
| OT mass mode | `uniform`（非一様質量用placeholderあり） |
| WavLM freeze | なし |
| optimizer | Adam、lr `5e-5` |
| scheduler | WarmupLR、1000 steps |
| max epoch | 20 |
| AMP | 有効 |
| batch | `numel`、`batch_bins=800000` |
| gradient accumulation | 16 |

`freeze_param`を指定していないため、次のすべてが更新対象です。

- WavLM large
- K-meansコードブック

主な勾配経路は次のとおりです。distortionとdiversityはコードブックと上流特徴の
両方を更新し、commitmentとvarianceは主に上流特徴を更新します。

```text
L_OT ---------------------------------------> K-means codebook
                                              ^
fixed healthy centroids ---------------------| (勾配更新なし)

L_distortion / L_diversity -----------------> K-means codebook
       |
L_commitment / L_variance
       |
       v
WavLM large layer 21以前のパラメータ
```

`layer: 21`はWavLMの出力としてlayer 21を選択する指定です。上流のlayer 21以前は
計算グラフに含まれ、freezeしていないため更新されます。

## 5. `asr3` stage仕様

### Stage 1: データ準備と波形整形

```bash
./run.sh --stage 1 --stop_stage 1
```

SAP `asr2`のstage 1--4を内部的に再利用します。

- `local/data.sh`によるSAPデータ準備
- speed perturbation: `0.9 1.0 1.1`
- 16 kHz、FLACへの整形
- 0.1秒以下、30秒以上の学習・validation音声の除外

主な生成物は以下です。

```text
dump/audio_raw/train/wav.scp
dump/audio_raw/train_sp0.9/wav.scp
dump/audio_raw/train_sp1.1/wav.scp
dump/audio_raw/dev/wav.scp
dump/audio_raw/test1/wav.scp
dump/audio_raw/test2/wav.scp
```

### Stage 2: SSL collect-stats

```bash
./run.sh --stage 2 --stop_stage 2
```

学習・validation音声のshapeを収集します。この段階では
`extract_feats_in_collect_stats: false`のため、WavLM本体を構築して特徴抽出する
処理は行いません。

```text
exp/ssl_stats_wavlm_large_l21_gumbel_km2000_ot_uniform/
  tokens.txt
  train/speech_shape
  valid/speech_shape
  logdir/stats.*.log
```

`tokens.txt`は`SSLTask`のインターフェースを満たすためのdummy token listです。
このレシピのK-means学習では教師テキストを使用しません。

### Stage 3: WavLM + Gumbel K-means学習

```bash
./run.sh --stage 3 --stop_stage 3
```

学習対象は元の`train`です。speed perturbationデータはtokenizer学習には入れず、
後段ASR用のunit抽出対象として使用します。

主な生成物は以下です。

```text
exp/ssl_wavlm_large_l21_gumbel_km2000_ot_uniform/
  config.yaml
  train.log
  checkpoint.pth
  valid.loss.best.pth
  valid.loss.ave.pth
```

ESPnetの保存設定や学習終了時点によって、存在する中間checkpoint名は変わることが
あります。Stage 4はデフォルトで`valid.loss.ave.pth`を読みます。

### Stage 4: hard unit抽出

```bash
./run.sh --stage 4 --stop_stage 4
```

Stage 3で選んだ一つのチェックポイントを、train、speed perturbation、dev、testの
全splitへ共通して使用します。splitごとに別チェックポイントを使うことは禁止です。
コードブックIDの意味がsplit間で変わってしまうためです。

`src_case=rm`では連続する同一IDを除去します。

```text
入力:  utt1 10 10 10 4 4 31
出力:  utt1 10 4 31
```

`src_case=ts`ではフレーム単位のID列をそのまま保持します。

生成物は以下です。

```text
exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/train/pseudo_labels.txt
exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/train_sp0.9/pseudo_labels.txt
exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/train_sp1.1/pseudo_labels.txt
exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/dev/pseudo_labels.txt
exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/test1/pseudo_labels.txt
exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/test2/pseudo_labels.txt
```

### Stage 5: CJK変換と後段ASRデータ作成

```bash
./run.sh --stage 5 --stop_stage 5
```

ASR2のsource BPEで各cluster IDを一文字として扱えるよう、ID `0`から順番に
Unicode CJK文字へ対応付けます。

```text
0 -> U+4E00
1 -> U+4E01
...
```

この実装では最大20900 clustersまでに制限しています。`run.sh`の`nclusters`と
SSL configの`loss[0].conf.num_clusters`が一致しない場合は、学習開始前にエラーに
します。

生成されるASR入力例は以下です。

```text
dump/raw/train/text.rm.wavlm_large_l21_gumbel_km2000_ot_uniform
dump/raw/train/text.ts.en
```

source側はCJK文字を空白なしで連結した文字列です。target側は元のSAP transcription
です。speed perturbationを使う場合、元のtrainと各perturbation版を次へ統合します。

```text
dump/raw/train_sp/
  text.rm.wavlm_large_l21_gumbel_km2000_ot_uniform
  text.ts.en
  utt2spk
  feats_type
```

### Stage 6: 後段discrete ASR

```bash
./run.sh --stage 6 --stop_stage 6
```

内部ではSAP `asr2.sh`のstage 7--15を呼び出します。

| ASR2 stage | 処理 |
|---:|---|
| 7 | source/target token listとBPEの作成 |
| 8--10 | LM処理。`use_lm=false`のためskip |
| 11 | N-gram。デフォルトではskip |
| 12 | discrete ASR collect-stats |
| 13 | discrete ASR学習 |
| 14 | decoding |
| 15 | scoring |

後段モデルは`conf/train_discrete_asr_e_branchformer1_1gpu.yaml`です。これは
SAP `asr2`の既存設定を再利用しています。

後段だけを細かく再開する場合は、`asr3`のstageを6に固定し、ASR2側のstageを
指定します。

```bash
./run.sh \
    --stage 6 --stop_stage 6 \
    --downstream_stage 13 \
    --downstream_stop_stage 15
```

## 6. 主要ファイルと実装対応

| ファイル | 役割 |
|---|---|
| [`espnet2/layers/gumbel_kmeans.py`](../../../espnet2/layers/gumbel_kmeans.py) | Gumbel-Softmax K-means層、温度、mask、各損失 |
| [`espnet2/layers/sinkhorn.py`](../../../espnet2/layers/sinkhorn.py) | 既定一様・任意質量対応のPOT Sinkhorn divergence |
| [`espnet2/asr/encoder/identity_encoder.py`](../../../espnet2/asr/encoder/identity_encoder.py) | WavLM特徴を変更せずSSL lossへ渡すencoder |
| [`espnet2/ssl/loss/gumbel_kmeans.py`](../../../espnet2/ssl/loss/gumbel_kmeans.py) | SSLTask用loss、特徴分散正則化、unit抽出 |
| [`espnet2/ssl/espnet_model.py`](../../../espnet2/ssl/espnet_model.py) | SSLモデルの`encode_units()` |
| [`espnet2/tasks/ssl.py`](../../../espnet2/tasks/ssl.py) | `s3prl` frontendと`gumbel_kmeans` lossの登録 |
| [`espnet2/asr/gumbel_kmeans_asr_model.py`](../../../espnet2/asr/gumbel_kmeans_asr_model.py) | SSL + K-means + ASRのend-to-endモデル |
| [`espnet2/tasks/asr.py`](../../../espnet2/tasks/asr.py) | ASRモデル名`gumbel_kmeans`の登録 |
| [`espnet2/bin/gumbel_kmeans_extract.py`](../../../espnet2/bin/gumbel_kmeans_extract.py) | SSL/ASRチェックポイントからのunit抽出CLI |
| [`run.sh`](run.sh) | SAP実験の既定値 |
| [`asr3.sh`](asr3.sh) | stage 1--6の実行本体 |
| [`conf/train_ssl_kmeans_wavlm_large.yaml`](conf/train_ssl_kmeans_wavlm_large.yaml) | tokenizer学習設定 |
| [`run_e2e.sh`](run_e2e.sh) | E2Eの`fixed`・`quantizer`・`full`比較launcher |
| [`asr_e2e.sh`](asr_e2e.sh) | 標準ASR1 pipelineを呼ぶE2E用wrapper |
| [`conf/train_e2e_gumbel_kmeans_wavlm_large.yaml`](conf/train_e2e_gumbel_kmeans_wavlm_large.yaml) | ASR-only E2E学習設定 |

`asr3`の`local`、`asr2.sh`、`utils`、`scripts`などは、SAP `asr2`またはESPnet
templateへのsymlinkです。SAP固有データ準備と既存ASR2処理を重複実装しないための
構成です。

## 7. end-to-end SSL + K-means + ASR実装

`ESPnetGumbelKMeansASRModel`では、次のデータフローを実装しています。

```text
waveform
  -> SSL frontend
  -> Gumbel K-means
  -> ASR preencoder / encoder
  -> CTC / attention decoder
```

合計損失は次の形です。

```text
L_total = asr_weight        * L_ASR
        + kmeans_weight     * L_distortion
        + commitment_weight * L_commitment
        + diversity_weight  * L_diversity
        + ot_weight         * L_OT
```

SSL frontendを`freeze_param`へ指定しない限り、ASR損失とクラスタリング損失の両方が
SSL frontendを更新します。SAPでは`run_e2e.sh`から次の3条件を同じモデル構成で
切り替えます。

| mode | epoch 1--30 | epoch 31--90 |
|---|---|---|
| `fixed` | ASRのみ学習 | ASRのみ学習 |
| `quantizer` | ASRのみ学習 | codebook + ASRを学習 |
| `full` | ASRのみ学習 | 学習可能なSSL frontend + codebook + ASRを学習 |

`warmup_epochs=30`、`max_epoch=90`が既定値です。staged freeze対象はoptimizerと
DDPの構築後に一時的に`requires_grad=False`へ切り替えられ、epoch 31の開始時に
`True`へ戻ります。このため、別jobへ分ける場合と異なりoptimizer・scheduler・
epochの状態を維持したままunfreezeされます。`--warmup_epochs 0`ならepoch 1から
選択した全moduleを更新します。

現在のconfigは`freeze_param`でWavLMの畳み込みfeature extractorを恒久的に固定して
います。そのため`full`でepoch 31に解除されるのは、それ以外のWavLM parameterと
codebookです。

構音障害ASR論文に合わせ、CNN feature extractorの恒久freezeは維持しつつ、WavLMの
Transformer部分とK-meansを最初からfreezeしない条件は次のように実行します。

```bash
./run_e2e.sh \
  --e2e_mode full \
  --warmup_epochs 0
```

この場合はepoch 1からWavLM Transformer、codebook、ASRを同時に更新します。既定の
実験名は`..._full_warmup0_ep90_en`となり、段階的freeze条件とは別管理されます。

E2E側のcluster数は`run_e2e.sh --nclusters N`で指定します。この値はASRモデルの
`quantizer_conf.num_clusters`と既定の実験ディレクトリ名へ反映されます。初期centroidの
既定パスも次の規則で決まります。

```text
../asr2/exp/kmeans/wavlm_large_21_base_Nclusters/km_N.npy
exp/asr_e2e_wavlm_large_l21_gumbel_kmN_MODE_warmup30_ep90_en/
```

`fixed`だけはwarm-up後のunfreezeがないため、既定名は
`..._fixed_ep90_en`となります。scheduleを変えた実験が同じcheckpointを誤って
resumeしないよう、warm-up epoch数と総epoch数をディレクトリ名に含めています。

異なる配置のcentroidには`--initial_centroid_path`を使い、実験ディレクトリ全体を
指定する場合は`--asr_exp`を使います。launcherは学習開始前にcentroidの行数と
`--nclusters`が一致することを検証します。

`fixed`では、parameterのfreezeに加えてGumbel samplingも無効化します。parameterを
固定するだけでは親ASRモデルのtrain modeによりGumbel noiseが残るためです。
`quantizer`と`full`では、forwardがhard one-hot、backwardがsoft gradientとなる
straight-through Gumbel-Softmaxを使います。

また、現在のASR結合モデルにはSSL loss側で使っている`variance_weight`は未接続です。
初回比較はOndaらと同様にASR lossだけを使い、K-means・commitment・diversity・OTの
weightを0とします。追加lossはASR-only E2Eの結果を見た後のablationとします。

## 8. 動作確認手順

### 8.1 実行環境と依存関係

必ずレシピディレクトリから実行します。`path.sh`がESPnet用conda環境を有効化します。

```bash
cd /home/hojo/espnet_dsu/egs2/sap/asr3
. ./path.sh
python -c "import torch, s3prl; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

WavLMは初回に`ckpt`へダウンロードされるため、S3PRLが利用可能であることと、十分な
ディスク容量があることを確認してください。OTを有効にする場合はPOT 0.9.7系が
必要です。ESPnetの`recipe` extraへ追加済みですが、個別に入れる場合は次を実行します。

```bash
python -m pip install "POT>=0.9.7,<0.10"
python -c "import ot; print(ot.__version__)"
```

Python上のimport名はパッケージ名と異なり、小文字の`ot`です。GeomLossとPyKeOpsは
この実装では使いません。Stage 3は学習開始前にPOTをimportし、バージョンが
`>=0.9.7,<0.10`であることを確認します。

健常音声centroidも確認します。

```bash
python -c "import numpy as np; x=np.load('../../librispeech_100/asr2/exp/kmeans/wavlm_large_21_base_2000clusters/km_2000.npy'); print(x.shape)"
# (2000, 1024)
```

### 8.2 静的チェック

リポジトリルートで実行します。

```bash
cd /home/hojo/espnet_dsu
bash -n egs2/sap/asr3/run.sh egs2/sap/asr3/asr3.sh

tools/miniconda/envs/disorder/bin/python -m espnet2.bin.ssl_train \
    --config egs2/sap/asr3/conf/train_ssl_kmeans_wavlm_large.yaml \
    --token_list dummy \
    --print_config
```

`--print_config`の出力で、少なくとも次を確認します。

```text
frontend: s3prl
frontend_conf.frontend_conf.upstream: wavlm_large
frontend_conf.layer: 21
loss[0].name: gumbel_kmeans
loss[0].conf.num_clusters: 2000
loss[0].conf.commitment_weight: 0.0
loss[0].conf.diversity_weight: 0.0
loss[0].conf.variance_weight: 0.0
loss[0].conf.ot_weight: 0.01
loss[0].conf.ot_mass_mode: uniform
loss[0].conf.initialize_from_reference: true
```

### 8.3 単体テスト

```bash
cd /home/hojo/espnet_dsu
pytest -q -o addopts='' \
    test/espnet2/layers/test_sinkhorn.py \
    test/espnet2/layers/test_gumbel_kmeans.py \
    test/espnet2/ssl/test_gumbel_kmeans_loss.py \
    test/espnet2/ssl/test_ssl_gumbel_kmeans_model.py \
    test/espnet2/asr/encoder/test_identity_encoder.py \
    test/espnet2/asr/test_gumbel_kmeans_asr_model.py
```

実装時点では16テストが成功しています。確認内容は以下です。

- straight-throughで入力特徴とコードブックへ勾配が流れること
- paddingがunitと損失から除外されること
- distortion/commitmentが特徴次元で正規化されること
- eval時のunit抽出が決定的であること
- 特徴分散正則化がcollapsed featureを検出すること
- SSL encoderとK-meansを同時にbackwardできること
- HuBERT lossとK-means lossを併用できること
- ASR lossとK-means lossを同時にbackwardできること
- 同一centroid集合間のSinkhorn divergenceが0になること
- OT lossがSAP codebookだけに勾配を流し、健常centroidを更新しないこと
- 健常centroidの読込、固定buffer化、SAP codebook初期化が行われること

### 8.4 短い学習smoke test

Stage 1と2を完了させた後、通常実験とは別の出力先で少数iterationだけ動かします。

```bash
cd /home/hojo/espnet_dsu/egs2/sap/asr3

./run.sh --stage 1 --stop_stage 2

./run.sh \
    --stage 3 --stop_stage 3 \
    --ssl_tag debug_wavlm_gumbel_km2000_ot_uniform \
    --ssl_stats_dir exp/ssl_stats_wavlm_large_l21_gumbel_km2000_ot_uniform \
    --ssl_exp exp/ssl_debug_wavlm_gumbel_km2000_ot_uniform \
    --ssl_args "--max_epoch 1 --num_iters_per_epoch 10 --batch_bins 320000"
```

確認項目は以下です。

```bash
test -s exp/ssl_debug_wavlm_gumbel_km2000_ot_uniform/config.yaml
test -s exp/ssl_debug_wavlm_gumbel_km2000_ot_uniform/train.log
rg "gumbel_kmeans_(loss|perplexity|variance|temperature|ot)" \
    exp/ssl_debug_wavlm_gumbel_km2000_ot_uniform/train.log
```

### 8.5 本学習後の確認

最初はtokenizerまで実行します。

```bash
./run.sh --stop_stage 3
```

ログでは一つの値だけでなく、次を同時に確認します。

| 指標 | 見方 |
|---|---|
| `gumbel_kmeans_distortion` | 学習とともに下がるか。ただしこれだけではcollapseを判定できない |
| `gumbel_kmeans_commitment` | encoder特徴と選択中心の乖離 |
| `gumbel_kmeans_diversity` | 0に近いほどバッチ平均の利用率が一様に近い |
| `gumbel_kmeans_perplexity` | 1付近に張り付く場合はほぼ単一clusterへcollapse |
| `gumbel_kmeans_variance` | 0に近いほど各特徴次元が目標分散を満たす |
| `gumbel_kmeans_ot` | SAP codebookと固定健常centroidのSinkhorn divergence |
| `gumbel_kmeans_temperature` | 2.0から0.5へ向かって減衰するか |

perplexityが2000へ近づくこと自体を成功条件にはしません。発話内容と無関係に均等な
割当を作るだけでもperplexityは高くなるため、後段ASR精度、unitの継続長、音素との
対応などと合わせて評価する必要があります。

### 8.6 unit抽出後の確認

```bash
./run.sh --stage 4 --stop_stage 5

head -n 3 \
    exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/dev/pseudo_labels.txt

wc -l \
    dump/audio_raw/dev/wav.scp \
    exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/dev/pseudo_labels.txt \
    dump/raw/dev/text.rm.wavlm_large_l21_gumbel_km2000_ot_uniform \
    dump/raw/dev/text.ts.en
```

4ファイルのutterance数が一致することを確認します。IDは`0`以上`1999`以下である
必要があります。

決定性を確認する場合は、別の`unit_dir`へ再抽出して比較します。

```bash
./run.sh \
    --stage 4 --stop_stage 4 \
    --unit_dir exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform_rerun

cmp \
    exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform/dev/pseudo_labels.txt \
    exp/units_wavlm_large_l21_gumbel_km2000_ot_uniform_rerun/dev/pseudo_labels.txt
```

### 8.7 後段ASRの確認

```bash
./run.sh --stage 6 --stop_stage 6
```

確認対象は、source/target token list、ASR collect-stats、ASR checkpoint、test1/test2の
decode結果とscoreです。具体的なディレクトリ名は`asr_tag`から生成されるため、
`exp/asr_*`と各`train.log`を確認してください。

## 9. パラメータを変更するとき

### cluster数

cluster数は2箜所を同時に変更します。

```text
run.sh: nclusters
conf/train_ssl_kmeans_wavlm_large.yaml:
  loss[0].conf.num_clusters
```

値が一致しない場合は`asr3.sh`がエラーにします。`src_lang`と`ssl_tag`にもcluster数を
含め、異なる実験のtoken listやcheckpointが衝突しないようにしてください。

### WavLM layer

次を同時に変更します。

```text
frontend_conf.layer
ssl_tag
src_lang
```

### rm/ts

`src_case=rm`は連続重複を除去し、`src_case=ts`は全フレームを保持します。比較実験では
出力ファイル名、source BPE、ASR experiment名が混ざらないよう`asr_tag`も分けます。

### memory不足

まず`batch_bins`を下げ、必要に応じて`accum_grad`を上げます。WavLMをfine-tuneする
設定なので、offline feature extractionやfrozen WavLMより多くのGPU memoryが必要です。

## 10. 既知の注意点と未検証事項

- フルSAPデータを使った20 epochのGPU学習は、実装作業時点では未実行です。
- 学習率、loss weight、温度減衰、batch sizeは初期実験用の値であり、最適化済みでは
  ありません。
- 分散正則化はcollapseを緩和しますが、collapseしないことを保証しません。
- E2EのSAP codebookはSAP train-set centroidから初期化します。参照ファイルが存在しない場合や、
  cluster数・特徴次元が異なる場合はモデル構築時にエラーになります。
- 2000 cluster同士のOTは`2000 x 2000`のコスト行列を使うため、従来設定よりGPU
  memoryと計算時間が増えます。memory不足時はbatchだけでなく`ot_iterations`も下げて
  比較してください。POTが未収束のwarningを出す場合は、反対に`ot_iterations`を増やす
  か`ot_epsilon`を調整し、OT値と計算時間を確認してください。
- WavLMの初回download、S3PRL、CUDA driver、GPU memoryは実行環境側で確認が必要です。
- `asr3`のtokenizer学習にはspeed perturbation音声を含めていません。unit抽出と後段
  ASR学習には含めています。
- end-to-end ASRモデルとSAP用設定は構築・forwardまで検証済みですが、フルデータでの
  長時間学習とWER評価は未実施です。

## 11. 推奨する初回確認順序

1. `path.sh`を読み、`torch`、`s3prl`、CUDAを確認する。
2. `--print_config`でWavLM layer、cluster数、loss weightを確認する。
3. 単体テストを実行する。
4. Stage 1と2を実行し、データ数とshapeを確認する。
5. 10 iterationのsmoke testを行う。
6. tokenizer本学習を行い、perplexity・variance・distortion・OTを確認する。
7. Stage 4と5でunit数、ID範囲、決定性を確認する。
8. Stage 6で後段ASRを学習し、既存offline K-means方式とWERを比較する。
9. `run_e2e.sh --e2e_mode fixed`でE2E構成内の固定tokenizer baselineを学習する。
10. fixed checkpointから`quantizer`と`full`を分岐し、WERとtoken統計を比較する。
