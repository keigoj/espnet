# E2E学習の検証結果（2026-11-10）

現行4090ランチャーが選ぶ1200k設定を使い、RTX 4090上で実データの学習更新に
成功した。単一層の「30 epoch凍結 → 60 epoch共同最適化」というフローは確認
できた。ただし、元論文の全実験条件の完全再現ではない。

## 論文・参照実装との対応

論文は保存済み[Onda et al. 2025](../../../doc/references/2505.16207.pdf)
（arXiv:2505.16207v1）を参照した。実装はユーザー指定commit
[`0c7942d1e0db7828f8adb2154561c28542ae7806`](https://github.com/espnet/espnet/tree/0c7942d1e0db7828f8adb2154561c28542ae7806)
を参照した。この後発ESPnet実装のデフォルトを、論文に記載された実験条件と
同一視しない。

|項目|確認結果|根拠・留意点|
|---|---|---|
|特徴抽出|WavLM-large、単一21層、D=1024|論文§3.1–3.2|
|クラスタ数・初期化|K=2000、既存K-means centroidで初期化|論文Table 1のクラスタ数に対応。GPUモデル構築後も元のnpyと完全一致|
|距離・係数|単位ノルム化なし、二乗Euclidean、logit scale=1|論文式(1)、§3.1、参照実装の`_distance_logits`|
|割当|学習時hard one-hot、backwardはsoft GumbelのStraight-Through|論文式(2)、参照quantizerの`forward`。数値・勾配比較も実施|
|ASRへの接続|割当(B,T,K) → 独立embedding(E=512) → 位置符号化 → ASR|参照`TokenEmbedding` / `ASRLoss`に対応。512次元等は論文ではなく実装既定値|
|最初の30 epoch|WavLMとcentroidを凍結、embeddingとASRを更新|論文§3.2。GPUではepoch=30の状態を検証|
|残り60 epoch|epoch=31でWavLMの対象部分とcentroidを解除|論文§3.2。ただしCNN恒久凍結は下記の差として残る|
|凍結中・評価時|Gumbelなし、決定的最近傍、凍結中のtokenizerはno-grad|参照`SpeechTokenizer`に対応。凍結期間の指定は参照実装のforward数に対し、こちらはepoch数|
|ASR損失|CTC/attention、K-means等の補助loss weightは0|論文§3.1のα=0に対応。CTC比率0.3そのものの一致は論文本文からは確認できない|
|温度|初期2.0、最低0.1、減衰0.999995 / trainable forward|初期2.0は論文§3.1。下限・減衰率・凍結中の温度停止は後発実装との対応|

Bは発話数、TはSSLフレーム数、DはSSL特徴次元、Kはクラスタ数、Eはembedding次元。
有効フレームの割当はK軸の和が1、paddingは全K成分が0であることも確認した。

### 完全再現と呼べない差

- 論文はLibriSpeech-100とWER、現行実験はSAPとCER/WERであり、データ・文字の
  tokenization・ASR構成・optimizer・LR・バッチ設定の全面一致は保証しない。
- 論文§3.2のfull finetuneはθsslも更新する条件だが、現行設定はCNN feature
  extractorを恒久凍結する。更新対象は主にWavLM Transformer側である。
- decodingは論文§3.1のbeam size=5に対して、現行`decode_e2e_ctc0.3.yaml`は20。
- 4090ランチャーの1200k設定にはTime Maskがある。論文本文だけではこの詳細を
  確認できない。base / 1600k AMP設定はSpecAugなしである。
- SIR用YAMLの`sir_weight=0.1`は2025年論文のASR-only条件に追加する別実験。
  今回GPUで試した1200k設定はSIRなし。
- 距離計算のFP32保護、padding ID=-1、旧API維持はこのリポジトリ側の処理。
  参照実装のpadding IDは0だが、ASRへ渡すpadding割当は双方とも全成分0。

上記の実験条件・ハイパーパラメータは、この検証では変更していない。

## GPU実行テスト

- GPU: NVIDIA GeForce RTX 4090、24 GiB
- 環境: `tools/miniconda/envs/disorder/bin/python`、PyTorch 2.6.0+cu118
- 設定: `conf/train_e2e_gumbel_kmeans_wavlm_large_4090_amp_1200k.yaml`
- `batch_bins=1200000`、`accum_grad=12`、BF16 AMP、GradScaler、fused AdamW、
  WarmupLR、grad clip=5を維持。既存の`Trainer.train_one_epoch`で実行。
- 実際の学習samplerから、中間長の9発話バッチ（最大約6.5秒）と最長側の
  3発話バッチ（最大約30秒）を選び、交互に12 microbatchを実行。
- 初期化した新モデルでepoch=30 / 31の状態を模擬し、各1 optimizer更新を実行。
  **実際に30 epoch学習したモデルの評価ではない。** LR schedulerの位置も境界相当
  に設定した。既存checkpointは読み込まず、上書きもしない。

|確認内容|epoch=30の凍結状態|epoch=31の共同学習状態|
|---|---|---|
|embedding / encoder / decoder|有限・非ゼロ勾配、実際の重み更新あり|同左|
|WavLM Transformer / centroid|勾配なし、重み変更なし|有限・非ゼロ勾配、実際の重み更新あり|
|CNN|勾配なし、重み変更なし|同左（恒久凍結）|
|temperature counter|0のまま|0 → 12（optimizer更新数ではなくforward数）|
|温度|全forwardで2.0|2.0から0.999995倍ずつ減衰|
|optimizer状態|embedding等のstep=1、tokenizer側は0|embedding等のstep=2、tokenizer側は1|
|最大割当メモリ|約3.63 GiB|約15.28 GiB|
|最大予約メモリ|約5.45 GiB|約17.06 GiB|

両段階ともlossと勾配ノルムは有限で、optimizer更新のskipやOOMはなかった。
評価時は同一入力から同一embeddingが得られ、温度counterは進まなかった。
量子化出力の形状は(B,T,K)=(9,326,2000)および(3,1500,2000)、ASRへの入力は
(B,T,E=512)。テンソルはCUDA上にあることも確認した。

これは短時間の実行テストであり、90 epoch完走・学習の収束・CER改善を保証する
ものではない。A6000複数GPU用base/SIR設定のDDP実行と、そのバッチ設定のメモリ
使用量は今回の1 GPUテストでは検証していない。

実測値: [gpu_smoke.json](exp/e2e_training_check_20261110/gpu_smoke.json)

### 再実行

```bash
cd /home/hojo/espnet_dsu/egs2/sap/asr3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/hojo/espnet_dsu \
/home/hojo/espnet_dsu/tools/miniconda/envs/disorder/bin/python \
  local/check_e2e_training.py \
  --output exp/e2e_training_check_20261110/gpu_smoke.json
```

## その他のテスト

1. 関連pytest: **70 passed、8 skipped、1 deselected**。CPU環境で、staged freeze、
   checkpointの温度counter保持、独立embedding、SIR、旧モデル互換、起動スクリプト
   を確認した。8 skipはCUDA用SIRテスト、1除外はCPUテスト用PythonのPOT不足。
   GPU実行確認は上記の学習用Pythonで別途行った。
2. 4種類の現行YAMLを学習用ASRTask parserで読み込み、距離・温度・embedding・
   loss weight・WavLM層をassertで確認した。
   [config_audit.json](exp/e2e_training_check_20261110/config_audit.json)
3. 指定commitの実際のquantizerクラスと、CPU/GPUそれぞれでforward count
   0 / 1 / 10000 / 1000000を比較した（合計8ケース）。新しいtok基底クラスと
   出力コンテナはこのcheckoutにないため置き換え、type-check用decoratorを除去
   したが、quantizerの数式は変更していない。同じcentroid・入力・乱数で、hard
   割当と最近傍割当は一致した。soft値の最大絶対差は約5.4e-7、勾配の最大絶対差は
   約9.2e-5（温度0.1のとき）。距離の加算順序によるFP32丸め差を考慮し、勾配は
   `rtol=1e-4, atol=1e-5`で一致した。温度と更新counterも一致した。
   [upstream_comparison.json](exp/e2e_training_check_20261110/upstream_comparison.json)

pytest再実行（リポジトリroot、CPUテスト用Python）:

```bash
python -m pytest -o addopts='' -q \
  test/egs2/sap/test_run_e2e.py \
  test/espnet2/asr/frontend/test_token_embedding.py \
  test/espnet2/asr/test_gumbel_kmeans_asr_model.py \
  test/espnet2/train/test_trainer.py \
  test/espnet2/layers/test_gumbel_kmeans.py \
  -k 'not loads_fixed_reference_and_initializes_codebook'
```

## 文書の不一致の訂正

前回READMEは4090ランチャーの説明を`specaug: null`としたが、実際に選ばれる
1200k YAMLはTime Maskありだった。base / 1600k AMP YAMLの設定と混同していた
ため、READMEと対応表を訂正した。設定ファイルと学習コードは変更していない。
