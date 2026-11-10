# ESPnet本体の微分可能K-meansとの対応

比較先はユーザー指定のcommit
[`0c7942d1e0db7828f8adb2154561c28542ae7806`](https://github.com/espnet/espnet/tree/0c7942d1e0db7828f8adb2154561c28542ae7806)
に固定する。2025年論文の実験configそのものとの一致を主張するものではない。

## クラスタ割当を選ぶ段階

```text
waveform (B, samples)
  -> WavLM layer 21: features (B, T, D=1024)
  -> quantizer: cluster assignment (B, T, K=2000)  <-- 最近傍 / Gumbelの選択箇所
  -> independent embedding: (B, T, K) @ (K, E=512) -> (B, T, E)
  -> positional encoding
  -> existing ASR encoder / decoder
```

`B`は発話数、`T`はSSLフレーム数、`D`はSSL特徴次元、`K`はクラスタ数、
`E`はASR用embedding次元。割当確率はK軸で正規化し、paddingフレームの割当は
全て0にする。ASRには有効フレーム長`(B,)`も渡す。

|段階|quantizer内の割当|学習対象|
|---|---|---|
|epoch 1--30（凍結）|最小二乗距離のcentroidを決定的に選択。乱数なし、tokenizerはno-grad|ASR用embeddingとASR|
|epoch 31--90（共同学習）|負の二乗距離logitにGumbel noiseを加え、hard one-hotを選択。backwardはsoft gradient|modeで選んだtokenizer部分、embedding、ASR|
|validation / decoding|決定的な最近傍。乱数なし|なし|

`fixed`は全epochでtokenizerを凍結する。`quantizer`は31 epochでcodebookだけを
解除し、`full`は学習可能なWavLM部分も解除する。CNN feature extractorの恒久
凍結は既存設定を維持する。単にencoderのparameterを固定しただけでsamplingが
止まるわけではないため、`model: tokenized_asr`の専用クラスがこの切り替えを
担当する。凍結時のsamplingを設定する追加フラグは不要である。

## 揃えた項目と根拠

|項目|今回の新規E2E設定|比較先|
|---|---|---|
|距離|二乗Euclidean、単位ノルム化なし|`DifferentiableKMeans._distance_logits`|
|距離係数|`logit_scale: 1.0`|`sigma_squared: 1.0`|
|初期温度|2.0|`temperature_init: 2.0`|
|最低温度|0.1|`temperature_floor: 0.1`|
|温度減衰|0.999995 / trainable forward|`temperature_decay: 0.999995`|
|forward / backward|hard one-hot / straight-through soft gradient|quantizerの`forward`|
|凍結中|no-gradの決定的最近傍|`SpeechTokenizer.forward`の凍結分岐|
|ASR入力|centroidと独立した512次元embedding、位置符号化あり、dropout 0.1|`TokenEmbedding`、`GANSpeechTokenizerTask`の`asr_embedding_conf`既定値|

根拠となるコード:

- [quantizer](https://github.com/espnet/espnet/blob/0c7942d1e0db7828f8adb2154561c28542ae7806/espnet2/tok/quantizer_differentiable_kmeans.py)
- [凍結・推論の切り替え](https://github.com/espnet/espnet/blob/0c7942d1e0db7828f8adb2154561c28542ae7806/espnet2/tok/tokenizer.py)
- [ASRへの接続](https://github.com/espnet/espnet/blob/0c7942d1e0db7828f8adb2154561c28542ae7806/espnet2/tok/asr_loss.py)
- [独立embedding](https://github.com/espnet/espnet/blob/0c7942d1e0db7828f8adb2154561c28542ae7806/espnet2/tok/token_embedding.py)
- [embeddingの既定設定](https://github.com/espnet/espnet/blob/0c7942d1e0db7828f8adb2154561c28542ae7806/espnet2/tasks/gan_tok.py)

## 維持した条件と残る差

- SAPデータ、K=2000、WavLM layer 21、ASR構成、optimizer、LR schedule、batch設定、
  SpecAug設定、既存loss weight、decode設定は変更していない。SIR用YAMLも同じ
  tokenizer / embedding設定に揃えるが、SIRの設定値は維持する。
- 比較先の凍結長はforward数`freeze_steps`、こちらは既存のepoch指定を維持する。
  最初30 epochを凍結する点は2025年論文§3.2の単一層条件に対応する。比較先と
  データ量・バッチ構成が異なるため、epochごとの温度推移の一致は保証しない。
- 距離計算のFP32保護、padding ID=-1、補助損失とSIRの既存APIは維持する。
  比較先の`encode`はpadding ID=0だが、割当分布のpaddingは双方とも0である。
- 共有quantizerクラスの`min_temperature=0.5`等のデフォルトは、旧checkpointと
  offline SSL実験の互換性のため変更しない。新規E2E YAMLで0.1を明示する。

2025年の[保存済み元論文](/home/hojo/espnet_dsu/doc/references/2505.16207.pdf)では、
式(1)と§3.1に距離係数1.0・初期温度2.0が明記される。一方、温度下限・減衰率・
凍結中の温度処理は未記載であり、上の数値はESPnet実装のデフォルトとの対応である。

## 既存モデルと30 epoch基準モデル

新規実験の既定名は`..._km2000_tokembed512_full_warmup30_ep90_en`。
旧centroid入力モデルと同じディレクトリへ自動resumeしないよう分ける。
新構成は`model: tokenized_asr`で常に独立embeddingを使う。旧構成は従来の
`model: gumbel_kmeans`をそのまま使い、旧クラス内に新旧切り替えの条件分岐を
持たせない。旧実験の再開時は、その実験の保存済み`config.yaml`を使う。

```bash
cd egs2/sap/asr3
./run_e2e_4090_amp.sh --stage 11 --stop_stage 11 \
  --warmup_epochs 30 --max_epoch 30
```

この新規実験は全30 epochでtokenizerを凍結し、embeddingとASRを学習する。
最後の`checkpoint.pth`はモデル・optimizer・scheduler・scaler・epoch情報を含む。
まだこの学習は実行していない。旧モデルには独立embeddingが存在せず、encoderの
入力次元も異なるため、その30 epoch重みを新構成へそのままstrict loadできない。

## 文書の訂正

base / 1600k AMP YAMLは`specaug: null`だが、4090ランチャーが選ぶ1200k YAMLは
Time Maskありである。前回READMEを一律に`specaug: null`と説明したのは誤り
だったため、設定を変えず、今回の検証でプリセットごとの説明に訂正した。
また、4090試走記録の温度減衰率0.999990843068は旧条件の履歴として残し、
新規実験の現行値0.999995と区別する。
