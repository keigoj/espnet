# RTX 4090 × 1 用のE2E学習設定

現行の`run_e2e_4090_amp.sh`は、新規実験向けに最初の30 epochでWavLMと
codebookを凍結し、31 epoch目に解除します。温度は凍結中2.0を維持し、解除後の
forwardから減衰します。LRスケジューラはリセットしません。新規実験の実行例は
[レシピREADME](../README.md)を参照してください。以下の試走・再開記録は、
凍結なしの旧条件です。その条件を再開する場合は`--warmup_epochs 0`と既存の
`--asr_exp`を明示してください。

> 長時間実行での更新：160万binsの設定は2 epoch目にOOMとなりました。
> 再開には下記の120万bins設定と専用ランチャーを推奨します。
> 以下の160万binsの測定値は短い試走時の結果です。

`train_e2e_gumbel_kmeans_wavlm_large_4090_amp.yaml` は、RTX 4090
24 GB / PyTorch 2.6.0+cu118 で試走した設定です。元のYAMLは変更していません。

```bash
cd /home/hojo/espnet_dsu/egs2/sap/asr3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./run_e2e.sh \
  --stage 11 --stop_stage 11 \
  --ngpu 1 --e2e_mode full --warmup_epochs 0 \
  --asr_config conf/train_e2e_gumbel_kmeans_wavlm_large_4090_amp.yaml \
  --asr_exp exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_amp4090_en
```

既存のデータと統計を使う、新規実験用のコマンドです。今回は長時間の本学習は
開始していません。`PYTORCH_CUDA_ALLOC_CONF` はPython起動前に設定してください。

## 主な設定

| 項目 | 設定 |
|---|---|
| ASRのAMP | `use_amp: true` |
| WavLMのAMP | `model_conf.autocast_frontend: true`（このGPUではBF16を自動選択） |
| 実バッチの上限 | `batch_bins: 1600000` |
| 勾配累積 | `accum_grad: 9` |
| AdamW | `fused: true`、学習率は元と同じ `0.0001` |
| データ読み込み | `num_workers: 4` |
| attention図 | `num_att_plot: 0` |
| cuDNNの探索 | `cudnn_benchmark: false`（可変長の多数の形状） |

実バッチは平均11.40発話、1更新あたり平均102.63発話です。
元ログの800,000 bins × 累積16では平均99.63発話／更新でした。
新設定では43,829バッチ／epoch、約4,869更新／epochとなり、
元の80,268バッチ・約5,016更新に近い学習率スケジュールを維持します。

量子化の温度はforward回数で減衰するため、`temperature_decay` を
`0.999990843068` に調整しました。学習1 epochでの減衰量は元ログとほぼ同じです。

## GPU試走の結果（2026-10-14）

実際のtrain_sp音声とBPEテキストを使用し、WavLM・量子化・ASRのforward、
backward、勾配クリップ、AdamW更新まで実行しました。1 epoch全体の完走や
WERの比較は未実施です。

| 条件 | 秒／バッチ | バッチ数／epoch | 学習部分の時間 |
|---|---:|---:|---:|
| 元ログの条件：800,000 bins、frontend FP32、累積16 | 0.1840 | 80,268 | 約246分（40バッチから推定） |
| 推奨：1,600,000 bins、frontend BF16、累積9 | 0.1856 | 43,829 | 約136分（160バッチから推定） |
| 元の実際の学習ログ | — | 80,268 | 約262分（実測） |

推奨条件では、発話長順に等間隔で選んだ160バッチに加え、
`発話数 × 最大音声長` と `発話数 × 最大音声長²` が大きいバッチを
それぞれ12件、計24バッチ追加しました。全184バッチでlossと更新時の勾配ノルムは
有限値でした。追加24バッチは時間の外挿から除き、メモリ測定には含めています。

最大GPU割り当てメモリは21.22 GiB、予約メモリは23.06 GiBでした。
データは時間測定前に読み込んでおり、読み込み時間・validation・保存時間は
推定に含みません。元ログでは読み込み待ちは約0.09 ms／バッチでした。
本学習は学習部分で約2時間15分〜2時間30分／epochを目安としてください。

2,000,000 binsは等間隔の160バッチでは通りましたが、追加した高負荷バッチで
OOMになりました。2,400,000・2,800,000 binsもOOMとなったため採用していません。
1,600,000 binsの推定時間は2,000,000 binsとほぼ同じでした。

## AMP使用時の距離計算

`espnet2/layers/gumbel_kmeans.py` の距離計算は、FP32への変換に加えて
局所的にautocastを無効化しました。変換だけでは行列積が再び低精度になり、
近接する重心間の距離が両方ゼロになる例を確認しています。
WavLMとASRはAMPを使い、距離計算をFP32で維持します。

この変更の回帰テストを追加しました。量子化・ASR関連の14テストは成功しています。
既存のOTテスト1件はテスト用Python環境にPOTがないため除外しています。
学習用Python環境でも距離・重心選択・backwardの回帰チェックを実行しました。

参考：[PyTorchのAMPとFP32領域の指定](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、
[PyTorchのoptimizer実装](https://docs.pytorch.org/docs/stable/optim.html)。

## 長時間実行後のOOMからの再開

160万bins / 累積9の本学習は、1 epochの学習部分を約2時間26分で完走した後、
2 epoch目の約32,000バッチでbackward中にOOMとなりました。
エラー時は18.32 GiBが割り当て済み、4.47 GiBが未使用の予約メモリで、
522 MiBの追加割り当てに失敗しました。メモリ断片化が疑われます。
終了したプロセスでallocator環境変数が適用されていたかは、ログだけでは確定できません。

再開用に `train_e2e_gumbel_kmeans_wavlm_large_4090_amp_1200k.yaml` と
`../run_e2e_4090_amp.sh` を追加しました。ランチャーはPython起動前に
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定し、起動時に表示します。

```bash
cd /home/hojo/espnet_dsu/egs2/sap/asr3
./run_e2e_4090_amp.sh \
  --stage 11 --stop_stage 11 \
  --asr_exp exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup0_ep90_en
```

`checkpoint.pth` の保存epochは1です。モデル・optimizer・scheduler・scalerを
読み込み、2 epoch目の先頭から再開します。2 epoch目途中の更新は保存されていません。
今回は本学習を再起動していません。

変更は120万bins / 累積12で、平均106.26発話／更新です。
学習率、モデル構造、量子化の温度減衰は元のAMP設定を維持します。
温度減衰を変えると保存済み `num_updates` から計算される再開時の温度が変わるため、
ここでは変えていません。バッチ数は56,443／epochとなるので、温度はepoch単位では
以前より速く下がります。

実際の保存済みチェックポイント（モデル・optimizer・scaler）を読み込んで、
通常160バッチ＋高メモリ負荷24バッチを試走しました。
lossと更新時の勾配ノルムはすべて有限値です。
最大割り当て17.53 GiB、最大予約18.81 GiBで、160万binsの短い試走より
割り当てメモリに約3.69 GiBの余裕ができました。
短い試走からの学習時間の外挿は約148分／epochで、長時間の完走は未検証です。
測定値は `benchmark_4090_amp_1200k_resume.json` に保存しています。
