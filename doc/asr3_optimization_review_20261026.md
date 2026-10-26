# asr3 全体最適化のWER悪化に関する調査

調査日: 2026-10-26。学習コード・既存設定・既存checkpointは変更していない。

後続修正（同日）: 以下は修正前の調査記録。ユーザー指定により、その後E2E用3設定の距離係数を1.0に変更し、Gumbelなし・hard=Trueの学習経路にもstraight-throughを追加した。保存済み実験のconfig・checkpointは変更していない。EmbeddingとTime Maskの追加は未実施。

## 対象と結論

- 通常版: `egs2/sap/asr2/exp/asr_wavlm_large_21_base_lam0p0_km2000_ts_char_en`
- 全体最適化: `egs2/sap/asr3/exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup0_ep90_en`

最も疑わしいのは、ASRが未学習の段階からWavLMを更新した結果、トークン利用が少数のクラスタへ偏り、正則化の少ないASRが訓練データに過適合していることである。これは原因を確定した結果ではなく、ログと小規模診断に基づく優先仮説である。通常版との入力表現や最適化条件も異なるため、現在の比較では全体最適化だけの効果は測れていない。

## 保存済みの評価結果

| 条件 | checkpoint | test1 WER | test2 WER |
|---|---|---:|---:|
| 通常discrete ASR | valid.acc上位10個の重み平均 | 18.3 | 19.2 |
| E2E | 18epoch単体、valid.acc.best | 19.7 | 21.0 |
| E2Eの悪化幅 | percentage points | +1.4 | +1.8 |

各実験の`decode_*/test{1,2}/score_wer/result.txt`のSum/Avgを確認した。test1/test2の単語数・発話数も一致する。decode設定は両方ともbeam=20、CTC weight=0.3、LMなし。

両方式のtrain_spは499,815発話、devは24,848発話で、ID集合と文字列の転記内容は一致する。ターゲットのBPEモデルとtoken_listは別ファイルで、token_listの順序にも差がある。Embedding等の重みを移植する場合には、文字列に基づく対応付けが必要。

E2Eの実験名にはep90とあるが、現在のtrain.logで完了を確認できるのは44 epochまで。単体対重み平均という評価条件差も残るため、上記の悪化幅をすべて全体最適化のせいとは断定できない。

## 1. トークン利用の偏りと過学習：最優先

保存済みWavLM事前学習モデル＋初期SAP centroidと、E2Eの18epoch.pthを、同じdev音声で比較した。

- 異なる16話者から1発話ずつ、長さ1〜6秒を選択。
- 合計73.11秒、3,661フレーム。
- frontendをeval、FP32、1発話ずつ実行し、最近傍のhard IDを測定。
- 全devの代表値やWER改善を示す実験ではない。話者・文章の網羅性にも限界がある。

| 指標 | 初期モデル | E2E 18epoch |
|---|---:|---:|
| 使用クラスタ数 / 2000 | 850 | 163 |
| hard IDの周辺分布perplexity | 616.6 | 31.2 |
| 最頻クラスタのフレーム占有率 | 1.17% | 23.44% |

同じ音声でも分布が大きく偏る。ASRに必要な情報へ集約された可能性と、情報を失う部分的collapseの可能性を区別する必要がある。使用クラスタ数だけでcollapse確定とはしない。次に全devのhard ID頻度、発話内の連続長、音素との対応、話者別・重症度別WERを測ると切り分けられる。

既存ログも過学習と整合する。

| epoch | train loss_asr | valid loss_asr | train acc | valid acc |
|---|---:|---:|---:|---:|
| 10 | 4.357 | 11.557 | 0.936 | 0.847 |
| 18 | 2.528 | 12.190 | 0.963 | 0.854 |
| 40 | 0.944 | 14.445 | 0.988 | 0.850 |

ログのtrain `loss`は勾配蓄積によるスケールの影響があるため、ここでは`loss_asr`を比較した。ログの`wer`は今回のbeam decode結果と同一指標として扱っていない。

`gumbel_kmeans_perplexity`は`espnet2/layers/gumbel_kmeans.py:312`で計算するsoft posteriorのバッチ周辺分布に基づく。上表のhard IDの全サンプル周辺分布とは異なり、使用クラスタ数でもない。ログの約40を「40クラスタしか使わない」と読み替えてはいけない。

## 2. 学習開始直後からWavLMとcentroidを更新している

`egs2/sap/asr3/run_e2e_4090_amp.sh:12`が`--warmup_epochs 0`を指定する。保存済みconfigも`staged_freeze_param: []`、`staged_freeze_until_epoch: 0`であり、実際に段階的凍結を使っていない。ランダム初期化のASRから、直ちにWavLMへ勾配が入る。

Ondaらの単一層実験は、90 epochのうち最初の30 epochでSSLとcentroidを固定し、後半60 epochで更新する。単一層K=2000ではWERが4.4/7.2から4.2/6.8へ改善しているが、SAPでの改善を保証するものではない。小さいKでの改善幅とも区別する必要がある。[論文1、§3.1–3.2/Table 1](https://arxiv.org/pdf/2505.16207)

優先する対策は固定tokenizerでASRを先に学習し、そのcheckpointからfixed継続とfullを分岐すること。full側ではWavLM用LRを下流ASRより小さくする。候補としてASR `1e-4`を維持してWavLM `1e-5`等から比較する。これは本調査からの提案であり、論文指定値ではない。

注意：現在のstaged freezeはrequires_gradだけを切り替える（`espnet2/train/trainer.py:193`）。quantizerがtrainモードなら凍結中もGumbelノイズと温度減衰は続く。`--warmup_epochs 30`だけでは、通常の決定的な固定トークン学習にはならない。ウォームアップ中のsampling方針も明示して揃える必要がある。

## 3. 通常版のEmbeddingとTime MaskがE2Eにはない

通常版は以下の経路である。

`hard ID -> 学習可能な2000クラス用Embedding(512次元) + 位置符号/dropout -> Time Mask -> ASR`

E2Eは以下である。

`WavLM -> straight-through hard assignment H -> H @ centroid M(1024次元) -> ASR`

根拠は`egs2/sap/asr2/conf/train_discrete_asr_e_branchformer1_1gpu.yaml:8`、`espnet2/mt/frontend/embedding.py:38`、`espnet2/asr/gumbel_kmeans_asr_model.py:128`、`espnet2/layers/gumbel_kmeans.py:289`。

E2Eでは、クラスタ境界を決めるMがASRへの入力ベクトルも兼ねる。通常版には境界と独立して調整できるEmbedding Eがある。これはそれ自体が計算バグという意味ではないが、「同じASRに全体最適化だけを追加した比較」ではない。

補足（元論文と射影の再確認）: E2Eにもencoder入口の`conv1d2`に学習可能な1024→256次元のConv1d変換が存在する（`espnet2/asr/encoder/e_branchformer_encoder.py:269`、`espnet/nets/pytorch_backend/transformer/subsampling.py:125`）。従って、ASR向け射影全般が欠けているわけではない。差分はcentroidとは独立したトークンEmbeddingの有無である。Ondaらの式(4)はDiffKM出力をASRへ渡す表記だが、その間のEmbedding/射影の詳細・次元は本文で明記されていない。また、式(3)のH @ Mはk-means再構成損失の定義であり、それだけを根拠にASR入力もH @ Mだとは断定できない。独立512次元Embeddingの追加は、論文の明記された層を復元する修正ではなく、通常版との比較条件を揃えるための実験候補として扱う。

比較用には`H @ E`をASR入力として、MとEを分ける構成が有力。hard IDをEmbeddingへ直接渡すだけでは上流へ微分できないので、ASRまでstraight-throughのHを使う必要がある。512次元、位置符号、dropout、Time Maskも通常版に揃え、まずfixed条件が通常版に近づくか確認する。

通常版にはTime Maskが10個、幅比率0〜0.05で入る。一方、E2E設定は`specaug: null`。過学習の観測とも合うため、Time Maskの復元は優先度が高い。

## 4. Gumbelの距離スケールと温度の役割

実装のlogitsは`-logit_scale * squared_distance`。実験値は`logit_scale=1/1024`である。論文1は距離に対する係数を1、温度を2から減衰させ、k-means補助損失の係数を0としている。ただし、特徴量スケールが同じとは確認できておらず、数値だけの直コピーは適切とは限らない。[論文1、式(1)–(4)/§3.1](https://arxiv.org/pdf/2505.16207)

同じ16発話から、Gumbelで選んだhard IDが決定的最近傍と一致する確率を解析的に求めた。各フレームの`max softmax(logits)`を平均した値であり、Monte Carlo推定ではない。

| 距離係数 | 初期モデルの一致率 | 18epochモデルの一致率 |
|---|---:|---:|
| 1/1024（現在） | 90.01% | 86.77% |
| 1/32（診断用） | 99.74% | 99.64% |
| 1（診断用） | 99.984% | 99.995% |

現在は学習時に約10〜13%のフレームで最近傍以外を選ぶ計算になる。これはGumbel法の意図された確率化だが、今回の特徴スケールで必要な強さかは調整余地がある。1/32や1が良いWERを与えるという検証はしていない。特に1ではsoft assignmentが飽和し、上流勾配が非常に小さくなるおそれがある。

`hard=True`の選択は`argmax((logits + G)/tau)`。正のtauで割ってもargmaxは変わらないため、温度を下げてもhard IDのノイズは減らない。tauは主にbackwardのsoft勾配を調節し、距離係数はGumbelに対するlogitsの強さも変える。この2つを分けて調整する。

温度カウンタはoptimizer.stepではなくforwardごとに増える（`espnet2/layers/gumbel_kmeans.py:330`）。現在の減衰率では約151,392 forwardsで2から0.5へ到達し、ログでは3 epoch目から下限。accum_grad=12の場合、optimizer更新数と約12倍異なる。スケジュールの単位を明示し、全体最適化を開始する時点から減衰させる候補を比較する。

重要な実装上の注意：現在の`sample_gumbel=False, hard=True`はargmax→one_hotであり、straight-throughではない（同ファイル280–282行）。この設定のままfull最適化すると、ASRからWavLMへの勾配が切れる。CPUの小規模backwardで、Trueでは入力勾配あり、Falseでは入力勾配Noneを確認した。決定的STを使うなら`one_hot - soft.detach() + soft`等の別経路を実装する必要がある。

## 5. 学習率・更新数・checkpoint選択

通常版はAdamの最大LR=0.006、warmup=15000、accum=8。E2EはAdamWの最大LR=0.0001、warmup=5000、accum=12で、WavLM・centroid・ASRを同一LRで更新する。

LRの数値だけで不足とは言えない。実際にE2Eのtrain loss_asrは小さくなっており、単純な学習不足よりも汎化が問題である。全体LRを上げるより、WavLMの更新を遅らせる・小さくする実験を優先する。バッチ計算に使う入力shapeは、通常版では離散系列長と語彙数（例: 181,2003）、E2Eでは音声サンプル数（同じ発話で57760）なので、batch_binsの数値も直接比較しない。

E2Eでは18epoch時点でgrad_norm=16.734、clip=99.957%とほぼ全更新で勾配クリップがかかる。ただし全体ノルムだけでSSLが原因とは断定できない。次にfrontend/quantizer/ASRごとの勾配ノルム・更新量を記録するとよい。

既存モデルの比較はcheckpointの選び方を揃え、候補選択はdevの実decode WERで行う。重み平均による回復量は未測定である。量子化境界も平均で変わり得るので、平均が必ず良くなるとは仮定しない。

## 6. 2本目の論文で参考にすべき点

Wangらは、対象ドメインに適応したHuBERTとbottleneckを使い、pseudo-label refinementを経てE2Eを学習する。CNNは固定し、温度は0.8、Kは100/500。k-means損失は初期DKM学習のみであり、E2Eで継続使用していない。SAPのWavLM layer21・K=2000とは出発点が異なる。[論文2、§II–III/V-B](https://arxiv.org/pdf/2609.16855)

SIRは、同じ内容を発話した健常者と障害者のtoken分布を比較し、健常者側をstop-gradientしたKLである。現在のcentroid間OTとは異なる。健常者のcentroidを参照するだけでSIRを再現したことにはならない。同内容のペアを利用できるか確認してから検討する。[論文2、§IV/式(8)](https://arxiv.org/pdf/2609.16855)

したがって、E2E設定でkmeans_weight=0、ot_weight=0であること自体を問題視する根拠は弱い。最初にk-means lossや一様利用率正則化を強くするより、学習順序と比較条件を整える。必要なら、偏りへの正則化を追加ablationとして切り分ける。

## 推奨する実験順序

1. **評価条件を揃える。** 既存モデルをdevで実decodeし、単体・平均のcheckpoint選択方針を統一する。
2. **固定条件を揃える。** E2E側にも独立Embeddingと通常版相当のTime Maskを用意し、決定的fixedで通常版に近い精度が出るかを確認する。ここが悪ければ全体最適化以前の差が残る。
3. **同じ初期点から分岐する。** 固定tokenizerで学習したASRを起点に、fixed継続とfullを比較する。WavLMのLRをASRより小さくし、CNN固定は維持する。
4. **量子化条件を一つずつ調整する。** 距離係数、温度の減衰単位・開始時期を個別比較する。決定的STは別条件とする。全dev hard頻度、PNMI等、dev WERを同時に記録する。
5. **残る汎化差を検討する。** 小K、pseudo-label refinement、ペア音声を使うSIRなどは、その後の実験とする。

最初から複数の条件をまとめて変えると改善要因が分からなくなるため、上記の各段階を保存する。

## 検証と限界

- 既存WER結果、実際の保存config、ログ、上位10checkpoint平均へのsymlinkを確認。
- 初期モデルと18epochの16話者診断をCPUで実行。詳細は同じディレクトリの`asr3_optimization_diagnostics_20261026.json`に保存。
- `test/espnet2/asr/test_gumbel_kmeans_asr_model.py`の既存4テスト関数を、学習用Python環境で直接呼び出して全件成功。ASR-onlyの勾配伝播・unit抽出などを確認した。
- 学習用環境にはpytestがないため、pytestコマンドとしての実行は不可。GPUドライバにもアクセスできず、GPUでの再学習・WER再評価・AMPの数値検証は未実施。
- 16話者診断はFP32であり、実験のBF16 frontendの数値条件とは異なる。偏りの発見は有力な手掛かりだが、WER悪化の原因の確定や、提案修正の改善保証にはならない。
