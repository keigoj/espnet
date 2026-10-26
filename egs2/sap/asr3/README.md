# SAP ASR3: differentiable discrete-token recipe

For the complete Japanese specification, implementation map, artifacts, and
verification checklist, see [IMPLEMENTATION_JA.md](IMPLEMENTATION_JA.md).

This recipe provides two complementary experiment paths.

The original offline-token path runs the following two-stage experiment:

1. Fine-tune WavLM layer 21 together with a Gumbel-Softmax K-means codebook,
   initialized from SAP K-means centroids, with optional healthy-reference OT.
2. Extract deterministic hard units and train the existing SAP discrete ASR model.

The top-level command is:

```bash
cd egs2/sap/asr3
./run.sh
```

The end-to-end path keeps tokenization inside the ASR computation graph:

```text
waveform -> WavLM -> Gumbel K-means -> E-Branchformer -> transcription
```

Run its fixed-tokenizer baseline with:

```bash
./run_e2e.sh --e2e_mode fixed
```

The cluster count controls both the default SAP K-means centroid path and the
default experiment-directory name:

```bash
./run_e2e.sh --e2e_mode fixed --nclusters 1000
# centroid: ../asr2/exp/kmeans/wavlm_large_21_base_1000clusters/km_1000.npy
# experiment: exp/asr_e2e_wavlm_large_l21_gumbel_km1000_fixed_ep90_en
```

Use `--initial_centroid_path` for a centroid file that does not follow this
layout, or `--asr_exp` to override the complete experiment path. The launcher
checks that the centroid file contains exactly `nclusters` rows before starting.

The same model and data path support three controlled comparisons:

| `e2e_mode` | Epochs 1--30 | Epochs 31--90 |
|---|---|---|
| `fixed` | ASR only | ASR only |
| `quantizer` | ASR only | codebook + ASR |
| `full` | ASR only | trainable WavLM layers + codebook + ASR |

`fixed` also disables Gumbel sampling during ASR training, so it uses the same
deterministic nearest-centroid assignment as offline K-means.  The other two
modes use hard straight-through Gumbel assignments during training.

The default schedule follows the single-layer experiment in Onda et al.: WavLM
and the codebook are frozen through epoch 30 and the selected modules are
unfrozen at the beginning of epoch 31 in the same training process:

```bash
./run_e2e.sh --e2e_mode full --warmup_epochs 30 --max_epoch 90
```

The optimizer and scheduler are not recreated at the transition. Set
`--warmup_epochs 0` to update the selected modules from epoch 1. The default
experiment name includes both the warm-up and total epoch counts, for example
`asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup30_ep90_en`, so schedules do
not accidentally resume from each other's checkpoints.

Temperature annealing pauses while the codebook is frozen. With the default
30-epoch warm-up, the temperature stays at 2.0 through epoch 30; epoch 31 starts
at 2.0 and subsequent training forwards advance the existing decay schedule.
The temperature counter is preserved in checkpoints. Gumbel sampling remains
enabled during the warm-up in `full` and `quantizer` modes. The learning-rate
scheduler continues throughout both phases without a reset.

`run_e2e_4090_amp.sh` also defaults to this 30-epoch freeze. For a new RTX 4090
experiment using the current distance coefficient and Time Mask settings:

```bash
./run_e2e_4090_amp.sh --stage 11 --stop_stage 11 \
  --asr_exp exp/asr_e2e_wavlm_large_l21_km2000_full_warmup30_scale1_timemask_en
```

The current config permanently freezes WavLM's convolutional feature extractor
with `freeze_param`; `full` unfreezes the remaining WavLM parameters at epoch
31. For the dysarthric-ASR condition without an SSL/codebook warm-up freeze,
keep the convolutional feature extractor fixed and update the WavLM Transformer
and codebook from the first epoch with:

```bash
./run_e2e.sh \
  --e2e_mode full \
  --warmup_epochs 0
```

This condition gets a distinct `full_warmup0_ep90` experiment-name tag,
so it cannot accidentally resume from the staged-freeze experiment.

All three modes start from the same offline SAP K-means centroids.  The default
E2E objective is ASR-only (`0.3 * CTC + 0.7 * attention`); K-means,
commitment, diversity, and OT loss weights are zero.  Training uses hard
straight-through Gumbel assignments, while evaluation is deterministic.

The E2E configs set `quantizer_conf.logit_scale: 1.0`, matching the paper's
squared-distance coefficient. With `hard: true`, training uses one-hot
assignments in the forward pass and soft assignment gradients in the backward
pass. This also holds when `quantizer_sample_gumbel: false` selects deterministic
nearest-centroid assignments; disabling noise does not disconnect SSL gradients.

`run_e2e.sh` uses the standard ASR1 stage numbers.  It starts at stage 2
because SAP's `data/` directories already exist and `local/data.sh` is empty.
Stages 6--9 are skipped because no LM or n-gram model is trained.

The stages are:

| Stage | Operation |
|---:|---|
| 1 | SAP data preparation, speed perturbation, waveform formatting |
| 2 | SSL collect-stats |
| 3 | WavLM + Gumbel K-means joint training |
| 4 | Hard-unit extraction |
| 5 | CJK conversion and downstream ASR data preparation |
| 6 | ASR2 token-list creation, ASR training, decoding, and scoring |

For example, train only the tokenizer first, inspect its validation metrics, and
then continue with extraction and ASR:

```bash
./run.sh --stop_stage 3
./run.sh --stage 4
```

The healthy reference defaults to the LibriSpeech WavLM layer-21 K-means file
configured by `loss[0].conf.reference_centroid_path`. It must have shape
`(2000, 1024)`. WavLM features are passed directly to K-means in this same
1024-dimensional space; no linear or Transformer projection is used. The
Sinkhorn divergence uses POT's PyTorch backend. Install the recipe dependencies
or run `python -m pip install "POT>=0.9.7,<0.10"` before enabling OT. Stage 3
checks the installed POT version before starting training.

`ot_mass_mode: uniform` is an explicit placeholder for future non-uniform
centroid masses. The low-level OT API already accepts separate SAP and healthy
cluster-mass tensors, but the current recipe supplies neither and therefore
remains uniformly weighted.

The current offline-token config uses K-means distortion, diversity, and
variance losses; OT and commitment are disabled.  Its most important
diagnostics in `exp/ssl_*/train.log` include `gumbel_kmeans_distortion`,
`gumbel_kmeans_diversity`, `gumbel_kmeans_variance`, and
`gumbel_kmeans_perplexity`. Perplexity staying close to 1 means that almost all
frames use one cluster.

`run.sh` defaults to 2000 clusters. If this is changed, update both `nclusters`
in `run.sh` and `loss[0].conf.num_clusters` in the SSL config. S3PRL must be
installed because WavLM is loaded through the ESPnet S3PRL frontend.
