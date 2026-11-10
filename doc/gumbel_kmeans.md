# Joint SSL and differentiable K-means

ESPnet supports a trainable K-means codebook whose assignments use
Gumbel-Softmax.  There are two supported training paths:

1. `SSL + K-means`: add `gumbel_kmeans` to an `SSLTask` loss list.
2. `SSL + K-means + ASR`: use an SSL frontend and the
   `gumbel_kmeans` model in an ASR1 recipe.

The first path is suitable for training a tokenizer and then dumping fixed
units for ASR2.  The second path allows the ASR loss to update the codebook and
the SSL frontend end to end.

## Gumbel K-means

For SSL features `h_t` and trainable centers `c_k`, assignment logits are the
negative squared Euclidean distances:

```text
logit(t, k) = -||h_t - c_k||^2
```

Training uses `torch.nn.functional.gumbel_softmax`.  With `hard: true`, its
straight-through estimator produces one-hot assignments in the forward pass
and soft gradients in the backward pass.  Evaluation and unit extraction use
deterministic nearest-center assignments.

The reported losses and diagnostics are:

- `distortion`: squared reconstruction error between each SSL feature and its
  Gumbel-Softmax-weighted centroid, normalized by the feature dimension;
- `commitment`: feature-to-quantized mean squared error;
- `diversity`: KL divergence between average cluster usage and a uniform
  distribution;
- `variance`: penalty when an encoder feature dimension has less than the
  configured batch-wise standard deviation;
- `ot`: uniform-mass Sinkhorn divergence from a fixed reference codebook;
- `perplexity`: effective number of clusters used by the batch.

The diversity term reduces dead-cluster behavior, but it does not by itself
prevent every feature and center from collapsing to the same value.  Combine
K-means with an SSL or ASR objective and monitor codebook perplexity.

## SSL + K-means

Add the loss to an SSL configuration.  Keeping the HuBERT loss makes the SSL
representation and centers learn simultaneously:

```yaml
loss:
  - name: hubert
    conf:
      num_classes: 500
      final_dim: 256
  - name: gumbel_kmeans
    conf:
      num_clusters: 500
      layer: -1
      kmeans_weight: 1.0
      commitment_weight: 0.1
      diversity_weight: 0.1
      variance_weight: 1.0
      variance_target: 1.0
      temperature: 2.0
      min_temperature: 0.5
      temperature_decay: 0.999995
      hard: true
      normalize_inputs: false
```

Existing HuBERT labels are still needed by the HuBERT loss.  To optimize only
the clustering objective, omit the HuBERT entry and enable the feature
variance term.  K-means distortion alone has a degenerate collapsed solution.

An end-to-end SAP example is available in `egs2/sap/asr3`.  It fine-tunes
WavLM with Gumbel K-means, extracts hard units, and passes them to the existing
discrete ASR2 training stages.

## Fixed-reference optimal transport

A Gumbel K-means codebook can be initialized and regularized by centroids from
a fixed reference corpus.  The SAP recipe uses healthy-speech WavLM layer-21
centroids with shape `(2000, 1024)`:

```yaml
preencoder: null
encoder: identity

loss:
  - name: gumbel_kmeans
    conf:
      num_clusters: 2000
      ot_weight: 0.01
      reference_centroid_path: ../../librispeech_100/asr2/exp/kmeans/wavlm_large_21_base_2000clusters/km_2000.npy
      initialize_from_reference: true
      ot_epsilon: 1.0
      ot_iterations: 20
      ot_mass_mode: uniform
```

Both codebooks are treated as uniformly weighted empirical distributions.
POT's PyTorch backend and log-domain solver use squared Euclidean centroid
distance divided by the feature dimension.  The debiased Sinkhorn divergence
is zero when the two codebooks are identical and is invariant to cluster-index
permutations.  Envelope gradients update the current codebook without keeping
all Sinkhorn iterations in the autograd graph.

`ot_weight` is the coefficient applied to the OT loss, whereas
`ot_mass_mode` controls the probability mass assigned to each centroid.  The
only training mode currently implemented is `uniform`.  As an extension hook,
the POT wrapper and `GumbelKMeans.optimal_transport_loss` already accept
strictly positive `source_mass` and `reference_mass` tensors with shape `(K,)`;
they are normalized to sum to one.  A future `cluster_usage` mode can therefore
connect smoothed SAP cluster occupancy and healthy-corpus cluster frequencies
without changing the OT solver.  Unsupported mode names fail explicitly.

The reference centroids are a persistent PyTorch buffer: they move with the
model and are included in checkpoints, but they receive no gradient.  The
current codebook remains an `nn.Parameter`.  The reference and current
codebooks must have exactly the same shape.  Using `encoder: identity` ensures
that SAP and healthy centroids stay in the same 1024-dimensional WavLM space.

POT 0.9.7.x is required when `ot_weight` is positive.  It is included in
ESPnet's `recipe` extra and can also be installed directly:

```bash
python -m pip install "POT>=0.9.7,<0.10"
python -c "import ot; print(ot.__version__)"
```

The `K x K` transport cost increases memory and runtime, so `ot_iterations`
and batch size should be monitored for large codebooks.  POT warns when the
configured maximum iteration count is reached before convergence.

## SSL + K-means + ASR

Use an ASR1 recipe because the model input is waveform rather than a dumped
unit sequence.  The SSL frontend must not be listed in `freeze_param` if it is
to be updated.

```yaml
frontend: s3prl
frontend_conf:
  frontend_conf:
    upstream: wavlm_large
  download_dir: ./hub
  multilayer_feature: false
  layer: 21

model: gumbel_kmeans
model_conf:
  ctc_weight: 0.3
  lsm_weight: 0.1
  asr_weight: 1.0
  kmeans_weight: 1.0
  commitment_weight: 0.1
  diversity_weight: 0.1
  ot_weight: 0.01
  quantizer_conf:
    num_clusters: 500
    reference_centroid_path: /path/to/healthy_centroids.npy
    initialize_from_reference: true
    ot_epsilon: 1.0
    ot_iterations: 20
    ot_mass_mode: uniform
    temperature: 2.0
    min_temperature: 0.5
    temperature_decay: 0.999995
    hard: true
    normalize_inputs: false
```

`quantizer_conf.input_dim` is inferred from the frontend.  Specify it explicitly
when `frontend: null` is used.  The quantized vector has the same dimension as
the SSL feature, so existing ASR preencoders and encoders can follow it.

For an ESPnet-trained SSL checkpoint, `frontend: espnet_ssl` can be used.  Set
`mask_feats: false` unless masking SSL features during supervised ASR training
is intentional.

### Severity-invariant regularization (SIR)

The joint ASR model supports SIR from Section IV, Eqs. (7)--(8), of
[Wang et al., *Differentiable and Severity-invariant Discrete Tokens for
Dysarthric Speech Recognition*](https://arxiv.org/pdf/2609.16855).
For each utterance selected for SIR, supply a reference utterance with **the same
linguistic content**. Severity labels and frame alignment are unnecessary;
the two utterances may have different durations. The paper uses healthy
references. The same API also accepts mild dysarthric references as an
experimental extension for SAP, distinct from the paper's healthy-reference
setting. Reference selection is the caller's responsibility; the model does
not determine which speakers are mild.

SIR averages the relaxed Gumbel-Softmax assignments from Eq. (2) over valid
frames separately for each utterance, then computes

```text
L_SIR = mean_{b: sir_pair_mask[b] = 1} KL(stop_gradient(p_reference[b]) || p_input[b])
loss = asr_weight * L_ASR + L_cluster + sir_weight * L_SIR
```

ASR and configured clustering losses use the entire batch. SIR is averaged
over the selected pairs, not over all batch utterances. An all-zero mask
sets SIR to zero and skips reference encoding. In that case, the primary
speech still trains the SSL frontend, codebook, and ASR through their usual
objectives. Omitting `sir_pair_mask` selects all utterances, preserving the
fully paired healthy-reference API for UASpeech and TORGO.

The relaxed assignments use the same Gumbel samples as the main quantization
path, before straight-through hardening. Padding is excluded. For numerical
stability, averaged distributions are floored at `quantizer_conf.eps` and
renormalized before KL computation. In deterministic mode, the distributions
use distance softmax without Gumbel noise.

The reference target is computed by the same SSL frontend and codebook, under
`no_grad`, without the downstream ASR encoder or SpecAugment. Both branches
use the same temperature; the reference forward does not advance annealing.
SIR updates the trainable SSL frontend and codebook through the input
branch. The ASR objective continues to train the full pipeline.

Add `sir_weight` to the existing `model_conf` to enable SIR:

```yaml
model: gumbel_kmeans
model_conf:
  sir_weight: 0.1  # Example only; tune for your data, not a paper default.
  # Keep the remaining model and quantizer configuration unchanged.
```

The default is `sir_weight: 0.0`; existing configurations and checkpoint
parameters are unchanged. The existing ASR, clustering, and OT weights remain
independent. The paper's Eq. (7) uses ASR + SIR during E2E training, whereas
this implementation also retains whichever clustering terms are configured.
The paper uses HuBERT and temperature 0.8 (Section V-B); enabling SIR does
not change this repository's frontend, temperature schedule, or other
experimental settings.

Create a `healthy.scp` with the **primary utterance IDs** as keys and
content-parallel reference waveform paths as values. The name `healthy_speech`
is retained for compatibility and also accepts mild dysarthric references.
Include every training utterance, using the actual IDs after speed
perturbation/filtering:

```text
dysarthric_utt_0001 /path/to/healthy_same_content_0001.wav
dysarthric_utt_0002 /path/to/healthy_same_content_0002.wav
dysarthric_utt_0003 /path/to/any_valid_placeholder.wav
```

For partially paired data, also create a `sir_pair_mask` file containing
exactly one integer per utterance:

```text
dysarthric_utt_0001 1
dysarthric_utt_0002 1
dysarthric_utt_0003 0
```

ESPnet requires all streams to contain the primary utterance IDs. An unpaired
row therefore still needs a valid path in `healthy.scp` so it can be loaded,
but the placeholder's samples never enter the SSL frontend or SIR computation.
For SAP, select mild references from training speakers with matching actual
transcripts; a shared question/prompt ID alone does not guarantee matching
spoken content. Leave unpaired and reference-only utterances in ASR training
with mask 0. The reference branch's stop-gradient does not prevent a mild
utterance from learning through ASR when it appears as a primary input.
Use training data only when selecting training references; the model does not
perform metadata-based severity selection or automatic pairing.

Use mono audio with the same sampling rate as the primary speech. The SCP
selects the reference for each input; pairing and control-speaker sampling
are the caller's responsibility. The paper randomly selects a control speaker
for matching content. A fixed SCP keeps that selection fixed unless it is
regenerated; it does not implement per-epoch random reference sampling.
Unrelated healthy speech or reference centroids alone cannot supply Eq. (8).

Pass the extra stream to the existing `espnet2.bin.asr_train` command:

```bash
--train_data_path_and_name_and_type /path/to/train/healthy.scp,healthy_speech,sound
--train_data_path_and_name_and_type /path/to/train/sir_pair_mask,sir_pair_mask,text_int
```

Omit the mask stream when all utterances have reference pairs.

For `egs2/sap/asr3/run_e2e.sh`, it can be appended via `--asr_args`:

```bash
sir_args="--model_conf sir_weight=0.1"
sir_args+=" --train_data_path_and_name_and_type /path/to/train/healthy.scp,healthy_speech,sound"
sir_args+=" --train_data_path_and_name_and_type /path/to/train/sir_pair_mask,sir_pair_mask,text_int"
./run_e2e.sh --e2e_mode full --asr_args "${sir_args}"
```

ESPnet's collator creates `healthy_speech_lengths` automatically. The Python
model API accepts `healthy_speech` and `healthy_speech_lengths`, paired by
batch position, and a binary `sir_pair_mask` of shape `(batch,)` or `(batch, 1)`.
The mask is non-sequential metadata; its collator does not create a lengths
tensor. Selected references must have positive lengths; unselected lengths
may be zero. An all-zero mask permits omitting both reference tensors in the
Python API. Training with positive `sir_weight` fails explicitly if selected
pairs are missing or batch dimensions/selected lengths are invalid. Validation
may omit pairs, in which case `loss_sir` is zero and `loss` excludes SIR. To report
SIR on validation data, also supply a corresponding
`--valid_data_path_and_name_and_type ...,healthy_speech,sound` stream and,
for partial pairing, a `...,sir_pair_mask,text_int` stream.
Inference and `encode_units` need only the primary speech. The unweighted SIR
loss is logged as `loss_sir`; `loss` includes its configured coefficient.
During DDP training, SIR is normalized using the total pair count across workers,
accounting for ESPnet's primary-batch loss weighting, including workers with no
local pairs. `sir_num_pairs` reports the total contributing pair count.
Evaluation uses the local pair mean and local pair count without model-side
collectives, since attention plotting runs on rank zero alone. Distributed
validation aggregates these statistics using ESPnet's primary-batch weights.
ESPnet's
epoch statistics average these per-batch counts rather than sum them.

For SAP doc-based mild-speaker selection, saved pairing artifacts, and a
training entry point, see [the SAP SIR recipe](../egs2/sap/asr3/README_sir_mild.md).

## Fixed-unit extraction for ASR2

Hard units can be dumped from either training path:

```bash
python -m espnet2.bin.gumbel_kmeans_extract \
    --task ssl \
    --train_config exp/ssl/config.yaml \
    --model_file exp/ssl/valid.loss.best.pth \
    --data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound \
    --output_dir exp/units/train \
    --remove_repeated true
```

For a joint ASR checkpoint, change `--task ssl` to `--task asr`.  The result is
written to `OUTPUT_DIR/pseudo_labels.txt` in keyed integer-sequence format.  A
single fixed checkpoint must be used to extract train, validation, and test
units so that cluster IDs remain consistent.  These units can then be converted
to the CJK representation expected by the existing ASR2 recipe.

## Offline center initialization

The Python API accepts centers exported from an existing K-means model:

```python
centers = torch.from_numpy(kmeans.cluster_centers_)
# Joint ASR model
model.quantizer.set_codebook(centers)
# SSL model (select the configured Gumbel K-means loss)
model.losses[gumbel_loss_index].quantizer.set_codebook(centers)
```

Center shape must be `(num_clusters, SSL_feature_dimension)`.  Once saved, the
centers are ordinary checkpoint parameters and are synchronized by DDP.
