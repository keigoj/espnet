#!/usr/bin/env bash

# End-to-end SAP ASR with a differentiable Gumbel K-means bottleneck:
# waveform -> WavLM -> Gumbel K-means -> E-Branchformer -> transcript.
set -e
set -u
set -o pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${recipe_dir}"

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# The SAP data directories already exist, and local/data.sh is intentionally
# empty.  Start from speed perturbation rather than ASR1 stage 1.
stage=2
stop_stage=13
skip_data_prep=false
skip_train=false
skip_eval=false

e2e_mode=full
warmup_epochs=30
max_epoch=90
ngpu=1
num_nodes=1
nj=4
inference_nj=4
gpu_inference=false

train_set=train
valid_set=dev
test_sets="test1 test2"
speed_perturb_factors="0.9 1.0 1.1"

dumpdir=dump/e2e
expdir=exp
audio_format=flac
fs=16k
min_wav_duration=0.1
max_wav_duration=30

token_type=bpe
nbpe=5000
bpemode=unigram
lang=en
nclusters=2000
initial_centroid_path=

asr_config=conf/train_e2e_gumbel_kmeans_wavlm_large.yaml
inference_config=conf/decode_e2e_ctc0.3.yaml
asr_args=
asr_stats_dir=
asr_exp=
pretrained_model=
ignore_init_mismatch=false
inference_asr_model=valid.acc.ave.pth

help_message=$(cat <<'EOF'
Usage: ./run_e2e.sh [options]

Modes (the ASR encoder/decoder is trained in every mode):
  --e2e_mode fixed      Keep WavLM and the K-means codebook frozen.
  --e2e_mode quantizer  Unfreeze the codebook after the warm-up.
  --e2e_mode full       Unfreeze trainable WavLM layers and the codebook after warm-up.

Schedule options:
  --warmup_epochs N     Freeze WavLM/codebook for the first N epochs (default: 30).
  --max_epoch N         Total training epochs (default: 90).

Cluster and output options:
  --nclusters N                  Number of K-means clusters.
  --initial_centroid_path PATH   Override the derived SAP centroid path.
  --asr_exp DIR                  Override the derived experiment directory.

Examples:
  # Fixed-tokenizer baseline for all 90 epochs
  ./run_e2e.sh --e2e_mode fixed

  # Onda-style schedule: freeze WavLM/codebook for epochs 1--30, then
  # jointly update trainable WavLM layers, codebook, and ASR for epochs 31--90.
  ./run_e2e.sh --e2e_mode full --warmup_epochs 30 --max_epoch 90

  # Update WavLM Transformer and the codebook from epoch 1. The CNN feature
  # extractor remains permanently frozen according to the YAML config.
  ./run_e2e.sh --e2e_mode full --warmup_epochs 0

  # Select the cluster count. The default centroid path and experiment name
  # are derived from this value.
  ./run_e2e.sh --e2e_mode fixed --nclusters 1000

Stages are the standard ASR1 stages: 2--5 data/token preparation, 10 stats,
11 training, 12 decoding, and 13 scoring.  Stages 6--9 are skipped because
this recipe does not train an LM or n-gram model.
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: no positional arguments are accepted"
    exit 2
fi

# Use the same managed Python environment as the ESPnet training stages for
# config validation below.
. ./path.sh

if ! [[ "${nclusters}" =~ ^[1-9][0-9]*$ ]]; then
    log "Error: --nclusters must be a positive integer: ${nclusters}"
    exit 2
fi
if ! [[ "${warmup_epochs}" =~ ^[0-9]+$ ]]; then
    log "Error: --warmup_epochs must be a non-negative integer: ${warmup_epochs}"
    exit 2
fi
if ! [[ "${max_epoch}" =~ ^[1-9][0-9]*$ ]]; then
    log "Error: --max_epoch must be a positive integer: ${max_epoch}"
    exit 2
fi
if [ "${e2e_mode}" != fixed ] && [ "${warmup_epochs}" -ge "${max_epoch}" ]; then
    log "Error: --warmup_epochs must be smaller than --max_epoch"
    exit 2
fi

if [ -z "${initial_centroid_path}" ]; then
    initial_centroid_path=../asr2/exp/kmeans/wavlm_large_21_base_${nclusters}clusters/km_${nclusters}.npy
fi

case "${e2e_mode}" in
    fixed)
        # Freezing parameters alone would still leave Gumbel noise active while
        # the parent ASR model is in train mode.  Force deterministic nearest
        # centroids for a true fixed-tokenizer baseline.
        freeze_args="--freeze_param frontend quantizer --model_conf quantizer_sample_gumbel=false"
        staged_freeze_args=
        ;;
    quantizer)
        freeze_args="--freeze_param frontend"
        staged_freeze_args="--staged_freeze_param quantizer"
        ;;
    full)
        freeze_args=
        staged_freeze_args="--staged_freeze_param frontend --staged_freeze_param quantizer"
        ;;
    *)
        log "${help_message}"
        log "Error: --e2e_mode must be fixed, quantizer, or full: ${e2e_mode}"
        exit 2
        ;;
esac

[ ! -f "${asr_config}" ] && { log "Error: missing ${asr_config}"; exit 1; }
[ ! -f "${inference_config}" ] && {
    log "Error: missing ${inference_config}"
    exit 1
}
[ ! -f "${initial_centroid_path}" ] && {
    log "Error: missing initial SAP centroids: ${initial_centroid_path}"
    exit 1
}

python3 - "${asr_config}" "${initial_centroid_path}" "${nclusters}" <<'PY'
import sys
import numpy as np
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    config = yaml.safe_load(f)
if config.get("model") != "gumbel_kmeans":
    raise ValueError("E2E config must use model: gumbel_kmeans")
model_conf = config.get("model_conf", {})
quantizer_conf = model_conf.get("quantizer_conf", {})
if not isinstance(quantizer_conf, dict):
    raise ValueError("model_conf.quantizer_conf must be a mapping")

path = sys.argv[2]
nclusters = int(sys.argv[3])
loaded = np.load(path, mmap_mode="r")
if isinstance(loaded, np.lib.npyio.NpzFile):
    try:
        if "centers" in loaded.files:
            centroids = loaded["centers"]
        elif len(loaded.files) == 1:
            centroids = loaded[loaded.files[0]]
        else:
            raise ValueError(
                f"Initial centroid .npz must contain 'centers' or one array: {path}"
            )
    finally:
        loaded.close()
else:
    centroids = loaded
if centroids.ndim != 2:
    raise ValueError(
        f"Initial centroids must be two-dimensional, got {centroids.shape}: {path}"
    )
if centroids.shape[0] != nclusters:
    raise ValueError(
        f"--nclusters={nclusters}, but {path} contains "
        f"{centroids.shape[0]} centroids"
    )
PY

if [ -n "${pretrained_model}" ] && [ ! -f "${pretrained_model%%:*}" ]; then
    log "Error: missing pretrained model: ${pretrained_model%%:*}"
    exit 1
fi

quantizer_args="--model_conf quantizer_conf.num_clusters=${nclusters}"
quantizer_args+=" --model_conf quantizer_conf.initial_centroid_path=${initial_centroid_path}"
mode_asr_args=${quantizer_args}
if [ -n "${freeze_args}" ]; then
    mode_asr_args="${freeze_args} ${mode_asr_args}"
fi
if [ -n "${asr_args}" ]; then
    mode_asr_args="${mode_asr_args} ${asr_args}"
fi
mode_asr_args+=" --max_epoch ${max_epoch}"
if [ -n "${staged_freeze_args}" ] && [ "${warmup_epochs}" -gt 0 ]; then
    mode_asr_args+=" ${staged_freeze_args}"
    mode_asr_args+=" --staged_freeze_until_epoch ${warmup_epochs}"
fi

sp_tag=
if [ -n "${speed_perturb_factors}" ]; then
    sp_tag=_sp
fi
if [ -z "${asr_stats_dir}" ]; then
    asr_stats_dir=${expdir}/asr_stats_e2e_raw_${lang}_${token_type}${nbpe}${sp_tag}
fi
if [ -z "${asr_exp}" ]; then
    if [ "${e2e_mode}" = fixed ]; then
        schedule_tag=ep${max_epoch}
    else
        schedule_tag=warmup${warmup_epochs}_ep${max_epoch}
    fi
    asr_exp=${expdir}/asr_e2e_wavlm_large_l21_gumbel_km${nclusters}_${e2e_mode}_${schedule_tag}_${lang}
fi

opts=(
    --stage "${stage}"
    --stop_stage "${stop_stage}"
    --skip_data_prep "${skip_data_prep}"
    --skip_train "${skip_train}"
    --skip_eval "${skip_eval}"
    --skip_packing true
    --skip_upload_hf true
    --ngpu "${ngpu}"
    --num_nodes "${num_nodes}"
    --nj "${nj}"
    --inference_nj "${inference_nj}"
    --gpu_inference "${gpu_inference}"
    --dumpdir "${dumpdir}"
    --expdir "${expdir}"
    --audio_format "${audio_format}"
    --fs "${fs}"
    --min_wav_duration "${min_wav_duration}"
    --max_wav_duration "${max_wav_duration}"
    --feats_type raw
    --feats_normalize none
    --speed_perturb_factors "${speed_perturb_factors}"
    --token_type "${token_type}"
    --nbpe "${nbpe}"
    --bpemode "${bpemode}"
    --lang "${lang}"
    --use_lm false
    --use_ngram false
    --asr_config "${asr_config}"
    --asr_args "${mode_asr_args}"
    --asr_stats_dir "${asr_stats_dir}"
    --asr_exp "${asr_exp}"
    --inference_config "${inference_config}"
    --inference_asr_model "${inference_asr_model}"
    --ignore_init_mismatch "${ignore_init_mismatch}"
    --train_set "${train_set}"
    --valid_set "${valid_set}"
    --test_sets "${test_sets}"
    --bpe_train_text "data/${train_set}/text"
    # --lm_train_text /dev/null
)
if [ -n "${pretrained_model}" ]; then
    opts+=(--pretrained_model "${pretrained_model}")
fi

if [ "${e2e_mode}" = fixed ]; then
    log "E2E mode=fixed, WavLM/codebook frozen for all ${max_epoch} epochs"
else
    log "E2E mode=${e2e_mode}, schedule=${warmup_epochs}+$((${max_epoch} - ${warmup_epochs})) epochs"
fi
log "Clusters=${nclusters}, centroid=${initial_centroid_path}"
log "ASR experiment=${asr_exp}"
./asr_e2e.sh "${opts[@]}"
