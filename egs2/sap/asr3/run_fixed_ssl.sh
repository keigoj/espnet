#!/usr/bin/env bash

# Extract units with a frozen pretrained WavLM and fixed SAP K-means centroids,
# then train and evaluate the downstream discrete-token ASR model.
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=4
stop_stage=6
downstream_stage=7
downstream_stop_stage=15

ngpu=1
extract_ngpu=1
nj=4
inference_nj=4
gpu_inference=false
python=python3
dumpdir=dump
expdir=exp

nclusters=2000
# Use the resolved training config because it contains task-required fields
# such as token_list.  No model checkpoint is loaded during fixed extraction.
ssl_config=exp/ssl_wavlm_large_l21_gumbel_km2000_sap_init_div05/config.yaml
ssl_tag=
src_lang=
tgt_lang=en

train_set=train
valid_set=dev
test_sets="test1 test2"
speed_perturb_factors="0.9 1.0 1.1"

src_case=rm  # rm: deduplicate consecutive units, ts: keep every frame unit
tgt_case=ts
src_nbpe=6000
tgt_nbpe=5000

asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml
asr_tag=
asr_args=
inference_args=

help_message=$(cat <<'EOF'
Usage: ./run_fixed_ssl.sh [options]

Stages:
  1: Prepare audio data by reusing ASR3 stage 1
  4: Extract units without an SSL checkpoint
     (frozen pretrained WavLM + fixed initial_centroid_path)
  5: Create the discrete-token ASR data directories
  6: Run downstream ASR2 stages 7--15

Examples:
  ./run_fixed_ssl.sh --stop_stage 4
  ./run_fixed_ssl.sh --stage 5 --stop_stage 6
  ./run_fixed_ssl.sh --stage 6 --downstream_stage 13
EOF
)

run_args=$(scripts/utils/print_args.sh "$0" "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: no positional arguments are accepted"
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ -z "${ssl_tag}" ]; then
    ssl_tag=wavlm_large_l21_sap_km${nclusters}_fixed
fi
if [ -z "${src_lang}" ]; then
    src_lang=${ssl_tag}
fi
if [ -z "${asr_tag}" ]; then
    asr_tag=${src_lang}_${tgt_lang}
fi

if [ ! -f "${ssl_config}" ]; then
    log "Error: missing SSL config: ${ssl_config}"
    exit 2
fi
if [ ! -f "${asr_config}" ]; then
    log "Error: missing ASR config: ${asr_config}"
    exit 2
fi
if [ "${src_case}" != rm ] && [ "${src_case}" != ts ]; then
    log "Error: --src_case must be rm or ts, not ${src_case}"
    exit 2
fi
if [ "${extract_ngpu}" -ne 0 ] && [ "${extract_ngpu}" -ne 1 ]; then
    log "Error: --extract_ngpu must be 0 or 1"
    exit 2
fi

config_values=$("${python}" -c '
import sys
from pathlib import Path

import numpy as np
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    config = yaml.safe_load(f)
quantizers = [x for x in config["loss"] if x["name"] == "gumbel_kmeans"]
if len(quantizers) != 1:
    raise ValueError("The recipe requires exactly one gumbel_kmeans loss")
conf = quantizers[0]["conf"]
path = conf.get("initial_centroid_path")
if not path:
    raise ValueError("initial_centroid_path is required for fixed extraction")
centroid_path = Path(path)
if not centroid_path.is_file():
    raise FileNotFoundError(f"Initial centroid file not found: {path}")
centroids = np.load(centroid_path)
expected_shape = (conf["num_clusters"], 1024)
if centroids.shape != expected_shape:
    raise ValueError(
        "Unexpected initial centroid shape: "
        f"expected {expected_shape}, got {centroids.shape}"
    )
print(conf["num_clusters"], path)
' "${ssl_config}")
config_nclusters=${config_values%% *}
initial_centroid_path=${config_values#* }
if [ "${config_nclusters}" -ne "${nclusters}" ]; then
    log "Error: --nclusters=${nclusters}, but ${ssl_config} uses ${config_nclusters}"
    exit 2
fi

data_audio=${dumpdir}/audio_raw
unit_dir=${expdir}/units_${ssl_tag}

common_asr3_args=(
    --ngpu "${ngpu}"
    --extract_ngpu "${extract_ngpu}"
    --nj "${nj}"
    --inference_nj "${inference_nj}"
    --gpu_inference "${gpu_inference}"
    --dumpdir "${dumpdir}"
    --expdir "${expdir}"
    --nclusters "${nclusters}"
    --ssl_config "${ssl_config}"
    --ssl_tag "${ssl_tag}"
    --unit_dir "${unit_dir}"
    --src_lang "${src_lang}"
    --tgt_lang "${tgt_lang}"
    --src_case "${src_case}"
    --tgt_case "${tgt_case}"
    --src_token_type bpe
    --src_nbpe "${src_nbpe}"
    --tgt_token_type bpe
    --tgt_nbpe "${tgt_nbpe}"
    --speed_perturb_factors "${speed_perturb_factors}"
    --asr_config "${asr_config}"
    --asr_args "${asr_args}"
    --inference_config "${inference_config}"
    --inference_args "${inference_args}"
    --asr_tag "${asr_tag}"
    --train_set "${train_set}"
    --valid_set "${valid_set}"
    --test_sets "${test_sets}"
)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Prepare audio data"
    ./asr3.sh "${common_asr3_args[@]}" --stage 1 --stop_stage 1
fi

train_sp_sets=
for factor in ${speed_perturb_factors}; do
    if "${python}" -c "assert ${factor} != 1.0" 2>/dev/null; then
        train_sp_sets+="${train_set}_sp${factor} "
    fi
done

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Extract fixed pretrained-WavLM + SAP-centroid units"
    log "Initial centroids: ${initial_centroid_path}"
    mkdir -p "${unit_dir}"
    printf '%s --stage 4 "$@"\n' "${run_args}" > "${unit_dir}/run.sh"
    chmod +x "${unit_dir}/run.sh"

    remove_repeated=false
    if [ "${src_case}" = rm ]; then
        remove_repeated=true
    fi
    extract_sets="${train_set} ${train_sp_sets}${valid_set} ${test_sets}"
    for dset in ${extract_sets}; do
        wav_scp=${data_audio}/${dset}/wav.scp
        if [ ! -f "${wav_scp}" ]; then
            log "Error: missing ${wav_scp}; run with --stage 1 first"
            exit 1
        fi
        log "Extract fixed units: ${dset}"
        "${python}" -m espnet2.bin.gumbel_kmeans_extract \
            --task ssl \
            --data_path_and_name_and_type "${wav_scp},speech,sound" \
            --train_config "${ssl_config}" \
            --model_file none \
            --output_dir "${unit_dir}/${dset}" \
            --remove_repeated "${remove_repeated}" \
            --ngpu "${extract_ngpu}" \
            --batch_size 1 \
            --num_workers 1
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 5 ]; then
    downstream_start=${stage}
    if [ "${downstream_start}" -lt 5 ]; then
        downstream_start=5
    fi
    downstream_stop=${stop_stage}
    if [ "${downstream_stop}" -gt 6 ]; then
        downstream_stop=6
    fi
    log "Stages ${downstream_start}-${downstream_stop}: Prepare and run downstream ASR"
    ./asr3.sh "${common_asr3_args[@]}" \
        --stage "${downstream_start}" \
        --stop_stage "${downstream_stop}" \
        --downstream_stage "${downstream_stage}" \
        --downstream_stop_stage "${downstream_stop_stage}"
fi

log "Finished. Fixed units: ${unit_dir}"
