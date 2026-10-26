#!/usr/bin/env bash

# Train a differentiable SSL tokenizer, extract hard units, and train ASR on them.
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a=${b}
        fi
    done
    echo "${a}"
}

stage=1
stop_stage=100
downstream_stage=7
downstream_stop_stage=15
skip_data_prep=false
skip_ssl_train=false
skip_unit_extraction=false
skip_asr_train=false
skip_eval=false

ngpu=1
num_nodes=1
extract_ngpu=1
nj=4
inference_nj=4
gpu_inference=false
python=python3
dumpdir=dump
expdir=exp

local_data_opts=
speed_perturb_factors=
audio_format=flac
fs=16k
min_wav_duration=0.1
max_wav_duration=30

ssl_config=
ssl_args=
ssl_tag=
ssl_stats_dir=
ssl_exp=
ssl_speech_fold_length=800
inference_ssl_model=valid.loss.ave.pth
unit_dir=
nclusters=2000

src_lang=gumbel_km2000
tgt_lang=en
src_case=rm
tgt_case=ts
src_token_type=bpe
src_nbpe=6000
tgt_token_type=bpe
tgt_nbpe=5000

asr_config=
asr_args=
asr_tag=
inference_config=
inference_args=

train_set=
valid_set=
test_sets=

help_message=$(cat <<'EOF'
Usage: ./asr3.sh --train_set train --valid_set dev --test_sets "test1 test2"

Stages:
  1: SAP data preparation, speed perturbation, and waveform formatting
  2: SSL + Gumbel K-means collect-stats
  3: SSL + Gumbel K-means training
  4: Deterministic hard-unit extraction
  5: Create the discrete-token ASR data directories
  6: Run ASR2 stages 7--15 (tokenization, ASR training, decoding, scoring)

Useful resume options:
  --stage 4 --stop_stage 5
  --stage 6 --downstream_stage 13 --downstream_stop_stage 15
EOF
)

log "$0 $*"
run_args=$(scripts/utils/print_args.sh "$0" "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: no positional arguments are accepted"
    exit 2
fi

. ./path.sh
. ./cmd.sh

[ -z "${train_set}" ] && { log "Error: --train_set is required"; exit 2; }
[ -z "${valid_set}" ] && { log "Error: --valid_set is required"; exit 2; }
[ -z "${test_sets}" ] && { log "Error: --test_sets is required"; exit 2; }
[ -z "${ssl_config}" ] && { log "Error: --ssl_config is required"; exit 2; }
[ -z "${asr_config}" ] && { log "Error: --asr_config is required"; exit 2; }
[ ! -f "${ssl_config}" ] && { log "Error: missing ${ssl_config}"; exit 2; }
[ ! -f "${asr_config}" ] && { log "Error: missing ${asr_config}"; exit 2; }

if [ "${src_case}" != rm ] && [ "${src_case}" != ts ]; then
    log "Error: --src_case must be rm or ts, not ${src_case}"
    exit 2
fi
if [ "${extract_ngpu}" -ne 0 ] && [ "${extract_ngpu}" -ne 1 ]; then
    log "Error: --extract_ngpu must be 0 or 1"
    exit 2
fi
if [ "${nclusters}" -gt 20900 ]; then
    log "Error: CJK unit conversion supports at most 20900 clusters"
    exit 2
fi
config_values=$("${python}" -c '
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    config = yaml.safe_load(f)
quantizers = [x for x in config["loss"] if x["name"] == "gumbel_kmeans"]
if len(quantizers) != 1:
    raise ValueError("The recipe requires exactly one gumbel_kmeans loss")
conf = quantizers[0]["conf"]
print(conf["num_clusters"], conf.get("ot_weight", 0.0))
' "${ssl_config}")
config_nclusters=${config_values%% *}
config_ot_weight=${config_values#* }
if [ "${config_nclusters}" -ne "${nclusters}" ]; then
    log "Error: --nclusters=${nclusters}, but ${ssl_config} uses ${config_nclusters}"
    exit 2
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! "${skip_ssl_train}" \
        && "${python}" -c 'import sys; assert float(sys.argv[1]) > 0' \
            "${config_ot_weight}" 2>/dev/null; then
    if ! pot_version=$("${python}" -c '
from packaging.version import Version

try:
    import ot
except Exception as error:
    raise SystemExit(f"POT import failed: {error}")

version = Version(ot.__version__)
if not Version("0.9.7") <= version < Version("0.10"):
    raise SystemExit(f"POT>=0.9.7,<0.10 is required, found {ot.__version__}")
print(ot.__version__)
' 2>&1); then
        log "${pot_version}"
        log "Error: OT training requires POT>=0.9.7,<0.10"
        log "Install it with: python -m pip install 'POT>=0.9.7,<0.10'"
        exit 2
    fi
    log "Using POT ${pot_version} for the optimal-transport loss"
fi

if [ -z "${ssl_tag}" ]; then
    ssl_tag=$(basename "${ssl_config}" .yaml)
fi
if [ -z "${ssl_stats_dir}" ]; then
    ssl_stats_dir=${expdir}/ssl_stats_${ssl_tag}
fi
if [ -z "${ssl_exp}" ]; then
    ssl_exp=${expdir}/ssl_${ssl_tag}
fi
if [ -z "${unit_dir}" ]; then
    unit_dir=${expdir}/units_${ssl_tag}
fi
if [ -z "${asr_tag}" ]; then
    asr_tag=${src_lang}_${tgt_lang}
fi

data_audio=${dumpdir}/audio_raw
data_feats=${dumpdir}/raw
ssl_token_list=${ssl_stats_dir}/tokens.txt

train_sp_sets=
if [ -n "${speed_perturb_factors}" ]; then
    for factor in ${speed_perturb_factors}; do
        if "${python}" -c "assert ${factor} != 1.0" 2>/dev/null; then
            train_sp_sets+="${train_set}_sp${factor} "
        fi
    done
    asr_train_set=${train_set}_sp
else
    asr_train_set=${train_set}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! "${skip_data_prep}"; then
    log "Stage 1: Prepare SAP audio with the shared ASR2 data stages"
    data_opts=(
        --stage 2
        --stop_stage 4
        --skip_eval true
        --use_lm false
        --ngpu "${ngpu}"
        --nj "${nj}"
        --dumpdir "${dumpdir}"
        --expdir "${expdir}"
        --audio_format "${audio_format}"
        --fs "${fs}"
        --min_wav_duration "${min_wav_duration}"
        --max_wav_duration "${max_wav_duration}"
        --speed_perturb_factors "${speed_perturb_factors}"
        --local_data_opts "${local_data_opts}"
        --train_set "${train_set}"
        --valid_set "${valid_set}"
        --test_sets "${test_sets}"
        --src_lang "${src_lang}"
        --tgt_lang "${tgt_lang}"
    )
    ./asr2.sh "${data_opts[@]}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! "${skip_ssl_train}"; then
    log "Stage 2: Collect SSL training shapes"
    train_scp=${data_audio}/${train_set}/wav.scp
    valid_scp=${data_audio}/${valid_set}/wav.scp
    [ ! -f "${train_scp}" ] && { log "Error: missing ${train_scp}; run stage 1"; exit 1; }
    [ ! -f "${valid_scp}" ] && { log "Error: missing ${valid_scp}; run stage 1"; exit 1; }

    logdir=${ssl_stats_dir}/logdir
    mkdir -p "${logdir}"
    printf '<blank>\n<unk>\n<sos/eos>\n' > "${ssl_token_list}"
    _nj=$(min "${nj}" "$(wc -l < "${train_scp}")" "$(wc -l < "${valid_scp}")")

    train_splits=()
    valid_splits=()
    for n in $(seq "${_nj}"); do
        train_splits+=("${logdir}/train.${n}.scp")
        valid_splits+=("${logdir}/valid.${n}.scp")
    done
    utils/split_scp.pl "${train_scp}" "${train_splits[@]}"
    utils/split_scp.pl "${valid_scp}" "${valid_splits[@]}"

    log "SSL collect-stats started: ${logdir}/stats.*.log"
    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${logdir}/stats.JOB.log" \
        "${python}" -m espnet2.bin.ssl_train \
        --collect_stats true \
        --config "${ssl_config}" \
        --use_preprocessor false \
        --token_list "${ssl_token_list}" \
        --train_data_path_and_name_and_type "${train_scp},speech,sound" \
        --valid_data_path_and_name_and_type "${valid_scp},speech,sound" \
        --train_shape_file "${logdir}/train.JOB.scp" \
        --valid_shape_file "${logdir}/valid.JOB.scp" \
        --output_dir "${logdir}/stats.JOB" \
        ${ssl_args}

    aggregate_opts=()
    for n in $(seq "${_nj}"); do
        aggregate_opts+=(--input_dir "${logdir}/stats.${n}")
    done
    "${python}" -m espnet2.bin.aggregate_stats_dirs \
        "${aggregate_opts[@]}" --skip_sum_stats --output_dir "${ssl_stats_dir}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! "${skip_ssl_train}"; then
    log "Stage 3: Train WavLM + Gumbel K-means jointly"
    train_scp=${data_audio}/${train_set}/wav.scp
    valid_scp=${data_audio}/${valid_set}/wav.scp
    [ ! -f "${ssl_stats_dir}/train/speech_shape" ] && {
        log "Error: missing SSL stats; run stage 2"
        exit 1
    }
    [ ! -f "${ssl_token_list}" ] && {
        mkdir -p "$(dirname "${ssl_token_list}")"
        printf '<blank>\n<unk>\n<sos/eos>\n' > "${ssl_token_list}"
    }
    mkdir -p "${ssl_exp}"
    printf '%s --stage 3 "$@"\n' "${run_args}" > "${ssl_exp}/run.sh"
    chmod +x "${ssl_exp}/run.sh"

    fold_length=$((ssl_speech_fold_length * 100))
    jobname=${ssl_exp}/train.log
    if echo "${cuda_cmd}" | grep -q -e queue.pl -e queue-freegpu.pl; then
        jobname=$(basename "${ssl_exp}")
    fi
    log "SSL training started: ${ssl_exp}/train.log"
    # shellcheck disable=SC2086
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${ssl_exp}/train.log" \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${ssl_exp}/.dist_init_" \
        --multiprocessing_distributed true -- \
        "${python}" -m espnet2.bin.ssl_train \
        --config "${ssl_config}" \
        --use_preprocessor false \
        --token_list "${ssl_token_list}" \
        --train_data_path_and_name_and_type "${train_scp},speech,sound" \
        --valid_data_path_and_name_and_type "${valid_scp},speech,sound" \
        --train_shape_file "${ssl_stats_dir}/train/speech_shape" \
        --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
        --fold_length "${fold_length}" \
        --resume true \
        --output_dir "${ssl_exp}" \
        ${ssl_args}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! "${skip_unit_extraction}"; then
    log "Stage 4: Extract deterministic hard units"
    ssl_train_config=${ssl_exp}/config.yaml
    ssl_model_file=${ssl_exp}/${inference_ssl_model}
    [ ! -f "${ssl_train_config}" ] && { log "Error: missing ${ssl_train_config}"; exit 1; }
    [ ! -f "${ssl_model_file}" ] && { log "Error: missing ${ssl_model_file}"; exit 1; }

    remove_repeated=false
    [ "${src_case}" = rm ] && remove_repeated=true
    extract_sets="${train_set} ${train_sp_sets} ${valid_set} ${test_sets}"
    for dset in ${extract_sets}; do
        wav_scp=${data_audio}/${dset}/wav.scp
        [ ! -f "${wav_scp}" ] && { log "Error: missing ${wav_scp}"; exit 1; }
        log "Extract units: ${dset}"
        "${python}" -m espnet2.bin.gumbel_kmeans_extract \
            --task ssl \
            --data_path_and_name_and_type "${wav_scp},speech,sound" \
            --train_config "${ssl_train_config}" \
            --model_file "${ssl_model_file}" \
            --output_dir "${unit_dir}/${dset}" \
            --remove_repeated "${remove_repeated}" \
            --ngpu "${extract_ngpu}" \
            --batch_size 1 \
            --num_workers 1
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Convert unit IDs and prepare downstream ASR data"
    mkdir -p "${unit_dir}" "${data_feats}"
    unit_map=${unit_dir}/distinct_cjk_token_list
    "${python}" -c \
        "for i in range(${nclusters}): print(i, chr(int('4e00', 16) + i))" \
        > "${unit_map}"

    prepare_sets="${train_set} ${train_sp_sets} ${valid_set} ${test_sets}"
    for dset in ${prepare_sets}; do
        labels=${unit_dir}/${dset}/pseudo_labels.txt
        source_audio_dir=${data_audio}/${dset}
        target_dir=${data_feats}/${dset}
        [ ! -f "${labels}" ] && { log "Error: missing ${labels}; run stage 4"; exit 1; }
        mkdir -p "${target_dir}"

        awk '
            NR == FNR { token[$1] = $2; next }
            {
                output = ""
                for (i = 2; i <= NF; ++i) {
                    if (!($i in token)) {
                        print "Unknown unit ID: " $i > "/dev/stderr"
                        exit 1
                    }
                    output = output token[$i]
                }
                print $1, output
            }
        ' "${unit_map}" "${labels}" > "${target_dir}/text.${src_case}.${src_lang}"
        cp "${source_audio_dir}/text" "${target_dir}/text"
        cp "${source_audio_dir}/text" "${target_dir}/text.${tgt_case}.${tgt_lang}"
        cp "${source_audio_dir}/utt2spk" "${target_dir}/utt2spk"
        echo raw > "${target_dir}/feats_type"
        utils/fix_data_dir.sh \
            --utt_extra_files "text.${src_case}.${src_lang} text.${tgt_case}.${tgt_lang}" \
            "${target_dir}"
    done

    if [ -n "${speed_perturb_factors}" ]; then
        combine_dirs=("${data_feats}/${train_set}")
        for dset in ${train_sp_sets}; do
            combine_dirs+=("${data_feats}/${dset}")
        done
        utils/combine_data.sh \
            --extra_files "text.${src_case}.${src_lang} text.${tgt_case}.${tgt_lang}" \
            "${data_feats}/${asr_train_set}" "${combine_dirs[@]}"
        echo raw > "${data_feats}/${asr_train_set}/feats_type"
    fi
    cp "${data_feats}/${asr_train_set}/text.${tgt_case}.${tgt_lang}" \
        "${data_feats}/lm_train.txt"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! "${skip_asr_train}"; then
    log "Stage 6: Train and evaluate the downstream discrete-token ASR model"
    asr2_opts=(
        --stage "${downstream_stage}"
        --stop_stage "${downstream_stop_stage}"
        --skip_data_prep true
        --skip_eval "${skip_eval}"
        --use_lm false
        --kmeans_method gumbel
        --nclusters "${nclusters}"
        --ngpu "${ngpu}"
        --nj "${nj}"
        --inference_nj "${inference_nj}"
        --gpu_inference "${gpu_inference}"
        --dumpdir "${dumpdir}"
        --expdir "${expdir}"
        --src_lang "${src_lang}"
        --tgt_lang "${tgt_lang}"
        --src_case "${src_case}"
        --tgt_case "${tgt_case}"
        --src_token_type "${src_token_type}"
        --src_nbpe "${src_nbpe}"
        --tgt_token_type "${tgt_token_type}"
        --tgt_nbpe "${tgt_nbpe}"
        --speed_perturb_factors "${speed_perturb_factors}"
        --asr_config "${asr_config}"
        --asr_tag "${asr_tag}"
        --train_set "${train_set}"
        --valid_set "${valid_set}"
        --test_sets "${test_sets}"
        --src_bpe_train_text "${data_feats}/${asr_train_set}/text.${src_case}.${src_lang}"
        --tgt_bpe_train_text "${data_feats}/${asr_train_set}/text.${tgt_case}.${tgt_lang}"
        --lm_train_text "${data_feats}/${asr_train_set}/text.${tgt_case}.${tgt_lang}"
    )
    [ -n "${asr_args}" ] && asr2_opts+=(--asr_args "${asr_args}")
    [ -n "${inference_config}" ] && asr2_opts+=(--inference_config "${inference_config}")
    [ -n "${inference_args}" ] && asr2_opts+=(--inference_args "${inference_args}")
    ./asr2.sh "${asr2_opts[@]}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
