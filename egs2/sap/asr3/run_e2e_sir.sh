#!/usr/bin/env bash
# SAP mild-reference SIR, preserving the base run_e2e.sh training schedule.
set -euo pipefail

# Reduce fragmentation across variable-length audio batches; honor explicit settings.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${recipe_dir}"

stage=5
stop_stage=11
sir_dir=data/sir_mild_le2
# Use model_conf.sir_weight from YAML unless explicitly overridden.
sir_weight=
asr_config=conf/train_e2e_gumbel_kmeans_wavlm_large_sir.yaml
asr_exp=exp/asr_e2e_wavlm_large_l21_gumbel_km2000_full_warmup30_ep90_en_sir_mild_le2_w01
asr_stats_dir=exp/asr_stats_e2e_raw_en_bpe5000_sp_sir_mild_le2
asr_args=
help_message="Usage: $0 [--stage 5|10|11] [--sir_dir DIR] [--sir_weight VALUE] [--asr_exp DIR]"
. utils/parse_options.sh
if [ "$#" -ne 0 ]; then
    echo "${help_message}" >&2
    exit 2
fi
if [ ! -f "${sir_dir}/summary.json" ]; then
    echo "Missing ${sir_dir}/summary.json; run local/prepare_sir_mild.py first." >&2
    exit 1
fi
for split in train valid; do
    for filename in healthy.scp sir_pair_mask; do
        if [ ! -s "${sir_dir}/${split}/${filename}" ]; then
            echo "Missing ${sir_dir}/${split}/${filename}" >&2
            exit 1
        fi
    done
done

sir_args=
if [ -n "${sir_weight}" ]; then
    sir_args="--model_conf sir_weight=${sir_weight}"
fi
sir_args+=" --train_data_path_and_name_and_type ${sir_dir}/train/healthy.scp,healthy_speech,sound"
sir_args+=" --train_data_path_and_name_and_type ${sir_dir}/train/sir_pair_mask,sir_pair_mask,text_int"
sir_args+=" --valid_data_path_and_name_and_type ${sir_dir}/valid/healthy.scp,healthy_speech,sound"
sir_args+=" --valid_data_path_and_name_and_type ${sir_dir}/valid/sir_pair_mask,sir_pair_mask,text_int"
if [ -n "${asr_args}" ]; then
    sir_args+=" ${asr_args}"
fi

exec bash ./run_e2e.sh \
    --stage "${stage}" --stop_stage "${stop_stage}" \
    --asr_config "${asr_config}" \
    --asr_exp "${asr_exp}" --asr_stats_dir "${asr_stats_dir}" \
    --asr_args "${sir_args}"
