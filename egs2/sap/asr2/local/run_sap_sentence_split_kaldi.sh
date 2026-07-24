#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base="/home/hojo/dataset/SAP0430_processed"
conda_sh="/home/hojo/anaconda3/etc/profile.d/conda.sh"
conda_env="aligner"
jobs=16
segment_jobs=8
acoustic_model="english_us_arpa"
dictionary="english_us_arpa"
g2p_model="english_us_mfa"
skip_original_align=0
splits=()

usage() {
  cat <<'EOF'
Usage:
  run_sap_sentence_split_kaldi.sh --splits SPLIT [SPLIT ...] [options]

Options:
  --base DIR                 SAP_0430_processed-style data root
  --conda-sh FILE            Conda profile script
  --conda-env NAME           Conda env containing MFA
  --jobs N                   MFA align jobs
  --segment-jobs N           MFA segment jobs
  --acoustic-model NAME      MFA acoustic model
  --dictionary NAME          MFA dictionary
  --g2p-model NAME           MFA G2P model
  --skip-original-align      Reuse existing mfa/aligned/<split>_textgrid

Example:
  local/run_sap_sentence_split_kaldi.sh --splits test1 test2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      base="$2"
      shift 2
      ;;
    --conda-sh)
      conda_sh="$2"
      shift 2
      ;;
    --conda-env)
      conda_env="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --segment-jobs)
      segment_jobs="$2"
      shift 2
      ;;
    --acoustic-model)
      acoustic_model="$2"
      shift 2
      ;;
    --dictionary)
      dictionary="$2"
      shift 2
      ;;
    --g2p-model)
      g2p_model="$2"
      shift 2
      ;;
    --skip-original-align)
      skip_original_align=1
      shift
      ;;
    --splits)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        splits+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#splits[@]} -eq 0 ]]; then
  echo "Specify at least one split with --splits." >&2
  usage >&2
  exit 2
fi

export SAP_PROCESSED_DIR="$base"

source "$conda_sh"
conda activate "$conda_env"

echo "==> Removing stale split-specific intermediate outputs"
for split in "${splits[@]}"; do
  rm -rf \
    "$base/kaldi/$split" \
    "$base/mfa/corpus/$split" \
    "$base/kaldi_sentence_first_base/$split" \
    "$base/kaldi_sentence_first_mfa_with_unaligned_short/$split" \
    "$base/mfa/corpus_sentence_first_long_for_segment/$split" \
    "$base/mfa/audio_sentence_first_long/$split" \
    "$base/mfa/segmented/sentence_first_long/$split" \
    "$base/mfa/corpus_sentence_first_segmented_for_align/$split" \
    "$base/mfa/aligned/sentence_first_segmented_long/${split}_textgrid"
  if [[ "$skip_original_align" -eq 0 ]]; then
    rm -rf \
      "$base/mfa/aligned/${split}_textgrid" \
      "$base/mfa/temp/${split}_align"
  fi
  rm -rf \
    "$base/mfa/temp/segment_sentence_first_long_$split" \
    "$base/mfa/temp/align_sentence_first_segmented_long_$split"
done

echo "==> Preparing original Kaldi data and MFA corpus"
python "$script_dir/prepare_original_kaldi_and_mfa_corpus.py" --splits "${splits[@]}"

if [[ "$skip_original_align" -eq 0 ]]; then
  for split in "${splits[@]}"; do
    echo "==> MFA align original split: $split"
    mfa align \
      "$base/mfa/corpus/$split" \
      "$dictionary" \
      "$acoustic_model" \
      "$base/mfa/aligned/${split}_textgrid" \
      --g2p_model_path "$g2p_model" \
      --output_format short_textgrid \
      -t "$base/mfa/temp/${split}_align" \
      -j "$jobs" \
      --clean \
      --no_final_clean \
      --overwrite
  done
else
  echo "==> Skipping original MFA align"
fi

echo "==> Building sentence-first base Kaldi and long-sentence corpus"
python "$script_dir/build_sentence_split_base_and_long_corpus.py" --splits "${splits[@]}"

for split in "${splits[@]}"; do
  long_tsv="$base/mfa/analysis_sentence_first/${split}_long_sentences_for_mfa_segment.tsv"
  long_count="$(awk 'NR > 1 && NF { c++ } END { print c + 0 }' "$long_tsv" 2>/dev/null || echo 0)"
  if [[ "$long_count" -eq 0 ]]; then
    echo "==> No over-30s sentence segments for MFA segment: $split"
    continue
  fi

  echo "==> MFA segment over-30s sentence clips: $split ($long_count files)"
  mfa segment \
    "$base/mfa/corpus_sentence_first_long_for_segment/$split" \
    "$dictionary" \
    "$acoustic_model" \
    "$base/mfa/segmented/sentence_first_long/$split" \
    --output_format short_textgrid \
    -t "$base/mfa/temp/segment_sentence_first_long_$split" \
    -j "$segment_jobs" \
    --clean \
    --no_final_clean \
    --overwrite \
    --no_cuda
done

echo "==> Exporting MFA segment DBs to align corpus"
python "$script_dir/export_mfa_segment_db_to_align_corpus.py" --splits "${splits[@]}"

for split in "${splits[@]}"; do
  segment_tsv="$base/mfa/analysis_sentence_first/${split}_sentence_first_mfa_segment_segments.tsv"
  segment_count="$(awk 'NR > 1 && NF { c++ } END { print c + 0 }' "$segment_tsv" 2>/dev/null || echo 0)"
  if [[ "$segment_count" -eq 0 ]]; then
    echo "==> No MFA-segmented utterances to align: $split"
    continue
  fi

  echo "==> MFA align sentence-segmented clips: $split ($segment_count utterances)"
  mfa align \
    "$base/mfa/corpus_sentence_first_segmented_for_align/$split" \
    "$dictionary" \
    "$acoustic_model" \
    "$base/mfa/aligned/sentence_first_segmented_long/${split}_textgrid" \
    --g2p_model_path "$g2p_model" \
    --output_format short_textgrid \
    -t "$base/mfa/temp/align_sentence_first_segmented_long_$split" \
    -j "$jobs" \
    --clean \
    --no_final_clean \
    --overwrite
done

echo "==> Building final Kaldi data with 50Hz phone labels"
python "$script_dir/build_final_segmented_kaldi_with_phones.py" --splits "${splits[@]}"

echo "==> Done: $base/kaldi_sentence_first_mfa_with_unaligned_short"
