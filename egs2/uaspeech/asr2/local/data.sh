#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# reference: https://github.com/ffxiong/uaspeech/blob/master/s5/local/prepare_uaspeech_data.sh

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# Options
stage=0
stop_stage=2
mlf_root=
audio_root=
datadir=data
nj=16
cleanup=true
use_single_mic=false  # if true, use only M5 mic for dysarthric speakers
speaker_groups="ctl,dys"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${audio_root}" ] && [ -n "${UASPEECH:-}" ]; then
    audio_root="${UASPEECH}"
fi
if [ -z "${mlf_root}" ] && [ -n "${UASPEECH:-}" ]; then
    mlf_root="${UASPEECH}"
fi

if [ -z "${mlf_root}" ]; then
    log "Error: --mlf_root must be set (directory containing 'mlf/<spk>/*_word.mlf')."
    exit 1
fi

if [ -z "${audio_root}" ] || [ ! -d "${audio_root}" ]; then
    log "Error: --audio_root must be set to UA-Speech audio root (or set UASPEECH in db.sh)."
    exit 1
fi

mlfdir="${mlf_root}/mlf"
audiodir="${audio_root}/audio"
workdir="${datadir}/local/uaspeech_prep"
mkdir -p "${workdir}"

speaker_groups=${speaker_groups// /}
IFS=',' read -r -a selected_groups <<< "${speaker_groups}"
if [ ${#selected_groups[@]} -eq 0 ]; then
    log "Error: --speaker_groups must contain at least one group (ctl and/or dys)."
    exit 1
fi
for grp in "${selected_groups[@]}"; do
    case "${grp}" in
        ctl|dys) ;;
        *)
            log "Error: Unsupported group '${grp}'. Use ctl and/or dys."
            exit 1
            ;;
    esac
done

prepare_group() {
    local settyp=$1   # ctl or dys
    local spkdir=$2   # audio subroot for this group
    local -a spksets=(${3})

    local tmpdir="${workdir}/${settyp}"
    mkdir -p "${tmpdir}"

    # Per-speaker files
    for x in "${spksets[@]}"; do
        local mlf="${mlfdir}/${x}/${x}_word.mlf"
        [ ! -f "${mlf}" ] && { log "No such file: ${mlf}"; exit 1; }

        local textfil="${tmpdir}/${x}.text"
        # Extract utt id and transcription: follow kaldi script behavior
        # Grep every line with path containing $x, take basename without ext as utt id,
        # and the next line as transcription.
        awk -v spk="${x}" '
            { lines[NR]=$0 }
            END {
                for (i=1;i<=NR;i++) {
                    if (index(lines[i], spk) > 0) {
                        # extract basename from path token after last slash
                        n=split(lines[i], a, "/");
                        base=a[n]; sub(/\..*$/, "", base);
                        if (i+1<=NR) { print base, lines[i+1]; }
                    }
                }
            }
        ' "${mlf}" > "${textfil}"

        local wavdir="${spkdir}/${x}"
        : > "${tmpdir}/${x}.wav.scp"
        local -a utt_list_cmd
        if "${use_single_mic}"; then
            utt_list_cmd=(awk '$1 ~ /_M5$/ { print $1 }' "${textfil}")
        else
            utt_list_cmd=(cut -d ' ' -f1 "${textfil}")
        fi
        while IFS= read -r utt; do
            wavpath="${wavdir}/${utt}.wav"
            if [ -f "${wavpath}" ]; then
                printf '%s %s\n' "${utt}" "${wavpath}" >> "${tmpdir}/${x}.wav.scp"
            else
                log "Warning: missing wav ${wavpath}, skip"
            fi
        done < <("${utt_list_cmd[@]}")
        LC_ALL=C sort -o "${tmpdir}/${x}.wav.scp" "${tmpdir}/${x}.wav.scp"

        if "${use_single_mic}"; then
            awk '
                NR==FNR { keep[$1]=1; next }
                { utt=$1; if (utt in keep) print $0; }
            ' "${tmpdir}/${x}.wav.scp" "${textfil}" > "${tmpdir}/${x}.text.filtered"
            mv -f "${tmpdir}/${x}.text.filtered" "${tmpdir}/${x}.text"
        fi

        cut -d " " -f 1 "${tmpdir}/${x}.wav.scp" | sed -e "s|^\(\([^_]\+\)_.*\)|\1 \2|g" > "${tmpdir}/${x}.utt2spk"
    done

    # Merge speakers
    local aldir="${workdir}/train_${settyp}all"
    mkdir -p "${aldir}"
    for dd in wav.scp text utt2spk; do
        for x in "${spksets[@]}"; do
            cat "${tmpdir}/${x}.${dd}"
        done | sort > "${aldir}/${dd}"
    done

    if [ "${settyp}" = dys ]; then
        cp "${aldir}/text" "${aldir}/text_org"
        grep -v -e "^F03_B3_C13_M2" -e "^F03_B3_C3_M5" -e "^F03_B3_CW100_M8" -e "^F04_B1_UW8_M8" \
                -e "^F04_B1_UW94_M3" -e "^F04_B1_UW94_M5" -e "^F04_B1_UW95_M6" -e "^F04_B1_UW96_M4" \
                -e "^F04_B1_UW96_M5" -e "^F04_B1_UW99_M3" -e "^F04_B1_UW9_M5" -e "^F04_B2_C10_M4" \
                -e "^F04_B2_C12_M6" -e "^F04_B2_C12_M7" -e "^F04_B2_C13_M5" "${aldir}/text_org" > "${aldir}/text"
        utils/fix_data_dir.sh "${aldir}"
    fi

    awk '{print $1, $2}' "${aldir}/utt2spk" | utils/utt2spk_to_spk2utt.pl > "${aldir}/spk2utt"

    # Split: B1+B3 -> train_like, B2 -> test
    for ss in train test; do
        local ddir="${workdir}/${ss}_${settyp}"
        mkdir -p "${ddir}"
        local blinfo='_B1_|_B3_'
        [ "${ss}" = test ] && blinfo='_B2_'
        for ff in wav.scp text utt2spk; do
            grep -E ${blinfo} "${aldir}/${ff}" > "${ddir}/${ff}" || true
        done
        [ -s "${ddir}/utt2spk" ] && awk '{print $1, $2}' "${ddir}/utt2spk" | utils/utt2spk_to_spk2utt.pl > "${ddir}/spk2utt" || true
        # Remove stale utt2dur if any (may be empty from previous runs)
        [ -f "${ddir}/utt2dur" ] && rm -f "${ddir}/utt2dur"
        [ -s "${ddir}/wav.scp" ] && utils/validate_data_dir.sh --no-feats "${ddir}" || true
    done
}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Speaker lists
    ctl_spk="CF02 CF03 CF04 CM04 CM05 CM06 CM08 CM09 CM10 CM12 CM13"
    dys_spk="F02 F03 F04 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16"

    for grp in "${selected_groups[@]}"; do
        case "${grp}" in
            ctl) spk_list="${ctl_spk}" ;;
            dys) spk_list="${dys_spk}" ;;
        esac
        prepare_group "${grp}" "${audiodir}" "${spk_list}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Build final splits for ESPnet
    mkdir -p "${datadir}/train" "${datadir}/dev" "${datadir}/test"

    # First, create train_all (B1+B3) and test (B2)
    # Combine ctl and dys groups for B1+B3
    mkdir -p "${datadir}/train_all"
    : > "${datadir}/train_all/wav.scp"; : > "${datadir}/train_all/text"; : > "${datadir}/train_all/utt2spk"
    for grp in "${selected_groups[@]}"; do
        for ff in wav.scp text utt2spk; do
            if [ -s "${workdir}/train_${grp}all/${ff}" ]; then
                grep -E '_B1_|_B3_' "${workdir}/train_${grp}all/${ff}" >> "${datadir}/train_all/${ff}" || true
            fi
        done
    done
    [ -s "${datadir}/train_all/utt2spk" ] && awk '{print $1, $2}' "${datadir}/train_all/utt2spk" | utils/utt2spk_to_spk2utt.pl > "${datadir}/train_all/spk2utt"

    # Create test from B2
    mkdir -p "${datadir}/test"
    : > "${datadir}/test/wav.scp"; : > "${datadir}/test/text"; : > "${datadir}/test/utt2spk"
    for grp in "${selected_groups[@]}"; do
        for ff in wav.scp text utt2spk; do
            if [ -s "${workdir}/test_${grp}/${ff}" ]; then
                cat "${workdir}/test_${grp}/${ff}" >> "${datadir}/test/${ff}"
            fi
        done
    done
    [ -s "${datadir}/test/utt2spk" ] && awk '{print $1, $2}' "${datadir}/test/utt2spk" | utils/utt2spk_to_spk2utt.pl > "${datadir}/test/spk2utt"

    # Split train_all into train (90%) and dev (10%)
    mkdir -p "${datadir}/train" "${datadir}/dev"
    
    # Get total number of utterances
    total_utts=$(wc -l < "${datadir}/train_all/utt2spk")
    dev_size=$((($total_utts + 9) / 10))  # 10%

    python3 -c "import random; random.seed(42); ids=list(range(1,$total_utts+1)); sample=sorted(random.sample(ids, $dev_size)); print('\n'.join(map(str, sample)))" \
    > "${workdir}/dev_indices.txt"

    for ff in wav.scp text utt2spk; do
        : > "${datadir}/train/${ff}"
        : > "${datadir}/dev/${ff}"
        awk -v dev="${datadir}/dev/${ff}" -v trn="${datadir}/train/${ff}" '
            NR==FNR { idx[$1]=1; next }           # 1st file: dev_indices を集合化
            (FNR in idx) { print > dev; next }    # 2nd file: 行番号(FNR)が集合にあれば dev
            { print > trn }                       # それ以外は train
        ' "${workdir}/dev_indices.txt" "${datadir}/train_all/${ff}"
    done

    # Generate spk2utt for train and dev
    [ -s "${datadir}/train/utt2spk" ] && awk '{print $1, $2}' "${datadir}/train/utt2spk" | utils/utt2spk_to_spk2utt.pl > "${datadir}/train/spk2utt"
    [ -s "${datadir}/dev/utt2spk" ] && awk '{print $1, $2}' "${datadir}/dev/utt2spk" | utils/utt2spk_to_spk2utt.pl > "${datadir}/dev/spk2utt"

    # Clean up temporary train_all
    rm -rf "${datadir}/train_all"

    # Validate final splits
    for split in train dev test; do
        [ -s "${datadir}/${split}/wav.scp" ] && utils/validate_data_dir.sh --no-feats "${datadir}/${split}"
    done
fi

${cleanup} && rm -rf "${workdir}"

log "Successfully finished. [elapsed=${SECONDS}s]"
