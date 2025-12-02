#!/bin/sh

data_dir=$1
subset_name=$2

echo $data_dir
echo $subset_name

for sub in ${subset_name[@]}; do
    echo "Processing subset: ${sub}"
    
    input_data_dir=${data_dir}/${sub}
    output_file=${data_dir}/${sub}_detail.txt

    if [ ! -d "${input_data_dir}" ]; then
        echo "  Warning: ${input_data_dir} not found. Skipping." >&2
        continue
    fi

    if [ -f "${input_data_dir}/spk2utt" ]; then
        num_spk=$(wc -l < "${input_data_dir}/spk2utt")
    else
        num_spk="N/A"
    fi
    text_file=${input_data_dir}/text
    utt2dur_file=${input_data_dir}/utt2dur
    segments_file=${input_data_dir}/segments

    if [ -f "${text_file}" ]; then
        char_count=$(python3 - "${text_file}" <<'PY'
import pathlib
import sys

text_path = pathlib.Path(sys.argv[1])
if not text_path.is_file():
    print("N/A")
    sys.exit(0)

total = 0
with text_path.open(encoding="utf-8") as text_f:
    for line in text_f:
        parts = line.rstrip("\n").split(maxsplit=1)
        if len(parts) < 2:
            continue
        cleaned = "".join(parts[1].split())
        total += len(cleaned)

print(total)
PY
        )
    else
        char_count="N/A"
    fi

    duration_info=$(python3 - "${utt2dur_file}" "${segments_file}" "${input_data_dir}/wav.scp" <<-'PY'
	import pathlib
	import shlex
	import sys

	utt2dur_path = pathlib.Path(sys.argv[1])
	segments_path = pathlib.Path(sys.argv[2])
	wavscp_path = pathlib.Path(sys.argv[3])

	AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".wv")

	def resolve_candidate(token: str, base_dir: pathlib.Path):
	    if token in {"-", "|"}:
	        return None
	    path = pathlib.Path(token)
	    if path.is_file():
	        return path
	    rel_path = base_dir / token
	    if rel_path.is_file():
	        return rel_path
	    return None

	def sum_utt2dur(path: pathlib.Path) -> float:
	    total = 0.0
	    with path.open(encoding="utf-8") as f:
	        for line in f:
	            parts = line.strip().split()
	            if len(parts) < 2:
	                continue
	            try:
	                total += float(parts[1])
	            except ValueError:
	                continue
	    return total

	def sum_segments(path: pathlib.Path) -> float:
	    total = 0.0
	    with path.open(encoding="utf-8") as f:
	        for line in f:
	            parts = line.strip().split()
	            if len(parts) < 4:
	                continue
	            try:
	                start = float(parts[2])
	                end = float(parts[3])
	            except ValueError:
	                continue
	            if end > start:
	                total += end - start
	    return total

	def sum_wavscp(path: pathlib.Path):
	    try:
	        import soundfile as sf
	    except ImportError:
	        return None
	    base_dir = path.parent
	    total = 0.0
	    with path.open(encoding="utf-8") as f:
	        for line in f:
	            line = line.strip()
	            if not line:
	                continue
	            parts = line.split(maxsplit=1)
	            if len(parts) < 2:
	                continue
	            try:
	                tokens = shlex.split(parts[1])
	            except ValueError:
	                continue
	            candidate = None
	            for tok in tokens:
	                resolved = resolve_candidate(tok, base_dir)
	                if resolved is not None:
	                    candidate = resolved
	                    break
	            if candidate is None:
	                for tok in tokens:
	                    lower_tok = tok.lower()
	                    if any(lower_tok.endswith(ext) for ext in AUDIO_EXTS):
	                        resolved = resolve_candidate(tok, base_dir)
	                        if resolved is not None:
	                            candidate = resolved
	                            break
	            if candidate is None:
	                continue
	            try:
	                with sf.SoundFile(candidate) as sf_desc:
	                    total += len(sf_desc) / sf_desc.samplerate
	            except Exception:
	                continue
	    return total if total > 0 else None

	total = None
	if utt2dur_path.is_file():
	    total = sum_utt2dur(utt2dur_path)
	elif segments_path.is_file():
	    total = sum_segments(segments_path)
	elif wavscp_path.is_file():
	    total = sum_wavscp(wavscp_path)

	if total is None:
	    print("N/A")
	    print("N/A")
	    sys.exit(0)

	hours = int(total // 3600)
	minutes = int((total % 3600) // 60)
	seconds = total - hours * 3600 - minutes * 60

	print(f"{total:.2f}")
	print(f"{hours}:{minutes:02d}:{seconds:06.3f}")
PY
    )

    duration_sec=$(printf '%s\n' "${duration_info}" | head -n 1)
    duration_hms=$(printf '%s\n' "${duration_info}" | tail -n 1)

    {
        echo "num_speakers: ${num_spk}"
        echo "total_characters: ${char_count}"
        echo "total_audio_seconds: ${duration_sec}"
        echo "total_audio_hms: ${duration_hms}"
    } > "${output_file}"

    echo "  num speakers: ${num_spk}"
    echo "  total characters: ${char_count}"
    echo "  total audio seconds: ${duration_sec}"
    echo "  total audio (H:MM:SS.sss): ${duration_hms}"
done
