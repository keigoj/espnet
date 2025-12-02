#!/bin/bash
set -euo pipefail

export PATH="/home/hojo/espnet_multi2/tools/sctk/bin:$PATH"

if [ $# -lt 5 ]; then
    echo "Usage: $0 hyp1.trn hyp2.trn hyp3.trn ref.trn output_dir"
    exit 1
fi

hyp1_trn=$1
hyp2_trn=$2
hyp3_trn=$3
ref_trn=$4
outdir=$5
mkdir -p "$outdir"

# 0. trn -> ctm （擬似0.1秒刻み）
trn_to_ctm() {
    local trn=$1
    local ctm=$2
    python3 - "$trn" "$ctm" <<'PY'
import sys
from pathlib import Path

trn_path, ctm_path = map(Path, sys.argv[1:])
lines = trn_path.read_text(encoding="utf-8").splitlines()
rows = []
for line in lines:
    sent, utt = line.rsplit("(", 1)
    tokens = sent.strip().split()
    utt = utt.rstrip(")\n")
    for idx, tok in enumerate(tokens):
        start = 0.1 * idx
        dur = 0.1
        rows.append(f"{utt} 1 {start:.2f} {dur:.2f} {tok} 1.0")
Path(ctm_path).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
}

hyp1_ctm=$outdir/hyp1.ctm
hyp2_ctm=$outdir/hyp2.ctm
hyp3_ctm=$outdir/hyp3.ctm
ref_ctm=$outdir/ref.ctm

trn_to_ctm "$hyp1_trn" "$hyp1_ctm"
trn_to_ctm "$hyp2_trn" "$hyp2_ctm"
trn_to_ctm "$hyp3_trn" "$hyp3_ctm"

# 1. ROVER（参照は投票に混ぜず評価用に保持）
combined_ctm=$outdir/combined.ctm
rover \
    -h "$hyp1_ctm" ctm \
    -h "$hyp2_ctm" ctm \
    -h "$hyp3_ctm" ctm \
    -m avgconf \
    -o "$combined_ctm"

# 2. ctm -> text（投票結果 & 参照）
ctm_to_trn() {
    local ctm=$1
    local trn=$2
    python3 - "$ctm" "$trn" <<'PY'
import sys
from pathlib import Path
from collections import defaultdict

ctm_path, trn_path = map(Path, sys.argv[1:])
entries = defaultdict(list)
with ctm_path.open() as f:
    for line in f:
        if not line.strip():
            continue
        utt, _, start, _, tok, *_ = line.split()
        entries[utt].append((float(start), tok))

lines = []
for utt in sorted(entries):
    seq = "".join(tok for _, tok in sorted(entries[utt]))
    tokens = " ".join(seq)
    lines.append(f"{tokens}\t({utt})")

trn_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

combined_trn=$outdir/combined.trn
ctm_to_trn "$combined_ctm" "$combined_trn"

# 3. CER算出（既存スクリプトを利用）
sclite \
    -r "$ref_trn" trn \
    -h "$combined_trn" trn \
    -i rm -o all stdout \
    > "$outdir/result.txt"

echo "ROVER結果: $combined_ctm"
