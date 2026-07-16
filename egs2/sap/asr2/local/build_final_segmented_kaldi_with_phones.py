#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import argparse
import os
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(os.environ.get("SAP_PROCESSED_DIR", "/home/hojo/dataset/SAP0430_processed"))
OUT_ROOT = BASE / "kaldi_sentence_first_mfa_with_unaligned_short"
BASE_ROOT = BASE / "kaldi_sentence_first_base"
ANALYSIS = BASE / "mfa" / "analysis_sentence_first"
SEGMENT_ALIGN_ROOT = BASE / "mfa" / "aligned" / "sentence_first_segmented_long"
ORIGINAL_ALIGN_ROOT = BASE / "mfa" / "aligned"
CLIP_ROOT = BASE / "mfa" / "audio_sentence_first_long"

FRAME_SHIFT = 0.02
FRAME_RATE = 50.0
SIL_LABEL = "SIL"
MAX_DURATION = 30.0

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sentence_builder", SCRIPT_DIR / "sap_sentence_split_utils.py"
)
sentence_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentence_builder)


def read_kaldi_table(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split(maxsplit=1))
    return rows


def read_base_rows(split: str):
    root = BASE_ROOT / split
    text = dict(read_kaldi_table(root / "text"))
    utt2spk = dict(read_kaldi_table(root / "utt2spk"))
    rows = []
    with (root / "segments").open(encoding="utf-8") as f:
        for line in f:
            utt, rec, begin, end = line.rstrip("\n").split()
            rows.append(
                {
                    "utt": utt,
                    "rec": rec,
                    "begin": float(begin),
                    "end": float(end),
                    "text": text[utt],
                    "spk": utt2spk[utt],
                    "source": "base",
                }
            )
    wavs = dict(read_kaldi_table(root / "wav.scp"))
    return rows, wavs


def read_segment_rows(split: str):
    path = ANALYSIS / f"{split}_sentence_first_mfa_segment_segments.tsv"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        clip_id, utt, spk, begin, end, _duration, text = line.split("\t", 6)
        rows.append(
            {
                "utt": utt,
                "rec": clip_id,
                "begin": float(begin),
                "end": float(end),
                "text": text,
                "spk": spk,
                "source": "sentence_long_mfa_segment",
            }
        )
    return rows


def build_textgrid_index(*roots: Path):
    paths = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.TextGrid"):
            paths[path.stem] = path
    return paths


def parse_short_textgrid_tier(path: Path, tier_name: str):
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    for i, line in enumerate(lines):
        if line != '"IntervalTier"':
            continue
        if i + 4 >= len(lines) or lines[i + 1].strip('"') != tier_name:
            continue
        n = int(lines[i + 4])
        j = i + 5
        intervals = []
        for _ in range(n):
            begin = float(lines[j])
            end = float(lines[j + 1])
            label = lines[j + 2].strip('"') or SIL_LABEL
            intervals.append((begin, end, label))
            j += 3
        return intervals
    return []


def label_frames(intervals, begin: float, end: float):
    duration = max(0.0, end - begin)
    num_frames = max(1, int(duration * FRAME_RATE + 0.5 + 1e-8))
    starts = [x[0] for x in intervals]
    labels = []
    for frame_index in range(num_frames):
        time = begin + (frame_index + 0.5) * FRAME_SHIFT
        if time >= end:
            time = max(begin, end - 1e-6)
        idx = bisect_right(starts, time) - 1
        if idx < 0 or idx >= len(intervals):
            labels.append(SIL_LABEL)
            continue
        interval_begin, interval_end, label = intervals[idx]
        labels.append(label if interval_begin <= time < interval_end else SIL_LABEL)
    return labels


def add_clip_wavs(split: str, wavs: dict[str, str], segment_rows):
    for row in segment_rows:
        rec = row["rec"]
        if rec in wavs:
            continue
        path = CLIP_ROOT / split / row["spk"] / f"{rec}.wav"
        if not path.exists():
            raise FileNotFoundError(path)
        wavs[rec] = str(path)


def write_kaldi(split: str, rows, wavs):
    out = OUT_ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x["utt"])
    wavs = dict(sorted(wavs.items()))

    spk2utt = defaultdict(list)
    for row in rows:
        spk2utt[row["spk"]].append(row["utt"])

    with (out / "wav.scp").open("w", encoding="utf-8") as f:
        for rec, wav in wavs.items():
            f.write(f"{rec} {wav}\n")
    with (out / "segments").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['utt']} {row['rec']} {row['begin']:.3f} {row['end']:.3f}\n")
    with (out / "text").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['utt']} {row['text']}\n")
    with (out / "utt2spk").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['utt']} {row['spk']}\n")
    with (out / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")


def write_phone_labels(split: str, rows):
    out = OUT_ROOT / split
    textgrids = build_textgrid_index(
        ORIGINAL_ALIGN_ROOT / f"{split}_textgrid",
        SEGMENT_ALIGN_ROOT / f"{split}_textgrid",
    )
    phone_cache = {}
    missing = []
    utt2phones = []
    utt2frames = []

    for row in sorted(rows, key=lambda x: x["utt"]):
        utt = row["utt"]
        rec = row["rec"]
        tg_path = textgrids.get(rec)
        if tg_path is None:
            utt2phones.append(utt)
            utt2frames.append(f"{utt} 0")
            missing.append(f"{utt}\t{rec}\tmissing_textgrid\t{row['source']}")
            continue
        intervals = phone_cache.get(rec)
        if intervals is None:
            intervals = parse_short_textgrid_tier(tg_path, "phones")
            phone_cache[rec] = intervals
        if not intervals:
            utt2phones.append(utt)
            utt2frames.append(f"{utt} 0")
            missing.append(f"{utt}\t{rec}\tmissing_phone_tier\t{row['source']}")
            continue
        labels = label_frames(intervals, row["begin"], row["end"])
        utt2phones.append(f"{utt} {' '.join(labels)}")
        utt2frames.append(f"{utt} {len(labels)}")

    (out / "utt2phones_50hz").write_text("\n".join(utt2phones) + "\n", encoding="utf-8")
    (out / "utt2num_frames_50hz").write_text("\n".join(utt2frames) + "\n", encoding="utf-8")
    (out / "missing_phone_labels").write_text(
        "utt\trecording_id\treason\tsource\n"
        + "\n".join(missing)
        + ("\n" if missing else ""),
        encoding="utf-8",
    )
    return missing


def validate(split: str, rows, wavs):
    utt_counts = Counter(row["utt"] for row in rows)
    duplicate_utts = sorted(utt for utt, count in utt_counts.items() if count > 1)
    if duplicate_utts:
        raise ValueError(f"{split}: duplicate utterance ids: {duplicate_utts[:10]}")

    missing_wavs = sorted({row["rec"] for row in rows if row["rec"] not in wavs})
    if missing_wavs:
        raise ValueError(f"{split}: missing wav.scp records: {missing_wavs[:10]}")

    overlong = [row for row in rows if row["end"] - row["begin"] > MAX_DURATION]
    if overlong:
        preview = ", ".join(f"{r['utt']}={r['end']-r['begin']:.3f}" for r in overlong[:10])
        raise ValueError(f"{split}: over {MAX_DURATION}s rows remain: {preview}")


def count_lines(path: Path):
    return sum(1 for _ in path.open(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["dev", "train"])
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary = ["split\tutts\thours\tbase_utts\tmfa_segment_utts\tmissing_phone_labels\tempty_phone_lines"]
    source_summary = ["split\tsource\tutts\thours"]

    for split in args.splits:
        base_rows, wavs = read_base_rows(split)
        segment_rows = read_segment_rows(split)
        add_clip_wavs(split, wavs, segment_rows)
        rows = base_rows + segment_rows
        validate(split, rows, wavs)
        write_kaldi(split, rows, wavs)
        missing = write_phone_labels(split, rows)

        by_source = defaultdict(lambda: [0, 0.0])
        for row in rows:
            duration = row["end"] - row["begin"]
            by_source[row["source"]][0] += 1
            by_source[row["source"]][1] += duration
        for source in sorted(by_source):
            count, seconds = by_source[source]
            source_summary.append(f"{split}\t{source}\t{count}\t{seconds / 3600:.6f}")

        out = OUT_ROOT / split
        phone_lines = (out / "utt2phones_50hz").read_text(encoding="utf-8").splitlines()
        empty_phone_lines = sum(1 for line in phone_lines if len(line.split()) == 1)
        counts = {
            name: count_lines(out / name)
            for name in ("text", "segments", "utt2spk", "utt2phones_50hz", "utt2num_frames_50hz")
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"{split}: line count mismatch: {counts}")

        hours = sum(row["end"] - row["begin"] for row in rows) / 3600
        summary.append(
            "\t".join(
                [
                    split,
                    str(len(rows)),
                    f"{hours:.6f}",
                    str(len(base_rows)),
                    str(len(segment_rows)),
                    str(len(missing)),
                    str(empty_phone_lines),
                ]
            )
        )
        print(summary[-1])

    (ANALYSIS / "sentence_first_final_summary.tsv").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    (ANALYSIS / "sentence_first_final_source_summary.tsv").write_text(
        "\n".join(source_summary) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
