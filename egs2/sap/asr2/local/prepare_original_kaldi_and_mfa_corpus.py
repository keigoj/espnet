#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import wave
from collections import defaultdict
from pathlib import Path


BASE = Path(os.environ.get("SAP_PROCESSED_DIR", "/home/hojo/dataset/SAP0430_processed"))


def read_manifest_paths(split: str):
    rows = []
    path = BASE / "manifest" / f"{split}.tsv"
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        wav_path = line.split("\t", 1)[0]
        utt = Path(wav_path).stem
        local_path = BASE / "data" / "processed" / split / f"{utt}.wav"
        rows.append((utt, local_path))
    return rows


def read_transcript_speakers(split: str):
    path = BASE / "manifest" / f"{split}.transcripts.tsv"
    speakers = {}
    if not path.exists():
        return speakers
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("id") and row.get("speaker"):
                speakers[row["id"]] = row["speaker"]
    return speakers


def wav_duration(path: Path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def safe_symlink(target: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    target_str = str(target)
    if link.exists() or link.is_symlink():
        if link.is_symlink() and os.readlink(link) == target_str:
            return
        link.unlink()
    link.symlink_to(target_str)


def prepare_split(split: str):
    rows = read_manifest_paths(split)
    texts = (BASE / "manifest" / f"{split}.wrd.without.parentheses").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(rows) != len(texts):
        raise ValueError(f"{split}: manifest/text length mismatch: {len(rows)} vs {len(texts)}")

    speakers = read_transcript_speakers(split)
    kaldi_dir = BASE / "kaldi" / split
    corpus_dir = BASE / "mfa" / "corpus" / split
    kaldi_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    spk2utt = defaultdict(list)
    wav_lines = []
    text_lines = []
    utt2spk_lines = []
    utt2dur_lines = []

    for (utt, wav_path), text in zip(rows, texts):
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)
        spk = speakers.get(utt, utt.rsplit("_", 2)[0])
        duration = wav_duration(wav_path)
        normalized = text.strip().upper()

        wav_lines.append(f"{utt} {wav_path}")
        text_lines.append(f"{utt} {normalized}")
        utt2spk_lines.append(f"{utt} {spk}")
        utt2dur_lines.append(f"{utt} {duration:.6f}")
        spk2utt[spk].append(utt)

        spk_dir = corpus_dir / spk
        safe_symlink(wav_path, spk_dir / f"{utt}.wav")
        (spk_dir / f"{utt}.lab").write_text(normalized + "\n", encoding="utf-8")

    (kaldi_dir / "wav.scp").write_text("\n".join(wav_lines) + "\n", encoding="utf-8")
    (kaldi_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (kaldi_dir / "utt2spk").write_text("\n".join(utt2spk_lines) + "\n", encoding="utf-8")
    (kaldi_dir / "utt2dur").write_text("\n".join(utt2dur_lines) + "\n", encoding="utf-8")
    with (kaldi_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")

    return split, len(rows), len(spk2utt), sum(float(x.rsplit(" ", 1)[1]) for x in utt2dur_lines) / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args()
    for row in [prepare_split(split) for split in args.splits]:
        print("\t".join([row[0], str(row[1]), str(row[2]), f"{row[3]:.6f}"]))


if __name__ == "__main__":
    main()
