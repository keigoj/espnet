#!/usr/bin/env python3
"""Filter Common Voice data files by removing utterances containing Latin alphabets."""

import argparse
import re
from pathlib import Path
from typing import Set

# Matches ASCII and full-width Latin alphabets.
ALPHABET_PATTERN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "List utterances that include ASCII or full-width alphabets and remove "
            "them from text/utt2spk/spk2utt/wav.scp files."
        )
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        default="egs2/csj/asr1/data/commonvoice",
        help="Root directory containing train_ja/dev_ja/test_ja subdirectories.",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=("train_ja", "dev_ja", "test_ja"),
        help="Subdirectories to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report offending utterances without modifying any files.",
    )
    return parser.parse_args()


def collect_flagged_utts(text_path: Path):
    flagged = []
    if not text_path.is_file():
        return flagged
    with text_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            utt = parts[0]
            transcription = parts[1] if len(parts) > 1 else ""
            if ALPHABET_PATTERN.search(transcription):
                flagged.append((utt, transcription))
    return flagged


def filter_text_like(path: Path, banned_utts: Set[str]):
    if not path.is_file() or not banned_utts:
        return 0
    kept_lines = []
    removed = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                kept_lines.append("\n")
                continue
            parts = line.split(maxsplit=1)
            if not parts:
                kept_lines.append("\n")
                continue
            utt = parts[0]
            if utt in banned_utts:
                removed += 1
                continue
            kept_lines.append(raw_line)
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept_lines)
    return removed


def filter_spk2utt(path: Path, banned_utts: Set[str]):
    if not path.is_file() or not banned_utts:
        return 0
    kept_lines = []
    removed_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                kept_lines.append(raw_line)
                continue
            fields = stripped.split()
            speaker, utts = fields[0], fields[1:]
            filtered_utts = [utt for utt in utts if utt not in banned_utts]
            if len(filtered_utts) != len(utts):
                removed_count += len(utts) - len(filtered_utts)
            if filtered_utts:
                kept_lines.append(" ".join([speaker, *filtered_utts]) + "\n")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept_lines)
    return removed_count


def process_subset(subset_dir: Path, dry_run: bool):
    text_path = subset_dir / "text"
    flagged = collect_flagged_utts(text_path)

    print(f"[{subset_dir.name}] flagged utterances: {len(flagged)}")
    for utt, transcription in flagged:
        print(f"{utt}\t{transcription}")

    if dry_run or not flagged:
        return len(flagged)

    banned_utts = {utt for utt, _ in flagged}

    removed_text = filter_text_like(text_path, banned_utts)
    removed_utt2spk = filter_text_like(subset_dir / "utt2spk", banned_utts)
    removed_wav = filter_text_like(subset_dir / "wav.scp", banned_utts)
    removed_spk2utt = filter_spk2utt(subset_dir / "spk2utt", banned_utts)

    print(
        f"Removed {removed_text} entries from text, "
        f"{removed_utt2spk} from utt2spk, "
        f"{removed_wav} from wav.scp, and "
        f"{removed_spk2utt} utterance references from spk2utt."
    )
    return len(flagged)


def main():
    args = parse_args()
    base = Path(args.data_root)
    if not base.is_dir():
        raise SystemExit(f"Data root not found: {base}")

    total_flagged = 0
    for subset_name in args.subsets:
        subset_dir = base / subset_name
        if not subset_dir.is_dir():
            print(f"Skipping missing subset: {subset_dir}")
            continue
        total_flagged += process_subset(subset_dir, args.dry_run)

    print(f"Total flagged utterances: {total_flagged}")


if __name__ == "__main__":
    main()
