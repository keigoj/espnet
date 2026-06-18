#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


TEXT_FILES = {
    "origin": "{split}.origin.wrd",
    "with-parentheses": "{split}.wrd.with.parentheses",
    "without-parentheses": "{split}.wrd.without.parentheses",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Kaldi data directories from SAP0430_processed manifests."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="SAP0430_processed directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/kaldi.",
    )
    parser.add_argument(
        "--text-source",
        choices=sorted(TEXT_FILES),
        default="without-parentheses",
        help="Transcript file to use for Kaldi text.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=16000.0,
        help="Sample rate used to convert manifest sample counts to durations.",
    )
    parser.add_argument(
        "splits",
        nargs="*",
        default=["train", "dev"],
        help="Splits to convert.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def convert_split(root: Path, out_dir: Path, split: str, text_source: str, sample_rate: float) -> None:
    manifest_dir = root / "manifest"
    wav_dir = root / "data" / "processed" / split
    tsv_path = manifest_dir / f"{split}.tsv"
    text_path = manifest_dir / TEXT_FILES[text_source].format(split=split)

    tsv_lines = read_lines(tsv_path)
    text_lines = read_lines(text_path)
    entries = tsv_lines[1:]
    if len(entries) != len(text_lines):
        raise ValueError(
            f"{split}: line count mismatch: {tsv_path} has {len(entries)} entries, "
            f"{text_path} has {len(text_lines)} lines"
        )

    records: list[tuple[str, str, str, str, float]] = []
    seen_utts: set[str] = set()
    missing_wavs: list[Path] = []

    for entry, text in zip(entries, text_lines, strict=True):
        parts = entry.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{split}: malformed TSV line: {entry!r}")

        manifest_wav, sample_count = parts
        wav_path = wav_dir / Path(manifest_wav).name
        utt_id = wav_path.stem
        spk_id = utt_id.split("_", 1)[0]
        if utt_id in seen_utts:
            raise ValueError(f"{split}: duplicate utterance id: {utt_id}")
        seen_utts.add(utt_id)
        if not wav_path.exists():
            missing_wavs.append(wav_path)

        duration = int(sample_count) / sample_rate
        records.append((utt_id, spk_id, str(wav_path.resolve()), text.strip(), duration))

    if missing_wavs:
        preview = "\n".join(str(path) for path in missing_wavs[:10])
        raise FileNotFoundError(
            f"{split}: {len(missing_wavs)} wav files are missing. First missing files:\n{preview}"
        )

    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda record: record[0])

    spk_to_utts: dict[str, list[str]] = defaultdict(list)
    with (
        (split_out / "wav.scp").open("w", encoding="utf-8") as wav_scp,
        (split_out / "text").open("w", encoding="utf-8") as text_file,
        (split_out / "utt2spk").open("w", encoding="utf-8") as utt2spk,
        (split_out / "utt2dur").open("w", encoding="utf-8") as utt2dur,
    ):
        for utt_id, spk_id, wav_path, text, duration in records:
            wav_scp.write(f"{utt_id} {wav_path}\n")
            text_file.write(f"{utt_id} {text}\n")
            utt2spk.write(f"{utt_id} {spk_id}\n")
            utt2dur.write(f"{utt_id} {duration:.6f}\n")
            spk_to_utts[spk_id].append(utt_id)

    with (split_out / "spk2utt").open("w", encoding="utf-8") as spk2utt:
        for spk_id in sorted(spk_to_utts):
            spk2utt.write(f"{spk_id} {' '.join(spk_to_utts[spk_id])}\n")

    print(f"{split}: wrote {len(records)} utterances, {len(spk_to_utts)} speakers to {split_out}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "kaldi"
    for split in args.splits:
        convert_split(root, out_dir, split, args.text_source, args.sample_rate)


if __name__ == "__main__":
    main()
