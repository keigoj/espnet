#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Kaldi-style utt2phones files to phoneme alignment TSV files "
            "or filtered utt2phones files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input utt2phones file. If set, --output must also be set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file. If set, --input must also be set.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing <split>/utt2phones files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/phoneme_alignment"),
        help="Output directory for converted split files.",
    )
    parser.add_argument(
        "--output-format",
        choices=("tsv", "utt2phones"),
        default="tsv",
        help=(
            "Output format. tsv writes 'utt_id<TAB>p1,p2,...'; "
            "utt2phones writes 'utt_id p1 p2 ...'."
        ),
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep utterances with empty phone labels as empty entries.",
    )
    parser.add_argument(
        "splits",
        nargs="*",
        default=["train", "dev"],
        help="Splits to convert when --input/--output are not set.",
    )
    return parser.parse_args()


def format_record(utt_id: str, phones: list[str], output_format: str) -> str:
    if output_format == "tsv":
        return f"{utt_id}\t{','.join(phones)}\n"
    if len(phones) == 0:
        return f"{utt_id}\n"
    return f"{utt_id} {' '.join(phones)}\n"


def output_path_for_split(out_dir: Path, split: str, output_format: str) -> Path:
    if output_format == "tsv":
        return out_dir / f"{split}.tsv"
    return out_dir / f"{split}.utt2phones"


def convert_file(
    input_path: Path,
    output_path: Path,
    skip_empty: bool = True,
    output_format: str = "tsv",
) -> None:
    total = 0
    written = 0
    skipped_empty = 0
    seen_utts: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for lineno, line in enumerate(fin, start=1):
            parts = line.rstrip("\n").split()
            if len(parts) == 0:
                continue

            total += 1
            utt_id = parts[0]
            phones = parts[1:]
            if utt_id in seen_utts:
                raise ValueError(f"{input_path}:{lineno}: duplicate utterance id: {utt_id}")
            seen_utts.add(utt_id)

            if len(phones) == 0 and skip_empty:
                skipped_empty += 1
                continue

            fout.write(format_record(utt_id, phones, output_format))
            written += 1

    print(
        f"{input_path}: wrote {written}/{total} utterances to {output_path} "
        f"({skipped_empty} empty skipped)"
    )


def main() -> None:
    args = parse_args()
    skip_empty = not args.keep_empty

    if (args.input is None) != (args.output is None):
        raise ValueError("--input and --output must be set together")

    if args.input is not None:
        convert_file(
            args.input,
            args.output,
            skip_empty=skip_empty,
            output_format=args.output_format,
        )
        return

    for split in args.splits:
        convert_file(
            args.data_dir / split / "utt2phones",
            output_path_for_split(args.out_dir, split, args.output_format),
            skip_empty=skip_empty,
            output_format=args.output_format,
        )


if __name__ == "__main__":
    main()
