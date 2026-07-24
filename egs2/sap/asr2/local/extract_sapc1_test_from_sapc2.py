#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


DEFAULT_SAPC2_ROOT = Path("/home/hojo/dataset/SAPC2-train-dev")
DEFAULT_OUTPUT_ROOT = Path("/home/hojo/dataset/SAP0430_processed")
SAPC2_MANIFEST_SPLITS = ("Train", "Dev")
TARGET_SPLITS = ("test1", "test2")

ONES = [
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
]
TENS = {
    20: "TWENTY",
    30: "THIRTY",
    40: "FORTY",
    50: "FIFTY",
    60: "SIXTY",
    70: "SEVENTY",
    80: "EIGHTY",
    90: "NINETY",
}
ORDINALS = {
    1: "FIRST",
    2: "SECOND",
    3: "THIRD",
    4: "FOURTH",
    5: "FIFTH",
    6: "SIXTH",
    7: "SEVENTH",
    8: "EIGHTH",
    9: "NINTH",
    10: "TENTH",
    11: "ELEVENTH",
    12: "TWELFTH",
    13: "THIRTEENTH",
    14: "FOURTEENTH",
    15: "FIFTEENTH",
    16: "SIXTEENTH",
    17: "SEVENTEENTH",
    18: "EIGHTEENTH",
    19: "NINETEENTH",
    20: "TWENTIETH",
    30: "THIRTIETH",
    40: "FORTIETH",
    50: "FIFTIETH",
    60: "SIXTIETH",
    70: "SEVENTIETH",
    80: "EIGHTIETH",
    90: "NINETIETH",
}
DIGITS = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SAPC1 test1/test2 wavs from the SAPC2 train/dev distribution "
            "and create matching SAP0430_processed manifests."
        )
    )
    parser.add_argument(
        "--sapc1-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "SAPC1",
        help="Directory containing SAPC1 test1.tsv and test2.tsv.",
    )
    parser.add_argument(
        "--sapc2-root",
        type=Path,
        default=DEFAULT_SAPC2_ROOT,
        help="SAPC2 train/dev root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="SAP0430_processed root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report planned writes without copying files.",
    )
    return parser.parse_args()


def wav_id_from_manifest_line(line: str) -> str | None:
    if ".wav" not in line:
        return None
    return Path(line.split()[0]).stem


def read_sapc2_records(sapc2_root: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for split in SAPC2_MANIFEST_SPLITS:
        manifest_path = sapc2_root / "manifest" / f"{split}.jsonl"
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records[record["id"]] = record
    return records


def read_target_entries(sapc1_dir: Path, split: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with (sapc1_dir / f"{split}.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            wav_id = wav_id_from_manifest_line(line)
            if wav_id is not None:
                fields = line.split()
                if len(fields) < 2:
                    raise ValueError(f"{split}: missing sample count in line: {line!r}")
                entries.append((wav_id, fields[1]))
    return entries


def write_lines(path: Path, lines: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_speaker_json(sapc2_root: Path, record: dict) -> Path:
    audio_relpath = Path(record["audio_filepath"])
    return sapc2_root / audio_relpath.parent / f"{record['speaker']}.json"


def number_to_words(number: int) -> str:
    if number < 0:
        return "MINUS " + number_to_words(abs(number))
    if number < 20:
        return ONES[number]
    if number < 100:
        tens = (number // 10) * 10
        remainder = number % 10
        return TENS[tens] if remainder == 0 else f"{TENS[tens]} {ONES[remainder]}"
    if number < 1000:
        hundreds = number // 100
        remainder = number % 100
        words = f"{ONES[hundreds]} HUNDRED"
        return words if remainder == 0 else f"{words} {number_to_words(remainder)}"
    if 1000 < number < 2000 and number % 100 == 0:
        return f"{number_to_words(number // 100)} HUNDRED"
    if 1900 <= number <= 1999:
        remainder = number % 100
        return "NINETEEN HUNDRED" if remainder == 0 else f"NINETEEN {number_to_words(remainder)}"
    if 2000 <= number <= 2099:
        remainder = number % 100
        return "TWO THOUSAND" if remainder == 0 else f"TWO THOUSAND {number_to_words(remainder)}"
    if number < 1000000:
        thousands = number // 1000
        remainder = number % 1000
        words = f"{number_to_words(thousands)} THOUSAND"
        return words if remainder == 0 else f"{words} {number_to_words(remainder)}"
    millions = number // 1000000
    remainder = number % 1000000
    words = f"{number_to_words(millions)} MILLION"
    return words if remainder == 0 else f"{words} {number_to_words(remainder)}"


def ordinal_to_words(number: int) -> str:
    if number in ORDINALS:
        return ORDINALS[number]
    if number < 100:
        tens = (number // 10) * 10
        remainder = number % 10
        return f"{TENS[tens]} {ORDINALS[remainder]}"
    if number < 1000:
        hundreds = number // 100
        remainder = number % 100
        prefix = f"{ONES[hundreds]} HUNDRED"
        return prefix + ("TH" if remainder == 0 else f" {ordinal_to_words(remainder)}")
    return number_to_words(number)


def digit_sequence_to_words(sequence: str) -> str:
    return " ".join(DIGITS[char] for char in sequence if char.isdigit())


def normalize_text_for_wrd(original_text: str) -> str:
    text = original_text
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = text.replace("&", " and ")

    def replace_area_code(match: re.Match[str]) -> str:
        return f"{match.group(1)} {digit_sequence_to_words(match.group(2))}"

    text = re.sub(
        r"(?i)\b(area code\s+)([0-9][0-9\-\s]{5,}[0-9])\b",
        replace_area_code,
        text,
    )

    def replace_order_number(match: re.Match[str]) -> str:
        return f"{match.group(1)} {digit_sequence_to_words(match.group(2))}"

    text = re.sub(r"(?i)\b(order number\s+)([0-9][0-9,\-\s]*[0-9])\b", replace_order_number, text)

    def replace_currency(match: re.Match[str]) -> str:
        symbol = match.group(1)
        amount = int(match.group(2).replace(",", ""))
        unit = {
            "$": "DOLLAR" if amount == 1 else "DOLLARS",
            "€": "EURO" if amount == 1 else "EUROS",
            "£": "POUND" if amount == 1 else "POUNDS",
        }[symbol]
        return f"{number_to_words(amount)} {unit}"

    text = re.sub(r"([$€£])\s*([0-9][0-9,]*)", replace_currency, text)

    def replace_percent(match: re.Match[str]) -> str:
        return f"{number_to_words(int(match.group(1).replace(',', '')))} PERCENT"

    text = re.sub(r"\b([0-9][0-9,]*)\s*%", replace_percent, text)

    def replace_time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        minute_words = "O " + ONES[minute] if 0 < minute < 10 else number_to_words(minute)
        suffix = match.group(3)
        suffix_words = f" {suffix[0].upper()} M" if suffix else ""
        return f"{number_to_words(hour)} {minute_words}{suffix_words}"

    text = re.sub(r"\b([0-9]{1,2}):([0-9]{2})\s*([AaPp][Mm])?\b", replace_time, text)

    def replace_attached_ampm(match: re.Match[str]) -> str:
        return f"{number_to_words(int(match.group(1)))} {match.group(2)[0].upper()} M"

    text = re.sub(r"\b([0-9]{1,2})\s*([AaPp][Mm])\b", replace_attached_ampm, text)

    def replace_ordinal(match: re.Match[str]) -> str:
        return ordinal_to_words(int(match.group(1)))

    text = re.sub(r"\b([0-9]+)(st|nd|rd|th)\b", replace_ordinal, text, flags=re.IGNORECASE)

    def replace_number(match: re.Match[str]) -> str:
        return number_to_words(int(match.group(0).replace(",", "")))

    text = re.sub(r"\b[0-9][0-9,]*\b", replace_number, text)
    text = re.sub(r"\b([AP])\.?M\.?\b", lambda match: f"{match.group(1)} M", text)
    text = text.replace("'", "'")
    text = re.sub(r"[^A-Za-z']+", " ", text.upper())
    return " ".join(text.split())


def normalize_without_parentheses(record: dict) -> str:
    text = record.get("text", "")
    if any(char.isdigit() for char in text) or any(symbol in text for symbol in "$€£%"):
        return normalize_text_for_wrd(record.get("original_text", ""))
    return text.upper()


def extract_split(
    split: str,
    entries: list[tuple[str, str]],
    records: dict[str, dict],
    sapc2_root: Path,
    output_root: Path,
    dry_run: bool,
) -> tuple[int, int]:
    wav_dir = output_root / "data" / "processed" / split
    doc_dir = output_root / "data" / "doc"
    manifest_dir = output_root / "manifest"

    missing_records = [wav_id for wav_id, _sample_count in entries if wav_id not in records]
    if missing_records:
        preview = "\n".join(missing_records[:10])
        raise FileNotFoundError(
            f"{split}: {len(missing_records)} IDs are missing from SAPC2 manifests. "
            f"First missing IDs:\n{preview}"
        )

    missing_wavs: list[Path] = []
    missing_jsons: list[Path] = []
    for wav_id, _sample_count in entries:
        record = records[wav_id]
        wav_path = sapc2_root / record["audio_filepath"]
        json_path = source_speaker_json(sapc2_root, record)
        if not wav_path.exists():
            missing_wavs.append(wav_path)
        if not json_path.exists():
            missing_jsons.append(json_path)
    if missing_wavs:
        preview = "\n".join(str(path) for path in missing_wavs[:10])
        raise FileNotFoundError(
            f"{split}: {len(missing_wavs)} wav files are missing. First missing files:\n{preview}"
        )
    if missing_jsons:
        preview = "\n".join(str(path) for path in missing_jsons[:10])
        raise FileNotFoundError(
            f"{split}: {len(missing_jsons)} speaker JSON files are missing. "
            f"First missing files:\n{preview}"
        )

    if not dry_run:
        wav_dir.mkdir(parents=True, exist_ok=True)
        doc_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)

    copied_speaker_jsons: set[str] = set()
    tsv_lines = [f"/projects/bczs/SAPC/manifest/{split}"]
    origin_lines: list[str] = []
    with_parentheses_lines: list[str] = []
    without_parentheses_lines: list[str] = []
    transcript_rows = [
        "id\tspeaker\tetiology\taudio_filepath\tduration\toriginal_text\t"
        "norm_text_with_disfluency\ttext"
    ]

    for wav_id, sample_count in entries:
        record = records[wav_id]
        source_wav = sapc2_root / record["audio_filepath"]
        target_wav = wav_dir / f"{wav_id}.wav"
        if not dry_run:
            shutil.copy2(source_wav, target_wav)

        speaker = record["speaker"]
        if speaker not in copied_speaker_jsons:
            copied_speaker_jsons.add(speaker)
            if not dry_run:
                shutil.copy2(source_speaker_json(sapc2_root, record), doc_dir / f"{speaker}.json")

        tsv_lines.append(f"/projects/bczs/SAPC/data/processed/{split}/{wav_id}.wav\t{sample_count}")
        origin_lines.append(record.get("original_text", ""))
        with_parentheses_lines.append(record.get("norm_text_with_disfluency", "").upper())
        without_parentheses_lines.append(normalize_without_parentheses(record))
        transcript_rows.append(
            "\t".join(
                [
                    wav_id,
                    speaker,
                    record.get("etiology", ""),
                    f"data/processed/{split}/{wav_id}.wav",
                    str(record.get("duration", "")),
                    record.get("original_text", ""),
                    record.get("norm_text_with_disfluency", ""),
                    record.get("text", ""),
                ]
            )
        )

    write_lines(manifest_dir / f"{split}.tsv", tsv_lines, dry_run)
    write_lines(manifest_dir / f"{split}.origin.wrd", origin_lines, dry_run)
    write_lines(manifest_dir / f"{split}.wrd.with.parentheses", with_parentheses_lines, dry_run)
    write_lines(manifest_dir / f"{split}.wrd.without.parentheses", without_parentheses_lines, dry_run)
    write_lines(manifest_dir / f"{split}.transcripts.tsv", transcript_rows, dry_run)

    return len(entries), len(copied_speaker_jsons)


def main() -> None:
    args = parse_args()
    sapc1_dir = args.sapc1_dir.resolve()
    sapc2_root = args.sapc2_root.resolve()
    output_root = args.output_root.resolve()

    records = read_sapc2_records(sapc2_root)
    for split in TARGET_SPLITS:
        entries = read_target_entries(sapc1_dir, split)
        wav_count, speaker_count = extract_split(
            split=split,
            entries=entries,
            records=records,
            sapc2_root=sapc2_root,
            output_root=output_root,
            dry_run=args.dry_run,
        )
        action = "would write" if args.dry_run else "wrote"
        print(f"{split}: {action} {wav_count} wavs and {speaker_count} speaker JSON files")


if __name__ == "__main__":
    main()
