#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


BASE = Path(os.environ.get("SAP_PROCESSED_DIR", "/home/hojo/dataset/SAP0430_processed"))
MAX_DURATION = 29.5
MIN_DURATION = 0.2


def safe_symlink(target: str, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_symlink() and os.readlink(link) == target:
            return
        link.unlink()
    link.symlink_to(target)


def quote_tg(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def write_textgrid(path: Path, speaker: str, duration: float, segments):
    intervals = []
    cursor = 0.0
    for seg in sorted(segments, key=lambda x: (x["begin"], x["end"])):
        begin = max(0.0, min(seg["begin"], duration))
        end = max(begin, min(seg["end"], duration))
        if begin > cursor:
            intervals.append((cursor, begin, ""))
        intervals.append((begin, end, seg["text"]))
        cursor = end
    if cursor < duration:
        intervals.append((cursor, duration, ""))
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "0",
        f"{duration:.6f}",
        "<exists>",
        "1",
        '"IntervalTier"',
        quote_tg(speaker),
        "0",
        f"{duration:.6f}",
        str(len(intervals)),
    ]
    for begin, end, text in intervals:
        lines.extend([f"{begin:.6f}", f"{end:.6f}", quote_tg(text)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_segments(split: str):
    db = BASE / "mfa" / "temp" / f"segment_sentence_first_long_{split}" / split / f"{split}.db"
    if not db.exists():
        return {}, []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    query = """
        select
            u.begin as begin,
            u.end as end,
            trim(coalesce(u.normalized_text, u.text, '')) as text,
            f.name as file_name,
            sp.name as speaker,
            sf.sound_file_path as wav_path,
            sf.duration as file_duration
        from utterance u
        join file f on u.file_id = f.id
        join speaker sp on u.speaker_id = sp.id
        join sound_file sf on sf.file_id = f.id
        order by f.name, u.begin, u.end
    """
    by_file = defaultdict(list)
    skipped = []
    for row in conn.execute(query):
        duration = float(row["end"]) - float(row["begin"])
        text = (row["text"] or "").strip()
        reason = None
        if not text:
            reason = "empty_text"
        elif duration < MIN_DURATION:
            reason = "too_short"
        elif duration > MAX_DURATION:
            reason = "too_long"
        if reason:
            skipped.append(
                f"{row['file_name']}\t{row['speaker']}\t{row['begin']:.3f}\t{row['end']:.3f}\t{duration:.3f}\t{reason}\t{text}"
            )
            continue
        by_file[row["file_name"]].append(
            {
                "begin": float(row["begin"]),
                "end": float(row["end"]),
                "text": text,
                "speaker": row["speaker"],
                "wav_path": row["wav_path"],
                "file_duration": float(row["file_duration"]),
            }
        )
    conn.close()
    return by_file, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["dev", "train"])
    args = parser.parse_args()

    analysis = BASE / "mfa" / "analysis_sentence_first"
    corpus_out = BASE / "mfa" / "corpus_sentence_first_segmented_for_align"
    analysis.mkdir(parents=True, exist_ok=True)
    summary = []

    for split in args.splits:
        by_file, skipped = load_segments(split)
        manifest = []
        for rec, segments in sorted(by_file.items()):
            if not segments:
                continue
            spk = segments[0]["speaker"]
            wav_path = segments[0]["wav_path"]
            duration = segments[0]["file_duration"]
            spk_dir = corpus_out / split / spk
            safe_symlink(wav_path, spk_dir / f"{rec}.wav")
            write_textgrid(spk_dir / f"{rec}.TextGrid", spk, duration, segments)
            for index, seg in enumerate(segments, 1):
                new_utt = f"{rec}_mfasg{index:03d}"
                manifest.append(
                    f"{rec}\t{new_utt}\t{spk}\t{seg['begin']:.3f}\t{seg['end']:.3f}\t{seg['end']-seg['begin']:.3f}\t{seg['text'].upper()}"
                )
        (analysis / f"{split}_sentence_first_mfa_segment_segments.tsv").write_text(
            "clip_id\tnew_utt\tspk\tbegin\tend\tduration\ttext\n"
            + "\n".join(manifest)
            + ("\n" if manifest else ""),
            encoding="utf-8",
        )
        (analysis / f"{split}_sentence_first_mfa_segment_skipped.tsv").write_text(
            "clip_id\tspk\tbegin\tend\tduration\treason\ttext\n"
            + "\n".join(skipped)
            + ("\n" if skipped else ""),
            encoding="utf-8",
        )
        summary.append((split, len(by_file), len(manifest), len(skipped)))

    lines = ["split\tfiles_with_segments\tsegments\tskipped"]
    lines.extend("\t".join(map(str, row)) for row in summary)
    (analysis / "sentence_first_mfa_segment_summary.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
