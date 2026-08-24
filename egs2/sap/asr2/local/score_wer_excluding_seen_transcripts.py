#!/usr/bin/env python3
"""Recompute WER after removing references seen in train or dev."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
GENERATED_FILES = (
    "ref.trn",
    "hyp.trn",
    "result.txt",
    "excluded_utterances.tsv",
    "summary.txt",
)
SCORE_RE = re.compile(
    r"^Scores: \(#C #S #D #I\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TrnRecord:
    transcript: str
    identifier: str
    original_line: str


@dataclass(frozen=True)
class ScoreStats:
    sentences: int
    correct: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def reference_words(self) -> int:
        return self.correct + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            raise ValueError("Cannot calculate WER with zero reference words")
        return 100.0 * self.errors / self.reference_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter ref.trn/hyp.trn by exact train/dev transcript overlap and "
            "recompute WER with sclite. Whitespace is normalized for comparison; "
            "case and punctuation remain significant."
        )
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        required=True,
        help="Existing score_wer directory containing ref.trn and hyp.trn.",
    )
    parser.add_argument(
        "--train-text",
        type=Path,
        default=RECIPE_DIR / "data" / "train" / "text",
        help="Kaldi-format training text (default: data/train/text).",
    )
    parser.add_argument(
        "--dev-text",
        type=Path,
        default=RECIPE_DIR / "data" / "dev" / "text",
        help="Kaldi-format development text (default: data/dev/text).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory (default: score_wer_exclude_train_dev next to "
            "the input score directory)."
        ),
    )
    parser.add_argument(
        "--sclite",
        type=Path,
        help="Path to sclite (default: search PATH, then tools/sctk*/bin/sclite).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files previously generated in the output directory.",
    )
    return parser.parse_args()


def normalize_transcript(text: str) -> str:
    return " ".join(text.split())


def read_kaldi_transcripts(path: Path) -> set[str]:
    transcripts: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected '<utterance-id> <transcript>'"
                )
            transcripts.add(normalize_transcript(fields[1]))
    return transcripts


def read_trn(path: Path) -> list[TrnRecord]:
    records: list[TrnRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            original_line = line.rstrip("\n")
            try:
                transcript, identifier = original_line.rsplit("\t", 1)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: expected '<transcript>\\t(identifier)'"
                ) from error
            if not identifier.startswith("(") or not identifier.endswith(")"):
                raise ValueError(f"{path}:{line_number}: invalid TRN identifier")
            records.append(
                TrnRecord(
                    transcript=normalize_transcript(transcript),
                    identifier=identifier,
                    original_line=original_line,
                )
            )
    return records


def find_sclite(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        candidates = [explicit_path.expanduser()]
    else:
        candidates = []
        from_path = shutil.which("sclite")
        if from_path is not None:
            candidates.append(Path(from_path))
        candidates.extend(sorted(REPO_ROOT.glob("tools/sctk*/bin/sclite")))

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate.resolve()
    raise FileNotFoundError(
        "sclite was not found; install SCTK or specify it with --sclite"
    )


def ensure_output_is_safe(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / name
        for name in GENERATED_FILES
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {paths}. Use --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_trn(path: Path, records: list[TrnRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.original_line + "\n")


def run_sclite(sclite: Path, ref_path: Path, hyp_path: Path) -> str:
    command = [
        str(sclite),
        "-r",
        str(ref_path),
        "trn",
        "-h",
        str(hyp_path),
        "trn",
        "-i",
        "rm",
        "-o",
        "all",
        "stdout",
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def parse_score_stats(result: str) -> ScoreStats:
    scores = [tuple(map(int, match)) for match in SCORE_RE.findall(result)]
    if not scores:
        raise ValueError("No per-utterance scores were found in sclite output")
    return ScoreStats(
        sentences=len(scores),
        correct=sum(score[0] for score in scores),
        substitutions=sum(score[1] for score in scores),
        deletions=sum(score[2] for score in scores),
        insertions=sum(score[3] for score in scores),
    )


def format_summary(
    score_dir: Path,
    train_text: Path,
    dev_text: Path,
    total_sentences: int,
    total_reference_words: int,
    train_only: int,
    dev_only: int,
    train_and_dev: int,
    retained_stats: ScoreStats,
) -> str:
    excluded = train_only + dev_only + train_and_dev
    return "\n".join(
        [
            f"score_dir: {score_dir}",
            f"train_text: {train_text}",
            f"dev_text: {dev_text}",
            "comparison: exact after whitespace normalization (case/punctuation sensitive)",
            f"total_sentences: {total_sentences}",
            f"excluded_sentences: {excluded}",
            f"excluded_train_only: {train_only}",
            f"excluded_dev_only: {dev_only}",
            f"excluded_train_and_dev: {train_and_dev}",
            f"retained_sentences: {retained_stats.sentences}",
            f"total_reference_words: {total_reference_words}",
            f"retained_reference_words: {retained_stats.reference_words}",
            f"correct: {retained_stats.correct}",
            f"substitutions: {retained_stats.substitutions}",
            f"deletions: {retained_stats.deletions}",
            f"insertions: {retained_stats.insertions}",
            f"errors: {retained_stats.errors}",
            f"wer: {retained_stats.wer:.8f}",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    score_dir = args.score_dir.expanduser().resolve()
    train_text = args.train_text.expanduser().resolve()
    dev_text = args.dev_text.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else score_dir.with_name("score_wer_exclude_train_dev")
    )

    for path in (train_text, dev_text, score_dir / "ref.trn", score_dir / "hyp.trn"):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {path}")

    ensure_output_is_safe(output_dir, args.overwrite)
    train_transcripts = read_kaldi_transcripts(train_text)
    dev_transcripts = read_kaldi_transcripts(dev_text)
    references = read_trn(score_dir / "ref.trn")
    hypotheses = read_trn(score_dir / "hyp.trn")

    if len(references) != len(hypotheses):
        raise ValueError(
            f"ref/hyp length mismatch: {len(references)} != {len(hypotheses)}"
        )

    retained_refs: list[TrnRecord] = []
    retained_hyps: list[TrnRecord] = []
    excluded_rows: list[tuple[str, str, str]] = []
    train_only = 0
    dev_only = 0
    train_and_dev = 0

    for reference, hypothesis in zip(references, hypotheses):
        if reference.identifier != hypothesis.identifier:
            raise ValueError(
                "ref/hyp identifier mismatch: "
                f"{reference.identifier} != {hypothesis.identifier}"
            )
        in_train = reference.transcript in train_transcripts
        in_dev = reference.transcript in dev_transcripts
        if in_train or in_dev:
            if in_train and in_dev:
                source = "train+dev"
                train_and_dev += 1
            elif in_train:
                source = "train"
                train_only += 1
            else:
                source = "dev"
                dev_only += 1
            excluded_rows.append((source, reference.identifier, reference.transcript))
        else:
            retained_refs.append(reference)
            retained_hyps.append(hypothesis)

    if not retained_refs:
        raise ValueError("All utterances were excluded; WER cannot be computed")

    ref_output = output_dir / "ref.trn"
    hyp_output = output_dir / "hyp.trn"
    write_trn(ref_output, retained_refs)
    write_trn(hyp_output, retained_hyps)

    result = run_sclite(find_sclite(args.sclite), ref_output, hyp_output)
    (output_dir / "result.txt").write_text(result, encoding="utf-8")
    retained_stats = parse_score_stats(result)
    if retained_stats.sentences != len(retained_refs):
        raise ValueError(
            "sclite utterance count mismatch: "
            f"{retained_stats.sentences} != {len(retained_refs)}"
        )

    with (output_dir / "excluded_utterances.tsv").open("w", encoding="utf-8") as handle:
        handle.write("source\ttrn_identifier\ttranscript\n")
        for source, identifier, transcript in excluded_rows:
            handle.write(f"{source}\t{identifier}\t{transcript}\n")

    total_reference_words = sum(len(record.transcript.split()) for record in references)
    summary = format_summary(
        score_dir=score_dir,
        train_text=train_text,
        dev_text=dev_text,
        total_sentences=len(references),
        total_reference_words=total_reference_words,
        train_only=train_only,
        dev_only=dev_only,
        train_and_dev=train_and_dev,
        retained_stats=retained_stats,
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, FileExistsError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
