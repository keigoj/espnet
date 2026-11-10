#!/usr/bin/env python3
"""Select SAP mild reference speakers and prepare content-parallel SIR streams."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

INTELLIGIBILITY_NAMES = {
    "intelligibility",
    "intelligbility",
    "intelligibility (overall)",
    "intelligbility (overall)",
}
SCALE_SOURCE = (
    "https://publish.illinois.edu/hkim17/files/2024/12/Handout_ASHA2024-1.pdf#page=24"
)


def read_keyed(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[0] in result:
                raise ValueError(f"Invalid/duplicate keyed row at {path}:{number}")
            result[fields[0]] = fields[1]
    return result


def base_speaker(speaker: str) -> str:
    return re.sub(r"^sp[0-9]+(?:\.[0-9]+)?-", "", speaker)


def normalize_text(text: str) -> str:
    # Keep punctuation and word order. Do not equate merely similar prompts.
    text = " ".join(text.casefold().split())
    if any(
        token in {"<unk>", "[pii]", "[unintelligible]", "xx", "xxx"}
        for token in text.split()
    ):
        return ""
    return text


def speaker_score(path: Path) -> dict:
    if not path.is_file():
        return {
            "mean": None,
            "count": 0,
            "min": None,
            "max": None,
            "reason": "missing_doc",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = []
    for recording in data.get("Files", []):
        levels = []
        for rating in recording.get("Ratings", []):
            name = " ".join(rating.get("Dimension Description", "").casefold().split())
            if name not in INTELLIGIBILITY_NAMES:
                continue
            try:
                value = float(rating.get("Level", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 1 <= value <= 7:
                levels.append(value)
        if levels:
            # Equal weight per rated recording, even if it has multiple ratings.
            scores.append(statistics.fmean(levels))
    if not scores:
        return {
            "mean": None,
            "count": 0,
            "min": None,
            "max": None,
            "reason": "missing_intelligibility",
        }
    return {
        "mean": statistics.fmean(scores),
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "reason": "rated",
    }


def read_data(directory: Path, audio_root: Path) -> dict:
    if (directory / "segments").exists():
        raise ValueError(
            f"Use formatted utterance-level audio, without segments: {directory}"
        )
    data = {
        name: read_keyed(directory / name) for name in ("wav.scp", "text", "utt2spk")
    }
    keys = set(data["wav.scp"])
    if any(set(values) != keys for values in data.values()):
        raise ValueError(f"wav.scp, text, and utt2spk keys must match in {directory}")
    uniq = directory / "utt2uniq"
    data["utt2uniq"] = read_keyed(uniq) if uniq.exists() else {key: key for key in keys}
    if set(data["utt2uniq"]) != keys:
        raise ValueError(f"utt2uniq keys must match wav.scp in {directory}")
    for key, value in data["wav.scp"].items():
        if value.rstrip().endswith("|"):
            raise ValueError(f"Formatted sound paths are required: {key}")
        path = Path(value)
        path = (audio_root / path).resolve() if not path.is_absolute() else path
        if not path.is_file():
            raise FileNotFoundError(path)
        data["wav.scp"][key] = str(path)
    data["utt2spk"] = {
        key: base_speaker(value) for key, value in data["utt2spk"].items()
    }
    return data


def prepare(
    doc_dir: Path,
    train_dir: Path,
    valid_dir: Path,
    audio_root: Path,
    output_dir: Path,
    max_mean: float,
    seed: int = 0,
) -> dict:
    if not math.isfinite(max_mean) or not 1 <= max_mean <= 7:
        raise ValueError("max_mean must be a finite value in [1, 7]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite SIR artifacts: {output_dir}")
    datasets = {
        "train": read_data(train_dir, audio_root),
        "valid": read_data(valid_dir, audio_root),
    }
    train_speakers = set(datasets["train"]["utt2spk"].values())
    valid_speakers = set(datasets["valid"]["utt2spk"].values())
    if train_speakers & valid_speakers:
        raise ValueError("Train and validation speakers must be disjoint")
    scores = {
        speaker: speaker_score(doc_dir / f"{speaker}.json")
        for speaker in sorted(train_speakers | valid_speakers)
    }
    mild = {
        speaker
        for speaker in train_speakers
        if scores[speaker]["mean"] is not None and scores[speaker]["mean"] <= max_mean
    }
    if not mild:
        raise ValueError("No rated mild training speakers satisfy the threshold")

    # Index only unperturbed training utterances, grouped by text and speaker.
    candidates = defaultdict(lambda: defaultdict(list))
    train = datasets["train"]
    for utt in sorted(train["wav.scp"]):
        speaker = train["utt2spk"][utt]
        content = normalize_text(train["text"][utt])
        if speaker in mild and utt == train["utt2uniq"][utt] and content:
            candidates[content][speaker].append(utt)
    if not candidates:
        raise ValueError("No unperturbed mild training references are available")
    # The placeholder is loaded for mask-0 rows but never encoded. Use a short
    # real training waveform to keep this loader overhead small.
    sample_file = train_dir / "utt2num_samples"
    samples = read_keyed(sample_file) if sample_file.exists() else {}
    placeholder = min(train["wav.scp"], key=lambda u: (int(samples.get(u, 0)), u))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mild_speakers.txt").open("w", encoding="utf-8") as stream:
        stream.write("".join(f"{speaker}\n" for speaker in sorted(mild)))
    with (output_dir / "speaker_scores.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "speaker_id",
                "split",
                "mean_intelligibility",
                "rated_recordings",
                "min_intelligibility",
                "max_intelligibility",
                "mild_reference",
                "status",
                "doc_path",
            ]
        )
        for speaker, score in scores.items():
            writer.writerow(
                [
                    speaker,
                    "train" if speaker in train_speakers else "valid",
                    score["mean"],
                    score["count"],
                    score["min"],
                    score["max"],
                    int(speaker in mild),
                    score["reason"],
                    str((doc_dir / f"{speaker}.json").resolve()),
                ]
            )

    summary = {
        "doc_dir": str(doc_dir.resolve()),
        "train_dir": str(train_dir.resolve()),
        "valid_dir": str(valid_dir.resolve()),
        "max_mean_intelligibility": max_mean,
        "seed": seed,
        "scale_source": SCALE_SOURCE,
        "selection": "mean of per-recording intelligibility ratings <= threshold",
        "source_policy": "rated speakers above threshold; missing and mild sources use ASR only",
        "reference_policy": "training only, unperturbed; uniform speaker then uniform recording",
        "pairing": "casefold and whitespace normalization of ASR transcripts; fixed per original utterance",
        "train_speakers": len(train_speakers),
        "mild_reference_speakers": len(mild),
        "unrated_train_speakers": sum(
            scores[s]["mean"] is None for s in train_speakers
        ),
        "placeholder_utterance": placeholder,
        "splits": {},
    }
    for split, data in datasets.items():
        destination = output_dir / split
        destination.mkdir()
        counts = defaultdict(int)
        files = [
            destination / name for name in ("healthy.scp", "sir_pair_mask", "pairs.tsv")
        ]
        with files[0].open("w", encoding="utf-8") as scp, files[1].open(
            "w", encoding="utf-8"
        ) as mask, files[2].open("w", encoding="utf-8", newline="") as pairs:
            writer = csv.writer(pairs, delimiter="\t")
            writer.writerow(
                [
                    "utterance_id",
                    "reference_id",
                    "source_speaker",
                    "reference_speaker",
                    "source_score",
                    "reference_score",
                    "content_sha256",
                    "status",
                ]
            )
            for utt in sorted(data["wav.scp"]):
                source = data["utt2spk"][utt]
                score = scores[source]["mean"]
                content = normalize_text(data["text"][utt])
                pool = candidates.get(content, {})
                reference = None
                if score is None:
                    status = "unrated_source"
                elif score <= max_mean:
                    status = "mild_source"
                elif not content or not pool:
                    status = "no_content_match"
                else:
                    rng = random.Random(f"{seed}:{data['utt2uniq'][utt]}")
                    ref_speaker = rng.choice(sorted(pool))
                    reference = rng.choice(pool[ref_speaker])
                    status = "paired"
                counts[status] += 1
                scp.write(f"{utt} {train['wav.scp'][reference or placeholder]}\n")
                mask.write(f"{utt} {int(reference is not None)}\n")
                writer.writerow(
                    [
                        utt,
                        reference or "",
                        source,
                        train["utt2spk"][reference] if reference else "",
                        score,
                        (
                            scores[train["utt2spk"][reference]]["mean"]
                            if reference
                            else ""
                        ),
                        (
                            hashlib.sha256(content.encode()).hexdigest()
                            if reference
                            else ""
                        ),
                        status,
                    ]
                )
        summary["splits"][split] = {"utterances": len(data["wav.scp"]), **dict(counts)}
    if not summary["splits"]["train"].get("paired", 0):
        raise ValueError("No SIR training pairs were found; inspect speaker_scores.tsv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-mean-intelligibility", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = prepare(
        args.doc_dir,
        args.train_dir,
        args.valid_dir,
        args.audio_root,
        args.output_dir,
        args.max_mean_intelligibility,
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
