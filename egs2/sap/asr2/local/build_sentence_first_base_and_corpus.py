#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import argparse
import os
import wave
from collections import defaultdict
from pathlib import Path


BASE = Path(os.environ.get("SAP_PROCESSED_DIR", "/home/hojo/dataset/SAP0430_processed"))
MAX_DURATION = 29.5
MIN_DURATION = 0.1

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sentence_builder", SCRIPT_DIR / "build_sentence_kaldi_from_mfa.py"
)
sentence_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentence_builder)


def safe_symlink(target: str, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_symlink() and os.readlink(link) == target:
            return
        link.unlink()
    link.symlink_to(target)


def write_wav_clip(src: str, dst: Path, start: float, end: float):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with wave.open(src, "rb") as inp:
        params = inp.getparams()
        sample_rate = inp.getframerate()
        start_frame = max(0, int(round(start * sample_rate)))
        end_frame = min(inp.getnframes(), int(round(end * sample_rate)))
        inp.setpos(start_frame)
        frames = inp.readframes(max(0, end_frame - start_frame))
    with wave.open(str(dst), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)


def write_kaldi(root: Path, split: str, rows, wavs):
    out = root / split
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x[0])
    spk2utt = defaultdict(list)
    for utt, _rec, _start, _end, _text, spk, _source in rows:
        spk2utt[spk].append(utt)
    with (out / "wav.scp").open("w", encoding="utf-8") as f:
        for rec, path in sorted(wavs.items()):
            f.write(f"{rec} {path}\n")
    with (out / "segments").open("w", encoding="utf-8") as f:
        for utt, rec, start, end, _text, _spk, _source in rows:
            f.write(f"{utt} {rec} {start:.3f} {end:.3f}\n")
    with (out / "text").open("w", encoding="utf-8") as f:
        for utt, _rec, _start, _end, text, _spk, _source in rows:
            f.write(f"{utt} {text}\n")
    with (out / "utt2spk").open("w", encoding="utf-8") as f:
        for utt, _rec, _start, _end, _text, spk, _source in rows:
            f.write(f"{utt} {spk}\n")
    with (out / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["dev", "train"])
    args = parser.parse_args()

    analysis = BASE / "mfa" / "analysis_sentence_first"
    analysis.mkdir(parents=True, exist_ok=True)
    base_root = BASE / "kaldi_sentence_first_base"
    segment_corpus = BASE / "mfa" / "corpus_sentence_first_long_for_segment"
    clip_root = BASE / "mfa" / "audio_sentence_first_long"

    summary = []
    for split in args.splits:
        manifest = sentence_builder.read_manifest(split)
        wav, utt2spk = sentence_builder.read_kaldi_aux(split)
        aligned_dir = BASE / "mfa" / "aligned" / f"{split}_textgrid"
        aligned = {p.stem: p for p in aligned_dir.rglob("*.TextGrid")}

        rows = []
        wavs = {}
        sentence_manifest = []
        long_for_segment = []
        excluded = []
        source_counts = defaultdict(int)

        for utt in sorted(manifest):
            if utt not in wav or utt not in utt2spk:
                excluded.append(f"{utt}\tmissing_aux\t0\t")
                continue
            duration = sentence_builder.wav_duration(wav[utt])
            spk = utt2spk[utt]

            if utt in aligned:
                wavs[utt] = wav[utt]
                origin_sentences = [
                    s
                    for s in sentence_builder.split_origin_sentences(manifest[utt]["origin"])
                    if sentence_builder.simple_tokens(s)
                ]
                if len(origin_sentences) <= 1 and duration <= 30.0:
                    rows.append((utt, utt, 0.0, duration, manifest[utt]["normalized"], spk, "aligned"))
                    source_counts["aligned_kept"] += 1
                    continue
                segments = sentence_builder.build_segments_for_utt(
                    utt,
                    manifest[utt]["origin"],
                    manifest[utt]["normalized"],
                    duration,
                    aligned[utt],
                )
                if not segments:
                    excluded.append(
                        f"{utt}\tno_sentence_segments\t{duration:.3f}\t{manifest[utt]['normalized']}"
                    )
                    continue
                if len(segments) == 1:
                    start, end, text = segments[0]
                    new_utt = utt if start <= 0.1 and duration - end <= 0.1 else f"{utt}_sent001"
                    rows.append((new_utt, utt, start, end, text, spk, "aligned_sent"))
                    source_counts["aligned_sentence"] += 1
                else:
                    source_counts["aligned_sentence_orig"] += 1
                    source_counts["aligned_sentence_segments"] += len(segments)
                    for index, (start, end, text) in enumerate(segments, 1):
                        new_utt = f"{utt}_sent{index:03d}"
                        rows.append((new_utt, utt, start, end, text, spk, "aligned_sent"))
                        sentence_manifest.append(
                            f"{utt}\t{new_utt}\t{spk}\t{duration:.3f}\t{start:.3f}\t{end:.3f}\taligned\t{text}"
                        )
                continue

            target_tokens = manifest[utt]["normalized"].split()
            token_spans = sentence_builder.sentence_token_spans(
                manifest[utt]["origin"], manifest[utt]["normalized"]
            )
            if not target_tokens or not token_spans:
                excluded.append(f"{utt}\tno_tokens\t{duration:.3f}\t{manifest[utt]['normalized']}")
                continue
            n_tokens = len(target_tokens)
            for index, (start_tok, end_tok) in enumerate(token_spans, 1):
                start_tok = max(0, min(start_tok, n_tokens))
                end_tok = max(start_tok, min(end_tok, n_tokens))
                if start_tok >= end_tok:
                    continue
                start = duration * start_tok / n_tokens
                end = duration * end_tok / n_tokens
                if end - start < MIN_DURATION:
                    continue
                text = " ".join(target_tokens[start_tok:end_tok])
                if end - start <= 30.0:
                    new_utt = utt if len(token_spans) == 1 else f"{utt}_sent{index:03d}"
                    rows.append((new_utt, utt, start, end, text, spk, "unaligned_short_or_sentence"))
                    wavs[utt] = wav[utt]
                    source_counts["unaligned_sentence_kept"] += 1
                    sentence_manifest.append(
                        f"{utt}\t{new_utt}\t{spk}\t{duration:.3f}\t{start:.3f}\t{end:.3f}\tunaligned_empty\t{text}"
                    )
                else:
                    clip_id = f"{utt}_sent{index:03d}"
                    clip_path = clip_root / split / spk / f"{clip_id}.wav"
                    write_wav_clip(wav[utt], clip_path, start, end)
                    spk_dir = segment_corpus / split / spk
                    safe_symlink(str(clip_path), spk_dir / f"{clip_id}.wav")
                    (spk_dir / f"{clip_id}.lab").write_text(text + "\n", encoding="utf-8")
                    long_for_segment.append(
                        f"{utt}\t{clip_id}\t{spk}\t{duration:.3f}\t{start:.3f}\t{end:.3f}\t{end-start:.3f}\t{text}"
                    )
                    source_counts["sentence_long_for_mfa_segment"] += 1

        write_kaldi(base_root, split, rows, wavs)
        (analysis / f"{split}_base_sentence_manifest.tsv").write_text(
            "orig_utt\tnew_utt\tspk\torig_duration\tstart\tend\tsource\ttext\n"
            + "\n".join(sentence_manifest)
            + ("\n" if sentence_manifest else ""),
            encoding="utf-8",
        )
        (analysis / f"{split}_long_sentences_for_mfa_segment.tsv").write_text(
            "orig_utt\tclip_id\tspk\torig_duration\tstart\tend\tduration\ttext\n"
            + "\n".join(long_for_segment)
            + ("\n" if long_for_segment else ""),
            encoding="utf-8",
        )
        (analysis / f"{split}_excluded.tsv").write_text(
            "utt\treason\tduration\ttext\n" + "\n".join(excluded) + ("\n" if excluded else ""),
            encoding="utf-8",
        )
        summary.append((split, len(rows), len(long_for_segment), len(excluded), dict(source_counts)))

    lines = ["split\tbase_utts\tlong_sentences_for_mfa_segment\texcluded\tsource_counts"]
    lines.extend("\t".join([row[0], str(row[1]), str(row[2]), str(row[3]), str(row[4])]) for row in summary)
    (analysis / "base_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
