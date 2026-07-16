#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
import wave
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


BASE = Path(os.environ.get("SAP_PROCESSED_DIR", "/home/hojo/dataset/SAP0430_processed"))
MAX_DURATION = 29.5
PAD = 0.05
MIN_DURATION = 0.1


def read_manifest(split: str):
    manifest_dir = BASE / "manifest"
    ids = [
        Path(line.split("\t", 1)[0]).stem
        for line in (manifest_dir / f"{split}.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    origins = (manifest_dir / f"{split}.origin.wrd").read_text(encoding="utf-8").splitlines()
    normalized = (
        manifest_dir / f"{split}.wrd.without.parentheses"
    ).read_text(encoding="utf-8").splitlines()
    if not (len(ids) == len(origins) == len(normalized)):
        raise ValueError(f"Manifest length mismatch for {split}")
    return {
        utt: {"origin": origin, "normalized": norm}
        for utt, origin, norm in zip(ids, origins, normalized)
    }


def read_kaldi_aux(split: str):
    kaldi_dir = BASE / "kaldi" / split
    wav = {}
    utt2spk = {}
    with (kaldi_dir / "wav.scp").open(encoding="utf-8") as f:
        for line in f:
            utt, path = line.rstrip("\n").split(maxsplit=1)
            wav[utt] = path
    with (kaldi_dir / "utt2spk").open(encoding="utf-8") as f:
        for line in f:
            utt, spk = line.rstrip("\n").split(maxsplit=1)
            utt2spk[utt] = spk
    return wav, utt2spk


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as f:
        return f.getnframes() / f.getframerate()


def parse_short_textgrid_words(path: Path):
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    for i, line in enumerate(lines):
        if line != '"IntervalTier"':
            continue
        if i + 4 >= len(lines) or lines[i + 1].strip('"') != "words":
            continue
        n = int(lines[i + 4])
        j = i + 5
        words = []
        for _ in range(n):
            begin = float(lines[j])
            end = float(lines[j + 1])
            label = lines[j + 2].strip('"')
            if label:
                words.append((begin, end, label))
            j += 3
        return words
    return []


def remove_prompt_and_parentheses(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return text


def split_origin_sentences(origin: str):
    text = remove_prompt_and_parentheses(origin)
    pieces = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:[\"']+)?", text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [text.strip()]


def simple_tokens(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text.upper())


def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sentence_token_spans(origin: str, normalized_text: str):
    target = normalized_text.split()
    n = len(target)
    raw_sentences = split_origin_sentences(origin)
    sent_tokens = [simple_tokens(s) for s in raw_sentences]
    sent_tokens = [x for x in sent_tokens if x]
    if not target:
        return []
    if len(sent_tokens) <= 1:
        return [(0, n)]

    total_src = sum(len(x) for x in sent_tokens) or len(target)
    if n > 80 or len(sent_tokens) > 6:
        spans = []
        start = 0
        cumulative = 0
        for i, src in enumerate(sent_tokens):
            cumulative += len(src)
            if i == len(sent_tokens) - 1:
                end = n
            else:
                end = round(n * cumulative / total_src)
                end = max(start + 1, min(end, n - (len(sent_tokens) - i - 1)))
            spans.append((start, end))
            start = end
        return spans

    @lru_cache(maxsize=None)
    def cost(i: int, start: int, end: int) -> float:
        src = sent_tokens[i]
        tgt = target[start:end]
        dist = edit_distance(src, tgt)
        denom = max(len(src), len(tgt), 1)
        return dist / denom + 0.02 * abs(len(src) - len(tgt))

    states = {0: (0.0, [])}
    m = len(sent_tokens)
    for i, src in enumerate(sent_tokens):
        new_states = {}
        remaining_src = sum(len(x) for x in sent_tokens[i:]) or 1
        remaining_sentences = m - i - 1
        for start, (state_cost, path) in states.items():
            remaining_target = n - start
            if i == m - 1:
                candidates = [n]
            else:
                expected = max(1, round(remaining_target * len(src) / remaining_src))
                slack = max(8, math.ceil(expected * 0.75))
                min_len = max(1, expected - slack)
                max_len = min(remaining_target - remaining_sentences, expected + slack)
                candidates = range(start + min_len, start + max_len + 1)
            for end in candidates:
                if end <= start or end > n:
                    continue
                c = state_cost + cost(i, start, end)
                old = new_states.get(end)
                if old is None or c < old[0]:
                    new_states[end] = (c, path + [(start, end)])
        states = new_states
        if not states:
            return [(0, n)]
    return states.get(n, (0.0, [(0, n)]))[1]


def proportional_index(index: int, source_len: int, target_len: int):
    if source_len == target_len:
        return index
    return round(index * target_len / max(source_len, 1))


def build_segments_for_utt(utt: str, origin: str, normalized_text: str, duration: float, tg_path: Path):
    target_tokens = normalized_text.split()
    words = parse_short_textgrid_words(tg_path)
    if not words or not target_tokens:
        return []

    token_spans = sentence_token_spans(origin, normalized_text)
    output = []
    for start_tok, end_tok in token_spans:
        start_tok = max(0, min(start_tok, len(target_tokens)))
        end_tok = max(start_tok, min(end_tok, len(target_tokens)))
        if start_tok >= end_tok:
            continue
        start_word = max(0, min(len(words) - 1, proportional_index(start_tok, len(target_tokens), len(words))))
        end_word = max(start_word + 1, min(len(words), proportional_index(end_tok, len(target_tokens), len(words))))
        text_tokens = target_tokens[start_tok:end_tok]

        chunk_start_word = start_word
        chunk_text_tokens = []
        current_start = max(0.0, words[chunk_start_word][0] - PAD)
        for word_index in range(start_word, end_word):
            candidate_end = min(duration, words[word_index][1] + PAD)
            token_index = proportional_index(word_index, len(words), len(target_tokens))
            token = target_tokens[min(max(token_index, start_tok), end_tok - 1)]
            if chunk_text_tokens and candidate_end - current_start > MAX_DURATION:
                end = min(duration, words[word_index - 1][1] + PAD)
                output.append((current_start, end, " ".join(chunk_text_tokens)))
                chunk_start_word = word_index
                current_start = max(0.0, words[chunk_start_word][0] - PAD)
                chunk_text_tokens = []
            chunk_text_tokens.append(token)
        if chunk_text_tokens:
            end = min(duration, words[end_word - 1][1] + PAD)
            if end - current_start > MAX_DURATION:
                end = current_start + MAX_DURATION
            output.append((current_start, end, " ".join(chunk_text_tokens)))

    cleaned = []
    seen = set()
    for start, end, text in output:
        if end - start < MIN_DURATION or not text:
            continue
        key = (round(start, 3), round(end, 3), text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((start, end, text))
    return cleaned


def read_rescue_rows(split: str):
    path = BASE / "mfa" / "analysis" / f"{split}_mfa_segment_rescue_segments.tsv"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        orig, new_utt, spk, begin, end, _dur, text = line.split("\t", 6)
        rows.append((new_utt, orig, float(begin), float(end), text, spk))
    return rows


def write_kaldi(root: Path, split: str, rows, wavs):
    out = root / split
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x[0])
    spk2utt = defaultdict(list)
    for utt, _rec, _start, _end, _text, spk in rows:
        spk2utt[spk].append(utt)
    with (out / "wav.scp").open("w", encoding="utf-8") as f:
        for rec, path in sorted(wavs.items()):
            f.write(f"{rec} {path}\n")
    with (out / "segments").open("w", encoding="utf-8") as f:
        for utt, rec, start, end, _text, _spk in rows:
            f.write(f"{utt} {rec} {start:.3f} {end:.3f}\n")
    with (out / "text").open("w", encoding="utf-8") as f:
        for utt, _rec, _start, _end, text, _spk in rows:
            f.write(f"{utt} {text}\n")
    with (out / "utt2spk").open("w", encoding="utf-8") as f:
        for utt, _rec, _start, _end, _text, spk in rows:
            f.write(f"{utt} {spk}\n")
    with (out / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")


def main():
    analysis = BASE / "mfa" / "analysis_sentence"
    analysis.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in ("dev", "train"):
        manifest = read_manifest(split)
        wav, utt2spk = read_kaldi_aux(split)
        aligned_dir = BASE / "mfa" / "aligned" / f"{split}_textgrid"
        aligned = {p.stem: p for p in aligned_dir.rglob("*.TextGrid")}

        rows = []
        wavs = {}
        manifest_lines = []
        excluded = []
        split_originals = 0
        split_segments = 0

        for utt in sorted(manifest):
            if utt not in wav or utt not in utt2spk:
                excluded.append(f"{utt}\tmissing_aux\t0\t")
                continue
            if utt not in aligned:
                duration = wav_duration(wav[utt])
                reason = "unaligned_long" if duration > 30 else "unaligned_short"
                excluded.append(f"{utt}\t{reason}\t{duration:.3f}\t{manifest[utt]['normalized']}")
                continue

            duration = wav_duration(wav[utt])
            spk = utt2spk[utt]
            wavs[utt] = wav[utt]

            origin_sentences = [s for s in split_origin_sentences(manifest[utt]["origin"]) if simple_tokens(s)]
            if len(origin_sentences) <= 1 and duration <= 30.0:
                rows.append((utt, utt, 0.0, duration, manifest[utt]["normalized"], spk))
                continue

            segments = build_segments_for_utt(
                utt, manifest[utt]["origin"], manifest[utt]["normalized"], duration, aligned[utt]
            )
            if not segments:
                excluded.append(f"{utt}\tno_sentence_segments\t{duration:.3f}\t{manifest[utt]['normalized']}")
                continue
            if len(segments) == 1:
                start, end, text = segments[0]
                # Keep original utterance id when it remains a single segment.
                new_utt = utt if start <= 0.1 and duration - end <= 0.1 else f"{utt}_sent001"
                rows.append((new_utt, utt, start, end, text, spk))
            else:
                split_originals += 1
                split_segments += len(segments)
                for index, (start, end, text) in enumerate(segments, 1):
                    new_utt = f"{utt}_sent{index:03d}"
                    rows.append((new_utt, utt, start, end, text, spk))
                    manifest_lines.append(
                        f"{utt}\t{new_utt}\t{spk}\t{duration:.3f}\t{start:.3f}\t{end:.3f}\t{text}"
                    )

        write_kaldi(BASE / "kaldi_sentence_mfa", split, rows, wavs)

        rescue_rows = read_rescue_rows(split)
        rescue_wavs = dict(wavs)
        for _utt, rec, _start, _end, _text, _spk in rescue_rows:
            if rec in wav:
                rescue_wavs[rec] = wav[rec]
        write_kaldi(BASE / "kaldi_sentence_mfa_rescued", split, rows + rescue_rows, rescue_wavs)

        (analysis / f"{split}_sentence_split_manifest.tsv").write_text(
            "orig_utt\tnew_utt\tspk\torig_duration\tstart\tend\ttext\n"
            + "\n".join(manifest_lines)
            + ("\n" if manifest_lines else ""),
            encoding="utf-8",
        )
        (analysis / f"{split}_excluded.tsv").write_text(
            "utt\treason\tduration\ttext\n" + "\n".join(excluded) + ("\n" if excluded else ""),
            encoding="utf-8",
        )
        summary.append(
            (
                split,
                len(manifest),
                len(aligned),
                len(rows),
                split_originals,
                split_segments,
                len(rescue_rows),
                len(rows) + len(rescue_rows),
                len(excluded),
            )
        )

    lines = [
        "split\tinput\taligned\tkaldi_sentence_utts\tsplit_originals\tsplit_segments\trescue_segments\trescued_sentence_utts\texcluded"
    ]
    lines.extend("\t".join(map(str, x)) for x in summary)
    (analysis / "sentence_split_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
