#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys
import re
import shlex
from typing import List, Tuple, Optional
from tqdm import tqdm

# --- Utilities ---

def find_mp3_path(cmd: str) -> Optional[str]:
    """
    Extract input media file path from a typical ffmpeg command like:
      ffmpeg -i /path/to/file.mp3 -f wav - | ...
    First try regex mirroring the perl (`(\S+)` = no spaces), then fall back to shlex parse.
    """
    m = re.match(r".* -i (\S+) -f .*", cmd)
    if m:
        return m.group(1)
    # fallback: tokenized parse
    try:
        toks = shlex.split(cmd)
        for i, tok in enumerate(toks):
            if tok == "-i" and i + 1 < len(toks):
                return toks[i + 1]
    except Exception:
        pass
    return None


def ffprobe_duration_seconds(path: str) -> Optional[float]:
    """Return duration in seconds using ffprobe (float), or None on failure."""
    try:
        # ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 file
        res = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1",
             path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
        return None
    except FileNotFoundError:
        raise SystemExit("Error: ffprobe not found. Please install FFmpeg.")
    except Exception:
        return None


def read_wav_scp(wav_scp_path: Path) -> List[Tuple[str, str]]:
    """Read wav.scp lines as (utt, cmd)."""
    pairs: List[Tuple[str, str]] = []
    with wav_scp_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # split once: uttid and the remaining command
            try:
                utt, cmd = ln.split(maxsplit=1)
            except ValueError:
                # malformed line; skip
                continue
            pairs.append((utt, cmd))
    return pairs


def write_segments(sorted_entries: List[Tuple[str, str, float]], out_path: Path) -> None:
    """Write segments lines: utt utt 0 <dur> with 6 decimals, sorted by utt."""
    with out_path.open("w", encoding="utf-8") as w:
        for utt, utt2, dur in sorted_entries:
            w.write(f"{utt} {utt2} 0 {dur:.6f}\n")


def maybe_run_kaldi_utils(data_dir: Path, utils_dir: Path) -> None:
    """Run Kaldi's fix/validate if available; otherwise, just warn."""
    fix = utils_dir / "fix_data_dir.sh"
    val = utils_dir / "validate_data_dir.sh"
    env = None

    if fix.exists():
        subprocess.run([str(fix), str(data_dir)], check=False)
    else:
        print(f"[warn] {fix} not found; skipping.", file=sys.stderr)

    if val.exists():
        subprocess.run([str(val), "--no-feats", str(data_dir)], check=False)
    else:
        print(f"[warn] {val} not found; skipping.", file=sys.stderr)


def make_segments_one_dir(data_dir: Path, utils_dir: Path, show_errors: bool = False) -> None:
    wav_scp = data_dir / "wav.scp"
    if not wav_scp.exists():
        print(f"[skip] {wav_scp} not found.", file=sys.stderr)
        return

    pairs = read_wav_scp(wav_scp)
    entries: List[Tuple[str, str, float]] = []

    pbar = tqdm(pairs, desc=f"Processing {data_dir}", unit="utt")
    for utt, cmd in pbar:
        mp3 = find_mp3_path(cmd)
        if mp3 is None:
            if show_errors:
                print(f"[error] could not parse -i <path> from cmd for utt={utt}", file=sys.stderr)
            continue

        dur = ffprobe_duration_seconds(mp3)
        if dur is None:
            if show_errors:
                print(f"[error] ffprobe failed for utt={utt}, path={mp3}", file=sys.stderr)
            continue

        entries.append((utt, utt, float(dur)))

    # Sort by utt (first column)
    entries.sort(key=lambda x: x[0])

    # Write segments
    out_path = data_dir / "segments"
    write_segments(entries, out_path)
    print(f"[ok] wrote {out_path} with {len(entries)} lines.")

    # Run Kaldi utils if available
    maybe_run_kaldi_utils(data_dir, utils_dir)


def main():
    ap = argparse.ArgumentParser(description="Create Kaldi segments from wav.scp using ffprobe (with progress bar).")
    ap.add_argument("--root", type=Path, required=True,
                    help="Root that contains subset subdirectories (each has wav.scp).")
    ap.add_argument("--subsets", nargs="+", default=["train_ja", "dev_ja", "test_ja"],
                    help="Subdirectory names under --root to process.")
    ap.add_argument("--utils-dir", type=Path, default=Path("utils"),
                    help="Path to Kaldi 'utils' directory (contains fix_data_dir.sh / validate_data_dir.sh).")
    ap.add_argument("--show-errors", action="store_true",
                    help="Print per-utterance parsing/probe errors.")
    args = ap.parse_args()

    for subset in args.subsets:
        data_dir = args.root / subset
        make_segments_one_dir(data_dir, args.utils_dir, show_errors=args.show_errors)


if __name__ == "__main__":
    main()
