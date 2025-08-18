#!/usr/bin/env python3
import argparse
import contextlib
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


def find_subsets(data_root: Path):
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        wav_scp = p / "wav.scp"
        if wav_scp.is_file():
            yield p, wav_scp


def which(cmd: str):
    from shutil import which as _which

    return _which(cmd)


def duration_with_wave(path: Path):
    import wave

    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            return None
        return frames / float(rate)


def duration_with_soxi_file(target: str):
    soxi = which("soxi")
    if not soxi:
        return None
    try:
        # soxi -D prints duration in seconds as float
        res = subprocess.run([soxi, "-D", target], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return float(res.stdout.decode().strip())
    except Exception:
        return None


def duration_with_soxi_pipeline(command_with_pipe: str, shell_executable: str = "/bin/bash"):
    """Measure duration for pipeline-style wav.scp entries ending with '|'.

    Executes: <command_without_trailing_pipe> | soxi -D -
    Requires --allow-commands to be enabled by the caller.
    """
    soxi = which("soxi")
    if not soxi:
        return None
    try:
        cmd = command_with_pipe.rstrip().rstrip("|").rstrip()
        if not cmd:
            return None
        full_cmd = f"set -o pipefail; {cmd} | {shlex.quote(soxi)} -D -"
        res = subprocess.run([
            "/bin/bash",
            "-c",
            full_cmd,
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return float(res.stdout.decode().strip())
    except Exception:
        return None


def duration_with_ffprobe(target: str):
    ffprobe = which("ffprobe")
    if not ffprobe:
        return None
    try:
        # Use ffprobe to get duration in seconds
        # For pipes, we can't easily probe; skip in that case
        if target.endswith("|"):
            return None
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            target,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return float(res.stdout.decode().strip())
    except Exception:
        return None


def compute_duration(entry_value: str, method_order: tuple, allow_commands: bool):
    # entry_value is the RHS of wav.scp
    # Could be a file path or a command ending with '|'
    # Try in given method order
    rhs = entry_value.strip()

    def try_wave(rhs_: str):
        if rhs_.endswith("|"):
            return None
        r = rhs_
        if (r.startswith("\"") and r.endswith("\"")) or (r.startswith("'") and r.endswith("'")):
            r = r[1:-1]
        p = Path(r)
        if p.exists() and p.is_file():
            try:
                return duration_with_wave(p)
            except Exception:
                return None
        return None

    def try_soxi(rhs_: str):
        if rhs_.endswith("|"):
            if not allow_commands:
                return None
            return duration_with_soxi_pipeline(rhs_)
        return duration_with_soxi_file(rhs_)

    def try_ffprobe(rhs_: str):
        if rhs_.endswith("|"):
            return None
        return duration_with_ffprobe(rhs_)

    method_map = {"soxi": try_soxi, "wave": try_wave, "ffprobe": try_ffprobe}
    for m in method_order:
        d = method_map[m](rhs)
        if d is not None:
            return d
    return None


def parse_wav_scp_line(line: str):
    line = line.rstrip("\n")
    if not line or line.lstrip().startswith("#"):
        return None, None
    # Split only on the first whitespace
    # Using shlex to respect quoted paths
    try:
        toks = shlex.split(line)
        if len(toks) < 2:
            return None, None
        uttid = toks[0]
        # Reconstruct RHS preserving potential trailing pipe
        rhs = line[line.find(toks[1]) :].strip()
        return uttid, rhs
    except Exception:
        # Fallback: simple split
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan each subset under a data directory, read wav.scp, and "
            "compute per-utterance durations with summary stats."
        )
    )
    default_root = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help=f"Path to data root containing subsets (default: {default_root})",
    )
    parser.add_argument(
        "--write-utt2dur",
        action="store_true",
        help="Write utt2dur in each subset directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-utterance output; print only summary per subset.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "soxi", "wave", "ffprobe"],
        default="auto",
        help="Duration method: prefer soxi (auto), or force a specific tool.",
    )
    parser.add_argument(
        "--allow-commands",
        action="store_true",
        help=(
            "Allow executing pipeline commands in wav.scp (entries ending with '|') "
            "by piping them to 'soxi -D -'. Disabled by default for safety."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar per subset (falls back silently if tqdm missing).",
    )
    parser.add_argument(
        "--hours-line",
        action="store_true",
        help=(
            "Print a single line like 'train: 12.34h, dev: 1.23h, test: 0.89h' "
            "after scanning all subsets. Suppresses per-subset printing."
        ),
    )
    parser.add_argument(
        "--line-order",
        type=str,
        default="train,dev,test",
        help="Comma-separated subset order for --hours-line. Others follow alphabetically.",
    )
    parser.add_argument(
        "-s",
        "--subset",
        action="append",
        help=(
            "Subset name(s) to process (e.g., train,dev). "
            "Can be passed multiple times or comma-separated. "
            "If omitted, all subsets under data-root are processed."
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.is_dir():
        print(f"[ERROR] data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    any_found = False
    subset_totals_sec = {}
    if args.method == "auto":
        method_order = ("soxi", "wave", "ffprobe")
    else:
        method_order = (args.method,)

    # Prepare tqdm if requested
    tqdm_mod = None
    if args.progress:
        try:
            from tqdm import tqdm as _tqdm
            tqdm_mod = _tqdm
        except Exception:
            tqdm_mod = None
    # Resolve subset filter
    target_subsets = None
    if args.subset:
        target_subsets = set()
        for item in args.subset:
            for name in item.split(","):
                name = name.strip()
                if name:
                    target_subsets.add(name)

    for subset_dir, wav_scp in find_subsets(data_root):
        if target_subsets is not None and subset_dir.name not in target_subsets:
            continue
        any_found = True
        if not args.hours_line:
            print(f"\n[Subset] {subset_dir.name}")

        durations = {}
        missing = 0
        with wav_scp.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in lines:
            uttid, rhs = parse_wav_scp_line(line)
            if not uttid:
                continue
            entries.append((uttid, rhs))

        pbar = None
        if args.progress and tqdm_mod is not None:
            pbar = tqdm_mod(total=len(entries), desc=subset_dir.name, unit="utt")

        for uttid, rhs in entries:
            dur = compute_duration(rhs, method_order=method_order, allow_commands=args.allow_commands)
            if dur is None:
                missing += 1
            else:
                durations[uttid] = dur
                if not args.quiet and not args.hours_line:
                    if pbar is not None:
                        try:
                            from tqdm import tqdm as _tqdm
                            _tqdm.write(f"{uttid}\t{dur:.3f}")
                        except Exception:
                            print(f"{uttid}\t{dur:.3f}")
                    else:
                        print(f"{uttid}\t{dur:.3f}")
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if args.write_utt2dur:
            out_path = subset_dir / "utt2dur"
            with out_path.open("w", encoding="utf-8") as o:
                for k, v in sorted(durations.items()):
                    o.write(f"{k} {v:.6f}\n")

        if durations:
            vals = list(durations.values())
            total_sec = sum(vals)
            subset_totals_sec[subset_dir.name] = total_sec
            if not args.hours_line:
                n = len(vals)
                mean = total_sec / n
                med = statistics.median(vals)
                min_v = min(vals)
                max_v = max(vals)
                print(
                    "Summary:",
                    f"n={n}",
                    f"total_sec={total_sec:.1f}",
                    f"total_hours={total_sec/3600:.2f}h",
                    f"mean={mean:.2f}s",
                    f"median={med:.2f}s",
                    f"min={min_v:.2f}s",
                    f"max={max_v:.2f}s",
                )
        else:
            if not args.hours_line:
                print("No durations computed.")

        if missing and not args.hours_line:
            print(f"[Warn] {missing} entries could not be measured.")

    if not any_found:
        print(f"[ERROR] No subsets with wav.scp found under: {data_root}", file=sys.stderr)
        sys.exit(2)

    if args.hours_line and subset_totals_sec:
        # Respect preferred order first
        preferred = [x.strip() for x in args.line_order.split(",") if x.strip()]
        printed = []
        parts = []
        for name in preferred:
            if name in subset_totals_sec:
                hours = subset_totals_sec[name] / 3600.0
                parts.append(f"{name}: {hours:.2f}h")
                printed.append(name)
        # Then the rest in alphabetical order
        for name in sorted(k for k in subset_totals_sec.keys() if k not in printed):
            hours = subset_totals_sec[name] / 3600.0
            parts.append(f"{name}: {hours:.2f}h")
        print(", ".join(parts))


if __name__ == "__main__":
    main()
