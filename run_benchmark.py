#!/usr/bin/env python3
"""Cross-platform entry point for the bundled PertReasonQA benchmark."""
import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

from verify_run import verify_dataset, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model ID supplied by your API provider")
    parser.add_argument("api_url", help="Full chat completions endpoint, including its path")
    parser.add_argument("run_id", help="Unique result name, e.g. model_a_10k")
    parser.add_argument("workers", nargs="?", type=int, default=8)
    parser.add_argument("mode", nargs="?", choices=("new", "resume"), default="new")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                        help="Fraction sampled from each split; default: all 10,000 samples")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be at least 1")
    if not 0 < args.sample_ratio <= 1:
        parser.error("--sample-ratio must be in (0, 1]")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run_id must start with a letter or digit and contain only letters, digits, _, - or .")
    missing = [name for name in ("requests", "tqdm") if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Missing dependencies: " + ", ".join(missing)
                     + ". Run: python -m pip install -r requirements-benchmark.txt")

    base = Path(__file__).resolve().parent
    output = base / "results" / f"{args.run_id}.jsonl"
    try:
        verify_dataset(base)
        verify_run(base, output, args.mode, args.model, args.api_url, args.sample_ratio)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    command = [
        sys.executable, str(base / "benchmark_test/scripts/run_improved_cot_resume.py"),
        "--api_url", args.api_url, "--model_name", args.model,
        "--data_dir", str(base / "data/sampled_data"),
        "--sample_ratio", str(args.sample_ratio), "--seed", "42", "--n_shots", "0",
        "--temperature", "0.1", "--top_p", "1.0", "--max_tokens", "1024",
        "--max_workers", str(args.workers), "--checkpoint_every", "10",
        "--output", str(output),
    ]
    if args.mode == "resume":
        command.append("--resume")
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
