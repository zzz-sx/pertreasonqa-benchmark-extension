#!/usr/bin/env python3
"""Resumable runner; leaves the original Improved CoT script untouched."""
import argparse
import collections
import getpass
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--checkpoint_every", type=int, default=10)
p.add_argument("--resume", action="store_true")
p.add_argument(
    "--repo_dir",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "pertreason-s"
)
p.add_argument("--elapsed_before_resume", type=float, default=0)
cp, args = p.parse_known_args()
if cp.checkpoint_every < 1 or cp.elapsed_before_resume < 0:
    raise SystemExit("Invalid checkpoint interval or elapsed time.")
if "--api_key" in args or any(a.startswith("--api_key=") for a in args):
    raise SystemExit("Do not pass --api_key on the command line; enter it at the hidden prompt.")

op = argparse.ArgumentParser(add_help=False)
op.add_argument("--output", required=True)
out_args, _ = op.parse_known_args(args)
out = Path(out_args.output).expanduser()
partial = Path(str(out) + ".partial")
runtime = Path(str(out) + ".runtime_seconds")
stats_path = out.with_name(out.stem + "_stats.json")

if out.exists() or stats_path.exists():
    raise SystemExit("Final result or stats file already exists; use a new --output.")
if cp.resume and not partial.is_file():
    raise SystemExit(f"Checkpoint not found: {partial}")
if not cp.resume and partial.exists() and partial.stat().st_size:
    raise SystemExit("Non-empty checkpoint exists; use --resume.")
if not cp.resume and runtime.exists():
    raise SystemExit(f"Runtime file already exists: {runtime}")

root = cp.repo_dir.expanduser().resolve()
script = root / "benchmark_test/scripts/test_improved_cot.py"
expected_sha256 = "c74b73e971866134ff06d7872ba8eb19342b75ef5a326b08c883ec9b712b0eaf"
actual_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()

if actual_sha256 != expected_sha256:
    raise SystemExit(
        "Original benchmark script does not match the pinned upstream version."
    )

source = script.read_text(encoding="utf-8")
print("PASS: Original benchmark script SHA256 verified.", flush=True)

key = getpass.getpass("Enter NEW API Key (input hidden): ").strip()
if not key:
    raise SystemExit("Empty API key.")
args.extend(["--api_key", key])
if re.search(r"add_argument\(\s*['\"]--checkpoint_every['\"]", source):
    args.extend(["--checkpoint_every", str(cp.checkpoint_every)])
sys.argv = [sys.argv[0]] + args

spec = importlib.util.spec_from_file_location("original_cot_resume", script)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def skey(sample):
    obj = {"sample_id": sample.get("id", "unknown"),
           "input_data": sample["input"], "true_label": sample["input"]["label"]}
    return hashlib.sha256(canon(obj).encode()).hexdigest()

def rkey(row):
    obj = {"sample_id": row.get("sample_id", "unknown"),
           "input_data": row["input_data"], "true_label": row["true_label"]}
    return hashlib.sha256(canon(obj).encode()).hexdigest()

def read_partial():
    if not partial.exists():
        return []
    raw = partial.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last = raw.rfind(b"\n")
        tail = raw[last + 1:]
        try:
            json.loads(tail.decode("utf-8"))
            raw += b"\n"
        except (json.JSONDecodeError, UnicodeDecodeError):
            raw = raw[:last + 1] if last >= 0 else b""
            partial.write_bytes(raw)
    rows = []
    for n, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except Exception as e:
            raise RuntimeError(f"Malformed checkpoint line {n}; refusing to skip it.") from e
        if not isinstance(row, dict):
            raise RuntimeError(f"Checkpoint line {n} is not a JSON object.")
        rows.append(row)
    return rows

def evaluate_resume(self, test_data, max_workers=1, checkpoint_path=None, checkpoint_every=None):
    source_counts = collections.Counter(skey(s) for s in test_data)
    old_rows = read_partial() if cp.resume else []
    old_counts = collections.Counter()
    queues = collections.defaultdict(collections.deque)

    for row in old_rows:
        try:
            k = rkey(row)
        except Exception as e:
            raise RuntimeError("Checkpoint record lacks required fields.") from e
        if k not in source_counts:
            raise RuntimeError("Checkpoint contains a sample outside the current test set.")
        old_counts[k] += 1
        if old_counts[k] > source_counts[k]:
            raise RuntimeError("Checkpoint has more records for a sample than the source data.")
        queues[k].append(row)

    ordered = [None] * len(test_data)
    pending = []
    for i, sample in enumerate(test_data):
        k = skey(sample)
        if queues[k]:
            ordered[i] = queues[k].popleft()
        else:
            pending.append((i, sample))

    if not cp.resume:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text("", encoding="utf-8")

    prior = 0.0
    elapsed_known = False
    if cp.resume and runtime.exists():
        prior = float(runtime.read_text(encoding="utf-8").strip() or "0")
        elapsed_known = True
    elif cp.resume and cp.elapsed_before_resume:
        prior = cp.elapsed_before_resume
        elapsed_known = True

    t0 = time.monotonic()
    fresh = []
    done = len(old_rows)
    interval = cp.checkpoint_every

    def save_runtime():
        tmp = Path(str(runtime) + ".tmp")
        tmp.write_text(f"{prior + time.monotonic() - t0:.3f}\n", encoding="utf-8")
        os.replace(tmp, runtime)

    def save(force=False):
        if not fresh:
            save_runtime()
            return
        if not force and len(fresh) < interval:
            return
        with partial.open("a", encoding="utf-8") as f:
            for row in fresh:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        fresh.clear()
        save_runtime()
        print(f"\nCHECKPOINT: {done}/{len(test_data)} -> {partial}", flush=True)

    print(f"Recovered {len(old_rows)}; remaining {len(pending)}; workers={max_workers}; checkpoint_every={interval}", flush=True)
    if cp.resume and not elapsed_known:
        print("Previous runtime unknown; stats elapsed time covers this resume session only.", flush=True)

    if max_workers > 1:
        with mod.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.evaluate_single, s): i for i, s in pending}
            with mod.tqdm(total=len(test_data), initial=len(old_rows), desc="Testing") as bar:
                for future in mod.as_completed(futures):
                    i = futures[future]
                    row = future.result()
                    ordered[i] = row
                    fresh.append(row)
                    done += 1
                    bar.update(1)
                    save()
    else:
        with mod.tqdm(total=len(test_data), initial=len(old_rows), desc="Testing") as bar:
            for i, sample in pending:
                row = self.evaluate_single(sample)
                ordered[i] = row
                fresh.append(row)
                done += 1
                bar.update(1)
                save()

    save(force=True)
    elapsed = prior + time.monotonic() - t0
    save_runtime()
    if any(row is None for row in ordered):
        raise RuntimeError("Some samples have no result; checkpoint retained.")
    self.results = ordered
    stats = self.compute_statistics(ordered, elapsed, len(test_data))
    stats["resumed_from_checkpoint"] = bool(cp.resume)
    stats["checkpoint_records_at_start"] = len(old_rows)
    stats["elapsed_time_history_complete"] = elapsed_known or not cp.resume
    return stats

mod.ImprovedCOTEvaluator.evaluate = evaluate_resume
started = time.time()
mod.main()
if out.exists() and stats_path.exists() and out.stat().st_mtime >= started and stats_path.stat().st_mtime >= started:
    partial.unlink(missing_ok=True)
    runtime.unlink(missing_ok=True)
    Path(str(runtime) + ".tmp").unlink(missing_ok=True)
    print("Final results and stats saved; checkpoint cleaned up.", flush=True)
else:
    print(f"Final files not confirmed; checkpoint retained at {partial}", flush=True)
