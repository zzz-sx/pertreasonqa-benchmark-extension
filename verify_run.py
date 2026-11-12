"""Verify the bundled evaluator, fixed dataset, and resumable run configuration."""
import hashlib
import json
from pathlib import Path

CORE_SHA256 = "c74b73e971866134ff06d7872ba8eb19342b75ef5a326b08c883ec9b712b0eaf"


def verify_dataset(base):
    base = Path(base)
    data = base / "data/sampled_data"
    entries = (base / "data/SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(entries) != 8:
        raise ValueError("Dataset manifest must contain exactly 8 files")
    for entry in entries:
        expected, name = entry.split(maxsplit=1)
        path = data / name.lstrip(" *")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset checksum mismatch: {path.name}")
    print("PASS: All 8 dataset files verified", flush=True)


def verify_run(base, output, mode, model, api_url, sample_ratio=1.0):
    base, output = Path(base), Path(output)
    core = base / "benchmark_test/scripts/test_improved_cot.py"
    script_hash = hashlib.sha256(core.read_bytes()).hexdigest()
    if script_hash != CORE_SHA256:
        raise ValueError("Bundled benchmark checksum mismatch")

    config = {
        "model": model,
        "api_url": api_url,
        # Keep the historical metadata key so existing checkpoints remain usable.
        "upstream_sha256": script_hash,
        "dataset_manifest_sha256": hashlib.sha256((base / "data/SHA256SUMS").read_bytes()).hexdigest(),
        "sample_ratio": sample_ratio,
        "seed": 42,
        "n_shots": 0,
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 1024,
    }
    meta = Path(str(output) + ".run.json")
    partial = Path(str(output) + ".partial")
    stats = output.with_name(output.stem + "_stats.json")
    if output.exists() or stats.exists():
        raise ValueError("Final results already exist; choose a new RUN_ID")
    if mode == "resume":
        if not meta.is_file() or not partial.is_file():
            raise ValueError("Resume requires both .run.json and .jsonl.partial files")
        if json.loads(meta.read_text(encoding="utf-8")) != config:
            raise ValueError("Run configuration mismatch; use the original model, API URL and sample ratio")
    elif mode == "new":
        if partial.exists() and partial.stat().st_size:
            raise ValueError("Checkpoint exists; use resume or a new RUN_ID")
        if Path(str(output) + ".runtime_seconds").exists():
            raise ValueError("Runtime checkpoint exists; use resume or a new RUN_ID")
        if meta.exists():
            if json.loads(meta.read_text(encoding="utf-8")) != config:
                raise ValueError("Existing run configuration mismatch; choose a new RUN_ID")
        else:
            if partial.exists():
                raise ValueError("Checkpoint exists without run configuration; choose a new RUN_ID")
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        raise ValueError("Invalid run mode")
    print("PASS: Bundled evaluator and run configuration verified", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify all bundled benchmark data (no API calls)")
    parser.parse_args()
    try:
        verify_dataset(Path(__file__).resolve().parent)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
