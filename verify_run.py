import hashlib
import json
import sys
from pathlib import Path

base, repo, output, mode, model, api_url = sys.argv[1:]
base, repo, output = map(Path, (base, repo, output))

upstream = repo / "benchmark_test/scripts/test_improved_cot.py"
script_hash = hashlib.sha256(upstream.read_bytes()).hexdigest()
expected = "c74b73e971866134ff06d7872ba8eb19342b75ef5a326b08c883ec9b712b0eaf"

if script_hash != expected:
    raise SystemExit("ERROR: Upstream benchmark version mismatch")

manifest = base / "data/SHA256SUMS"
data_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

config = {
    "model": model,
    "api_url": api_url,
    "upstream_sha256": script_hash,
    "dataset_manifest_sha256": data_hash,
    "sample_ratio": 1.0,
    "seed": 42,
    "n_shots": 0,
    "temperature": 0.1,
    "top_p": 1.0,
    "max_tokens": 1024,
}

meta = Path(str(output) + ".run.json")

if mode == "resume":
    if not meta.exists():
        raise SystemExit("ERROR: Missing run configuration")
    if json.loads(meta.read_text(encoding="utf-8")) != config:
        raise SystemExit("ERROR: Run configuration mismatch")
elif mode == "new":
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")) != config:
            raise SystemExit("ERROR: Existing run configuration mismatch")
    else:
        if output.exists() or Path(str(output) + ".partial").exists():
            raise SystemExit("ERROR: Existing result or checkpoint without configuration")
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
else:
    raise SystemExit("ERROR: Invalid run mode")

print("PASS: Run configuration verified")
