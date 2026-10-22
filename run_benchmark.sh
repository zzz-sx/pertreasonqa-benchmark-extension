#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 MODEL_NAME API_URL RUN_ID [WORKERS] [new|resume]"
    exit 1
fi

MODEL_NAME="$1"
API_URL="$2"
RUN_ID="$3"
WORKERS="${4:-8}"
MODE="${5:-new}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$BASE")/pertreason-s"
DATA="$BASE/data/sampled_data"
OUTPUT="$BASE/results/${RUN_ID}.jsonl"

RESUME_ARGS=()

case "$MODE" in
    new) ;;
    resume) RESUME_ARGS=(--resume) ;;
    *)
        echo "MODE must be new or resume"
        exit 1
        ;;
esac

mkdir -p "$BASE/results"

(cd "$DATA" && sha256sum -c ../SHA256SUMS >/dev/null)

python "$BASE/verify_run.py"     "$BASE" "$REPO" "$OUTPUT" "$MODE" "$MODEL_NAME" "$API_URL"

python "$BASE/benchmark_test/scripts/run_improved_cot_resume.py" \
    --repo_dir "$REPO" \
    --api_url "$API_URL" \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA" \
    --sample_ratio 1.0 \
    --seed 42 \
    --n_shots 0 \
    --temperature 0.1 \
    --top_p 1.0 \
    --max_tokens 1024 \
    --max_workers "$WORKERS" \
    --checkpoint_every 10 \
    --output "$OUTPUT" \
    "${RESUME_ARGS[@]}"
