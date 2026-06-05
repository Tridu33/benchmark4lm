#!/usr/bin/env bash
# Image VLM SFT Training Task Launcher
set -euo pipefail

FRAMEWORK="${1:-huggingface}"
MODEL="${2:-llava-hf/llava-1.5-7b-hf}"
IMAGE_DIR="${3:-}"
DATASET="${4:-}"
BATCH_SIZE="${5:-2}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " Image VLM SFT Training"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo "=========================================="

python3 "$SCRIPT_DIR/profile_train.py" \
    --framework "$FRAMEWORK" --model "$MODEL" \
    --image-dir "$IMAGE_DIR" --train-dataset "$DATASET" \
    --batch-size "$BATCH_SIZE" --out-dir "$OUT_DIR"

echo "Training complete."
