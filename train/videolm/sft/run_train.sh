#!/usr/bin/env bash
# VideoLM SFT Training Task Launcher
set -euo pipefail
FRAMEWORK="${1:-huggingface}"
MODEL="${2:-THUDM/CogVideoX-2b}"
NUM_FRAMES="${3:-16}"
BATCH_SIZE="${4:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../../out"
mkdir -p "$OUT_DIR"
echo "=========================================="
echo " VideoLM SFT Training"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo "=========================================="
python3 "$SCRIPT_DIR/profile_train.py" --framework "$FRAMEWORK" --model "$MODEL" \
    --num-frames "$NUM_FRAMES" --batch-size "$BATCH_SIZE" --out-dir "$OUT_DIR"
echo "Training complete."
