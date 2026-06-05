#!/usr/bin/env bash
# SoundLM SFT Training Task Launcher
set -euo pipefail
FRAMEWORK="${1:-huggingface}"
MODEL="${2:-openai/whisper-large-v3}"
DATASET="${3:-}"
BATCH_SIZE="${4:-8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../../out"
mkdir -p "$OUT_DIR"
echo "=========================================="
echo " SoundLM SFT Training"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo "=========================================="
python3 "$SCRIPT_DIR/profile_train.py" --framework "$FRAMEWORK" --model "$MODEL" \
    --train-dataset "$DATASET" --batch-size "$BATCH_SIZE" --out-dir "$OUT_DIR"
echo "Training complete."
