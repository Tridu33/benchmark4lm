#!/usr/bin/env bash
# Speech/Audio LLM Inference Task Launcher
set -euo pipefail

FRAMEWORK="${1:-whisper}"
MODEL="${2:-openai/whisper-large-v3}"
AUDIO_DIR="${3:-}"
GPU_ID="${4:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " Speech LLM Inference Task"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo "=========================================="

python3 "$SCRIPT_DIR/profile_infer.py" \
    --framework "$FRAMEWORK" \
    --model "$MODEL" \
    --audio-dir "$AUDIO_DIR" \
    --gpu-id "$GPU_ID" \
    --out-dir "$OUT_DIR"

echo "Inference task complete."
