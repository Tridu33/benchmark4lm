#!/usr/bin/env bash
# Video Generation LLM Inference Task Launcher
set -euo pipefail

FRAMEWORK="${1:-diffusers}"
MODEL="${2:-THUDM/CogVideoX-2b}"
NUM_FRAMES="${3:-49}"
NUM_VIDEOS="${4:-10}"
GPU_ID="${5:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " Video Generation Inference Task"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo " Frames    : $NUM_FRAMES"
echo " Videos    : $NUM_VIDEOS"
echo "=========================================="

python3 "$SCRIPT_DIR/profile_infer.py" \
    --framework "$FRAMEWORK" \
    --model "$MODEL" \
    --num-frames "$NUM_FRAMES" \
    --num-videos "$NUM_VIDEOS" \
    --gpu-id "$GPU_ID" \
    --out-dir "$OUT_DIR"

echo "Inference task complete."
