#!/usr/bin/env bash
# Image VLM PEFT Training Task Launcher
set -euo pipefail

PEFT_METHOD="${1:-lora}"
MODEL="${2:-llava-hf/llava-1.5-7b-hf}"
DATASET="${3:-}"
LORA_RANK="${4:-16}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " Image VLM PEFT Training"
echo " Method    : $PEFT_METHOD"
echo " Model     : $MODEL"
echo " LoRA Rank : $LORA_RANK"
echo "=========================================="

python3 "$SCRIPT_DIR/profile_train.py" \
    --peft-method "$PEFT_METHOD" --model "$MODEL" \
    --lora-rank "$LORA_RANK" --out-dir "$OUT_DIR"

echo "PEFT training complete."
