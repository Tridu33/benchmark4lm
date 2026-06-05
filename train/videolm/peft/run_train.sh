#!/usr/bin/env bash
# VideoLM PEFT Training Task Launcher
set -euo pipefail
PEFT_METHOD="${1:-lora}"
MODEL="${2:-THUDM/CogVideoX-2b}"
LORA_RANK="${3:-16}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../../out"
mkdir -p "$OUT_DIR"
echo "=========================================="
echo " VideoLM PEFT Training"
echo " Method    : $PEFT_METHOD"
echo " Model     : $MODEL"
echo "=========================================="
python3 "$SCRIPT_DIR/profile_train.py" --peft-method "$PEFT_METHOD" --model "$MODEL" \
    --lora-rank "$LORA_RANK" --out-dir "$OUT_DIR"
echo "PEFT training complete."
