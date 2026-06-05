#!/usr/bin/env bash
# LLM PEFT Training Task Launcher
set -euo pipefail

PEFT_METHOD="${1:-lora}"
MODEL="${2:-meta-llama/Llama-3-8B}"
DATASET="${3:-}"
BATCH_SIZE="${4:-4}"
LR="${5:-2e-4}"
LORA_RANK="${6:-16}"
QUANTIZE="${7:-none}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " LLM PEFT Training Task"
echo " Method    : $PEFT_METHOD"
echo " Model     : $MODEL"
echo " LoRA Rank : $LORA_RANK"
echo " Quantize  : $QUANTIZE"
echo "=========================================="

python3 "$SCRIPT_DIR/profile_train.py" \
    --peft-method "$PEFT_METHOD" \
    --model "$MODEL" \
    --train-dataset "$DATASET" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --quantize "$QUANTIZE" \
    --out-dir "$OUT_DIR"

echo "PEFT training complete."
