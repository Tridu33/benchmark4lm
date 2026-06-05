#!/usr/bin/env bash
# RAG Application Task Launcher
set -euo pipefail

FRAMEWORK="${1:-langchain}"
LLM_MODEL="${2:-gpt-4o}"
EMBEDDING_MODEL="${3:-text-embedding-3-small}"
EVAL_DATASET="${4:-}"
NUM_QUESTIONS="${5:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " RAG Application Evaluation"
echo " Framework  : $FRAMEWORK"
echo " LLM        : $LLM_MODEL"
echo " Embedding  : $EMBEDDING_MODEL"
echo " Questions  : $NUM_QUESTIONS"
echo "=========================================="

python3 "$SCRIPT_DIR/profile_rag.py" \
    --framework "$FRAMEWORK" \
    --llm-model "$LLM_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --eval-dataset "$EVAL_DATASET" \
    --num-questions "$NUM_QUESTIONS" \
    --out-dir "$OUT_DIR" \
    --langfuse \
    --phoenix

echo "RAG evaluation complete."
