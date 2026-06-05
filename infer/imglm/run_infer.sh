#!/usr/bin/env bash
# Image Multimodal LLM Inference Task Launcher
set -euo pipefail

FRAMEWORK="${1:-vllm}"
MODEL="${2:-llava-hf/llava-1.5-7b-hf}"
IMAGE_DIR="${3:-}"
MAX_TOKENS="${4:-256}"
GPU_ID="${5:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " Image VLM Inference Task"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo "=========================================="

case "$FRAMEWORK" in
    vllm)
        echo "[vLLM] Starting VLM inference server ..."
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --limit-mm-per-prompt image:1 \
            --gpu-memory-utilization 0.9 \
            --port 8000 &
        PID=$!
        sleep 30

        # Run multimodal benchmark
        python3 "$SCRIPT_DIR/profile_infer.py" \
            --framework vllm --model "$MODEL" \
            --image-dir "$IMAGE_DIR" \
            --max-tokens "$MAX_TOKENS" \
            --gpu-id "$GPU_ID" \
            --out-dir "$OUT_DIR"

        kill $PID 2>/dev/null || true
        ;;
    huggingface)
        python3 "$SCRIPT_DIR/profile_infer.py" \
            --framework huggingface --model "$MODEL" \
            --image-dir "$IMAGE_DIR" \
            --max-tokens "$MAX_TOKENS" \
            --gpu-id "$GPU_ID" \
            --out-dir "$OUT_DIR"
        ;;
    *)
        echo "Unknown framework: $FRAMEWORK"
        exit 1
        ;;
esac

echo "Inference task complete."
