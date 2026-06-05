#!/usr/bin/env bash
# ============================================================
# LLM Inference Task Launcher
# Launches distributed or single-card inference via the chosen
# framework, while profile_infer.py monitors from the side.
#
# Usage:
#   bash run_infer.sh vllm meta-llama/Llama-3-8B-Instruct
#   bash run_infer.sh sglang lmsys/vicuna-7b-v1.5
#   torchrun --nproc_per_node=4 bash run_infer.sh vllm ...
# ============================================================

set -euo pipefail

FRAMEWORK="${1:-vllm}"
MODEL="${2:-meta-llama/Llama-3-8B-Instruct}"
PROMPT_FILE="${3:-}"
MAX_TOKENS="${4:-256}"
NUM_PROMPTS="${5:-100}"
GPU_ID="${6:-0}"
TP_SIZE="${7:-1}"
PORT="${8:-8000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " LLM Inference Task"
echo " Framework : $FRAMEWORK"
echo " Model     : $MODEL"
echo " GPU ID    : $GPU_ID"
echo " TP Size   : $TP_SIZE"
echo "=========================================="

run_vllm() {
    echo "[vLLM] Starting vLLM OpenAI-compatible server ..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 256 \
        --port "$PORT" \
        --host 0.0.0.0 &
    VLLM_PID=$!

    echo "[vLLM] Waiting for server to be ready ..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[vLLM] Server ready."
            break
        fi
        sleep 1
    done

    # Run benchmark via OpenAI API
    python3 -c "
import time, requests
prompts = [
    'Explain attention mechanisms in transformers.',
    'Write a Python Fibonacci function.',
    'Summarize supervised vs unsupervised learning.',
    'What are challenges in training LLMs?',
    'Describe a modern recommendation system.',
]
base = 'http://localhost:${PORT}/v1'
for i, p in enumerate(prompts):
    t0 = time.time()
    r = requests.post(f'{base}/completions', json={
        'model': '${MODEL}', 'prompt': p,
        'max_tokens': ${MAX_TOKENS}, 'temperature': 0.0,
    })
    dt = time.time() - t0
    d = r.json()
    tok = d.get('usage', {}).get('total_tokens', 0)
    print(f'Request {i}: {dt*1000:.0f}ms, {tok} tokens')
"

    kill $VLLM_PID 2>/dev/null || true
}

run_sglang() {
    echo "[SGLang] Starting SGLang server ..."
    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp "$TP_SIZE" \
        --port "$PORT" &
    SGL_PID=$!

    sleep 10

    python3 -c "
import time, requests
prompts = ['Explain attention mechanisms.', 'Write a sort function in Python.']
for i, p in enumerate(prompts):
    t0 = time.time()
    r = requests.post('http://localhost:${PORT}/generate', json={
        'text': p, 'sampling_params': {'max_new_tokens': ${MAX_TOKENS}, 'temperature': 0.0}
    })
    dt = time.time() - t0
    print(f'Request {i}: {dt*1000:.0f}ms')
"

    kill $SGL_PID 2>/dev/null || true
}

run_ollama() {
    echo "[Ollama] Pulling model (if not cached) ..."
    ollama pull "${MODEL##*/}" 2>/dev/null || true

    echo "[Ollama] Running inference ..."
    MODEL_SHORT="${MODEL##*/}"
    for i in $(seq 1 5); do
        time ollama run "$MODEL_SHORT" "Explain attention mechanisms in 3 sentences." \
            --num-predict "$MAX_TOKENS"
    done
}

run_llamafile() {
    echo "[Llamafile] Starting llamafile server ..."
    if [ ! -f "${MODEL}" ]; then
        echo "[Llamafile] ERROR: Model file not found: $MODEL"
        exit 1
    fi
    ./"$MODEL" --server --port "$PORT" --nobrowser &
    LF_PID=$!
    sleep 5

    python3 -c "
import time, requests
for i in range(5):
    t0 = time.time()
    r = requests.post('http://localhost:${PORT}/v1/chat/completions', json={
        'model': '${MODEL}', 'messages': [{'role':'user','content':'Explain transformers.'}],
        'max_tokens': ${MAX_TOKENS}, 'temperature': 0.0
    })
    print(f'Request {i}: {(time.time()-t0)*1000:.0f}ms')
"

    kill $LF_PID 2>/dev/null || true
}

case "$FRAMEWORK" in
    vllm)         run_vllm ;;
    sglang)       run_sglang ;;
    ollama)       run_ollama ;;
    llamafile)    run_llamafile ;;
    ktransformers)
        echo "[kTransformers] Use profile_infer.py directly for kTransformers."
        python3 "$SCRIPT_DIR/profile_infer.py" \
            --framework ktransformers \
            --model "$MODEL" \
            --max-tokens "$MAX_TOKENS" \
            --num-prompts "$NUM_PROMPTS" \
            --gpu-id "$GPU_ID"
        ;;
    trt_llm)
        echo "[TensorRT-LLM] Use profile_infer.py directly for TensorRT-LLM."
        python3 "$SCRIPT_DIR/profile_infer.py" \
            --framework trt_llm \
            --model "$MODEL" \
            --max-tokens "$MAX_TOKENS" \
            --num-prompts "$NUM_PROMPTS" \
            --gpu-id "$GPU_ID"
        ;;
    *)
        echo "Unknown framework: $FRAMEWORK"
        echo "Supported: vllm, sglang, ollama, llamafile, ktransformers, trt_llm"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo " Inference task complete."
echo "=========================================="
