# LLM Inference Profiler

Performance profiling and analysis scripts for **Language Large Model** inference.

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **vLLM** | High-throughput LLM serving with PagedAttention |
| **SGLang** | Structured generation language engine |
| **Ollama** | Local LLM runner (OpenAI-compatible API) |
| **Llamafile** | Single-file LLM distribution and serving |
| **kTransformers** | CPU+GPU hybrid inference with KV offloading |
| **TensorRT-LLM** | NVIDIA optimized LLM inference engine |

## Monitored Metrics

- **TTFT** (Time-To-First-Token): P50/P95/P99 latency for first token
- **TPOT** (Time-Per-Output-Token): Per-token generation latency
- **Throughput**: tokens/sec, requests/sec
- **GPU Utilization**: Average, peak, minimum
- **GPU Memory**: Peak and average VRAM usage
- **CPU Utilization & Memory**: System resource usage
- **PCIe Bandwidth**: RX/TX throughput
- **KV Cache**: Used vs max capacity
- **Max Concurrent Requests**: Per-GPU request capacity
- **FLOP/s**: Estimated floating-point operations per second
- **Temperature & Power**: GPU thermal and power metrics

## Quick Start

```bash
# vLLM profiling
python profile_infer.py \
    --framework vllm \
    --model meta-llama/Llama-3-8B-Instruct \
    --max-tokens 256 \
    --num-prompts 100 \
    --out-dir ./out

# SGLang profiling
python profile_infer.py \
    --framework sglang \
    --model lmsys/vicuna-7b-v1.5 \
    --out-dir ./out

# Via task script (starts server + benchmarks)
bash run_infer.sh vllm meta-llama/Llama-3-8B-Instruct

# With NVIDIA Nsight Systems profiling
python profile_infer.py \
    --framework vllm \
    --model meta-llama/Llama-3-8B-Instruct \
    --nsys \
    --out-dir ./out

# With LangFuse tracing
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
python profile_infer.py --framework vllm --model ... --langfuse
```

## Output

- `out/result.md` — Comprehensive Markdown performance report
- `out/*.chrome_trace.json` — PyTorch Profiler Chrome trace (viewable in `chrome://tracing`)
- `out/*.nsys-rep` — NVIDIA Nsight Systems report (view in `nsight-sys`)
- `out/*_timeseries.csv` — Time-series metrics CSV

## Performance Characteristics for LLM Inference

LLM inference is characterized by:
- **Memory-bound decode phase**: Each step loads all model weights for one token
- **Compute-bound prefill**: Attention over prompt tokens is compute-heavy
- **KV cache growth**: Memory grows linearly with sequence length and batch size
- **PagedAttention** (vLLM): Eliminates KV cache fragmentation
- **Continuous batching**: Maximizes GPU utilization across varying sequence lengths
