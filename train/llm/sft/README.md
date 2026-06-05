# LLM SFT Training Profiler

Full **Supervised Fine-Tuning (SFT)** performance profiling for Language Large Models.

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **HuggingFace Transformers** | `Trainer` + `Accelerate` + `Datasets` |
| **DeepSpeed** | ZeRO-1/2/3, Offload, Stage 3 |
| **Megatron-LM** | NVIDIA's data/ tensor/pipeline parallel |
| **LLaMA-Factory** | Unified fine-tuning framework |
| **Axolotl** | YAML-config based fine-tuning |
| **Colossal-AI** | Auto-parallel training |
| **LMDeploy** | Serving-focused training |
| **TGI** | Text Generation Inference |

## Monitored Metrics

### Training Convergence
- **Training loss**: Per-step loss curve (initial, final, min)
- **Eval loss**: Periodic evaluation loss
- **Convergence status**: Converging / Stable / Diverging
- **Gradient norm**: Mean and max gradient norms
- **Learning rate schedule**: Warmup + decay tracking

### Training Speed
- **Steps/sec**: Training iterations per second
- **Samples/sec**: Effective samples processed per second
- **Step time distribution**: P50/P95/P99 step latency

### Resource Utilization
- **GPU utilization**: Average, peak, minimum
- **GPU memory**: Peak and average VRAM
- **CPU utilization & memory**: Data loading overhead
- **PCIe bandwidth**: Host-device data transfer
- **Temperature & power**: GPU thermal metrics

### Code-Level Profiling
- **PyTorch Profiler**: Operator-level CUDA/CPU time breakdown
- **Nsight Systems**: GPU kernel execution, memory copy, CUDA API
- **Memory allocation**: Potential fragmentation detection

## Quick Start

```bash
# HuggingFace SFT
python profile_train.py \
    --framework huggingface \
    --model meta-llama/Llama-3-8B \
    --train-dataset train.jsonl \
    --batch-size 4 \
    --epochs 3 \
    --lr 2e-5 \
    --out-dir ./out

# DeepSpeed (ZeRO-2)
python profile_train.py \
    --framework deepspeed \
    --model meta-llama/Llama-3-8B \
    --batch-size 4 \
    --lr 2e-5 \
    --out-dir ./out

# Via task script (distributed)
bash run_train.sh huggingface meta-llama/Llama-3-8B train.jsonl 4 3 2e-5 512 1

# With Nsight profiling
python profile_train.py --framework huggingface --model ... --nsys

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 bash run_train.sh deepspeed meta-llama/Llama-3-8B
```

## Training Performance Patterns

SFT training for LLMs is characterized by:
- **Memory-heavy**: Full fine-tuning requires 3-4x model size in VRAM
- **Activation checkpointing**: Trades compute for memory savings
- **Gradient accumulation**: Effective batch size = micro_batch × accum_steps
- **ZeRO optimization**: DeepSpeed shards optimizer states across GPUs
- **Warmup essential**: Linear warmup prevents early divergence
- **Long sequences**: O(n²) attention cost for long training sequences
