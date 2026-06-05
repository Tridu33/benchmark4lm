# LLM PEFT Training Profiler

**Parameter-Efficient Fine-Tuning (PEFT)** profiling for Language Large Models.

## Supported Methods

| Method | Description |
|--------|-------------|
| **LoRA** | Low-Rank Adaptation — trainable rank decomposition matrices |
| **QLoRA** | Quantized LoRA — 4-bit/8-bit base model + LoRA adapters |
| **Prefix Tuning** | Trainable prefix tokens prepended to input |
| **Prompt Tuning** | Trainable soft prompts |

## PEFT-Specific Metrics

- **Trainable parameter ratio**: Percentage of parameters being updated
- **Adapter memory savings**: VRAM reduction vs full fine-tuning
- **Quantization overhead**: 4-bit/8-bit dequantization cost
- **Rank selection impact**: LoRA rank vs quality trade-off
- **Training convergence**: PEFT vs full SFT loss comparison

## Quick Start

```bash
# LoRA training
python profile_train.py \
    --peft-method lora \
    --model meta-llama/Llama-3-8B \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lr 2e-4 \
    --out-dir ./out

# QLoRA (4-bit)
python profile_train.py \
    --peft-method qlora \
    --model meta-llama/Llama-3-8B \
    --quantize 4bit \
    --lora-rank 64 \
    --out-dir ./out

# Prefix Tuning
python profile_train.py \
    --peft-method prefix_tuning \
    --model meta-llama/Llama-3-8B \
    --out-dir ./out

# Via task script
bash run_train.sh lora meta-llama/Llama-3-8B train.jsonl 4 2e-4 16 none
```

## PEFT Performance Patterns

- **Memory efficiency**: LoRA uses 60-80% less VRAM than full SFT
- **QLoRA advantage**: 4-bit quantization enables 70B model fine-tuning on 48GB GPU
- **Rank scaling**: Higher rank = more trainable params = better quality but slower
- **Target modules**: Attention layers (q_proj, v_proj) give best quality/efficiency ratio
- **No inference overhead**: Merged adapters add zero latency at inference time
