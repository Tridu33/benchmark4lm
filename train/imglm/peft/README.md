# Image VLM PEFT Training Profiler

**Parameter-Efficient Fine-Tuning** for Vision-Language Models (LoRA/QLoRA for LLaVA, Qwen-VL).

## Quick Start

```bash
python profile_train.py \
    --peft-method lora \
    --model llava-hf/llava-1.5-7b-hf \
    --lora-rank 16 \
    --out-dir ./out

bash run_train.sh lora llava-hf/llava-1.5-7b-hf train.jsonl 16
```

## VLM PEFT Patterns

- **LoRA on LLM only**: Vision encoder typically frozen
- **Projector LoRA**: Can also apply LoRA to vision-language projector
- **Memory advantage**: Enables VLM fine-tuning on consumer GPUs
- **Target modules**: Attention layers in the LLM component
