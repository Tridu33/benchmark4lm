# Speech LLM PEFT Training Profiler

**Parameter-Efficient Fine-Tuning** for Speech/Audio Large Models (LoRA/QLoRA for Whisper).

## Quick Start

```bash
python profile_train.py \
    --peft-method lora \
    --model openai/whisper-large-v3 \
    --lora-rank 16 \
    --out-dir ./out

bash run_train.sh lora openai/whisper-large-v3 16
```

## Speech PEFT Patterns

- **LoRA on decoder**: Apply LoRA to the speech decoder for ASR adaptation
- **Encoder frozen**: Audio encoder typically kept frozen
- **Language adaptation**: Efficient way to add language-specific fine-tuning
- **Memory efficiency**: Enables large Whisper model fine-tuning on single GPU
