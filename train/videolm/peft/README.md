# Video Generation PEFT Training Profiler

**Parameter-Efficient Fine-Tuning** for Video Generation Models (LoRA for UNet/DiT video models).

## Quick Start

```bash
python profile_train.py \
    --peft-method lora \
    --model THUDM/CogVideoX-2b \
    --lora-rank 16 \
    --out-dir ./out

bash run_train.sh lora THUDM/CogVideoX-2b 16
```

## Video PEFT Patterns

- **LoRA on UNet**: Apply LoRA to attention layers in the video UNet
- **Massive memory savings**: 10-50x VRAM reduction vs full fine-tuning
- **Target modules**: `to_q`, `to_k`, `to_v`, `to_out.0` in temporal + spatial attention
- **Style adaptation**: LoRA excels at video style transfer (anime, cinematic, etc.)
- **Motion LoRA**: Specialized adapters for motion pattern learning
