# Video Generation SFT Training Profiler

Full **Supervised Fine-Tuning** for Video Generation Large Models (CogVideo, VideoCrafter, AnimateDiff, SVD).

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **HuggingFace Diffusers** | UNet/DiT training with Accelerate |
| **DeepSpeed** | ZeRO optimization for video UNet training |

## Video Training Metrics

- **Frame generation quality**: Temporal consistency loss tracking
- **3D attention overhead**: Temporal convolution/attention cost
- **Video batch throughput**: Videos/sec training throughput
- **GPU memory per frame**: VRAM for video tensors (4D+)
- **Gradient stability**: Video training is prone to gradient explosions

## Quick Start

```bash
python profile_train.py \
    --framework huggingface \
    --model THUDM/CogVideoX-2b \
    --video-dir ./train_videos \
    --num-frames 16 \
    --batch-size 1 \
    --out-dir ./out

bash run_train.sh huggingface THUDM/CogVideoX-2b 16 1
```

## Video Training Patterns

- **Memory intensive**: Video tensors are 4D+ (batch, channels, frames, H, W)
- **Small batches**: Typically batch_size=1 due to memory constraints
- **Gradient accumulation**: Essential for effective batch size
- **Temporal consistency**: Key quality metric for generated video
- **Long training**: Video models require many more steps than image models
