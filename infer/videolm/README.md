# Video Generation LLM Inference Profiler

Performance profiling for **Video Generation Large Models** (CogVideo, VideoCrafter, ModelScope, AnimateDiff, SVD).

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **Diffusers** | HuggingFace DiffusionPipeline for video generation |
| **CogVideo** | THUDM CogVideo pipeline |
| **ModelScope** | Alibaba ModelScope video generation |

## Video-Specific Metrics

- **Generation FPS**: Frames generated per second
- **Per-video latency**: Total time from prompt to video output
- **Temporal consistency overhead**: 3D convolution/attention cost
- **Frame-level memory**: GPU memory per video frame
- **UNet step distribution**: Time per denoising step
- **Guidance scale impact**: Classifier-free guidance overhead

## Quick Start

```bash
# Diffusers video generation profiling
python profile_infer.py \
    --framework diffusers \
    --model THUDM/CogVideoX-2b \
    --num-frames 49 \
    --num-videos 20 \
    --out-dir ./out

# Via task script
bash run_infer.sh diffusers THUDM/CogVideoX-2b 49 10
```

## Performance Characteristics for Video Generation

Video generation is the most resource-intensive inference task:
- **Diffusion steps**: 20-50 UNet forward passes per video
- **3D temporal attention**: Processes time dimension across frames
- **Memory spikes**: Video tensors are 4D+ (batch, channels, frames, height, width)
- **Sequential generation**: Videos are typically generated one at a time
- **Guidance overhead**: Classifier-free guidance doubles UNet calls per step
