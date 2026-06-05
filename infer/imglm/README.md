# Image Multimodal LLM (VLM) Inference Profiler

Performance profiling for **Vision-Language Models** (LLaVA, Qwen-VL, InternVL, BLIP-2).

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **vLLM (vision)** | vLLM with multimodal input support |
| **HuggingFace** | Transformers `AutoModelForVision2Seq` pipeline |

## VLM-Specific Metrics

- **Image preprocessing latency**: Vision encoder feature extraction time
- **Vision token overhead**: Additional KV cache for image patches
- **Multimodal prefill TTFT**: Slower prefill due to vision tokens
- **Cross-attention overhead**: Image-text alignment computation
- **GPU memory spikes**: Vision encoder + LLM loaded simultaneously

## Quick Start

```bash
# vLLM multimodal profiling
python profile_infer.py \
    --framework vllm \
    --model llava-hf/llava-1.5-7b-hf \
    --image-dir ./test_images \
    --max-tokens 256 \
    --out-dir ./out

# HuggingFace pipeline
python profile_infer.py \
    --framework huggingface \
    --model llava-hf/llava-1.5-7b-hf \
    --image-dir ./test_images \
    --out-dir ./out

# Via task script
bash run_infer.sh vllm llava-hf/llava-1.5-7b-hf ./test_images
```

## Performance Characteristics for VLM Inference

VLM inference adds vision-specific overhead:
- **Vision encoder**: CLIP/ViT processes images before LLM (adds 50-200ms)
- **Image token expansion**: A single image becomes 576-2560 vision tokens
- **KV cache blowup**: Vision tokens consume significant cache memory
- **Mixed modality**: Image encoding (compute-heavy) + text generation (memory-bound)
- **Resolution impact**: Higher resolution = more patches = more tokens = slower prefill
