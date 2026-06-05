# Image VLM SFT Training Profiler

Full **Supervised Fine-Tuning** for Vision-Language Models (LLaVA, Qwen-VL, InternVL).

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **HuggingFace** | Transformers + Accelerate for multimodal training |
| **DeepSpeed** | ZeRO optimization for VLM training |
| **LLaMA-Factory** | Unified VLM fine-tuning |

## VLM Training Metrics

- **Vision encoder + LLM loss**: Multimodal training convergence
- **Projector training**: Vision-language alignment layer learning
- **Image feature extraction**: Per-batch vision encoding overhead
- **Multimodal batch throughput**: Samples/sec for image+text batches
- **Memory for image features**: VRAM for vision embeddings

## Quick Start

```bash
python profile_train.py \
    --framework huggingface \
    --model llava-hf/llava-1.5-7b-hf \
    --image-dir ./train_images \
    --train-dataset train.jsonl \
    --batch-size 2 \
    --out-dir ./out

bash run_train.sh huggingface llava-hf/llava-1.5-7b-hf ./train_images train.jsonl 2
```

## VLM Training Patterns

- **Vision encoder frozen**: Often only projector + LLM are trained
- **Image resolution**: Higher resolution = more vision tokens = more memory
- **Mixed precision**: bf16 recommended for stable multimodal training
- **Two-stage training**: (1) projector warmup, (2) full VLM fine-tuning
