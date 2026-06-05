# Speech LLM SFT Training Profiler

Full **Supervised Fine-Tuning** for Speech/Audio Large Models (Whisper, SpeechT5, SeamlessM4T).

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **HuggingFace** | `AutoModelForSpeechSeq2Seq` + Accelerate |
| **DeepSpeed** | ZeRO optimization for encoder-decoder models |

## Speech Training Metrics

- **Audio feature extraction**: Log-mel spectrogram overhead per batch
- **Encoder-decoder loss**: CTC / Seq2Seq training convergence
- **WER tracking**: Word Error Rate during training
- **GPU memory for audio features**: VRAM for spectrogram tensors
- **Training throughput**: Audio+text batch processing speed

## Quick Start

```bash
python profile_train.py \
    --framework huggingface \
    --model openai/whisper-large-v3 \
    --audio-dir ./train_audio \
    --train-dataset train.jsonl \
    --batch-size 8 \
    --out-dir ./out

bash run_train.sh huggingface openai/whisper-large-v3 ./train_audio 8
```

## Speech Training Patterns

- **Encoder-heavy**: Audio encoder dominates compute
- **Long sequences**: 30s audio → 3000 mel frames
- **Batch size sensitivity**: Audio features are large → smaller batches
- **Mixed precision**: fp16 works well for Whisper fine-tuning
