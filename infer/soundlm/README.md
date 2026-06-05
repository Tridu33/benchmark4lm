# Speech/Audio LLM Inference Profiler

Performance profiling for **Speech Large Models** (Whisper, SpeechT5, SeamlessM4T, Qwen-Audio).

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **Whisper** | OpenAI Whisper via HuggingFace pipeline |
| **HuggingFace** | `AutoModelForSpeechSeq2Seq` pipeline |
| **API** | OpenAI Whisper API (cloud-based) |

## Speech-Specific Metrics

- **Audio preprocessing latency**: Feature extraction (log-mel spectrogram)
- **Streaming decode latency**: Chunked audio processing time
- **Transcription throughput**: Tokens/sec for ASR output
- **Encoder-decoder asymmetry**: Encode (audio) vs decode (text) timing
- **Batch audio processing**: Multi-file throughput

## Quick Start

```bash
# Whisper profiling
python profile_infer.py \
    --framework whisper \
    --model openai/whisper-large-v3 \
    --audio-dir ./test_audio \
    --out-dir ./out

# Via task script
bash run_infer.sh whisper openai/whisper-large-v3 ./test_audio

# API-based profiling
python profile_infer.py --framework api --out-dir ./out
```

## Performance Characteristics for Speech Inference

Speech inference is characterized by:
- **Audio feature extraction**: Log-mel spectrogram computation (CPU or GPU)
- **Encoder-heavy**: Audio encoder dominates processing time
- **Streaming vs batched**: Whisper supports chunked processing
- **Variable-length audio**: Duration affects processing time linearly
- **Language detection**: Additional overhead for automatic language identification
