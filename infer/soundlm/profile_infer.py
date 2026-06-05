#!/usr/bin/env python3
"""
Speech/Audio LLM Inference Profiler
Targets: Whisper, SpeechT5, SeamlessM4T, Qwen-Audio, etc.

Monitors: Audio preprocessing latency, TTFT for audio input,
          streaming token output, GPU/CPU utilization, memory.

Usage:
    python profile_infer.py \
        --framework whisper \
        --model openai/whisper-large-v3 \
        --audio-dir ./test_audio \
        --out-dir ./out
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core_profiler import (
    GpuMonitor, CpuMonitor, TorchProfilerCtx,
    ProfilingResult, collect_hw_summary, collect_sw_summary,
    generate_report, LangFuseTracer,
)

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Speech LLM Inference Profiler")
    p.add_argument("--framework", choices=["whisper", "huggingface", "api"], default="whisper")
    p.add_argument("--model", type=str, default="openai/whisper-large-v3")
    p.add_argument("--audio-dir", type=str, default="",
                   help="Directory containing audio files (.wav, .mp3)")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="./out")
    p.add_argument("--langfuse", action="store_true")
    p.add_argument("--task-script", type=str, default="")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for audio processing")
    return p.parse_args()


def _load_audio_files(args) -> List[str]:
    """List audio files from directory."""
    if args.audio_dir and os.path.isdir(args.audio_dir):
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        files = [
            os.path.join(args.audio_dir, f)
            for f in sorted(os.listdir(args.audio_dir))
            if Path(f).suffix.lower() in exts
        ]
        return files[:args.num_samples]

    # Return synthetic paths — the benchmark will use synthetic audio
    return [f"synthetic_{i}.wav" for i in range(args.num_samples)]


def _generate_synthetic_audio(duration_sec: float = 5.0, sr: int = 16000) -> "torch.Tensor":
    """Generate a random audio tensor for benchmarking."""
    if TORCH_AVAILABLE:
        return torch.randn(int(duration_sec * sr))
    import numpy as np
    return np.random.randn(int(duration_sec * sr))


def profile_whisper(args, result, gpu_mon, cpu_mon):
    """Whisper large model profiling — HF pipeline."""
    if not HF_AVAILABLE or not TORCH_AVAILABLE:
        print("[ERROR] transformers and torch required.")
        sys.exit(1)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device=device,
    )

    audio_files = _load_audio_files(args)
    print(f"[Whisper] Benchmarking {len(audio_files)} audio files ...")

    t_start = time.time()
    with TorchProfilerCtx(output_dir=args.out_dir, name="whisper_profile") as pt_prof:
        for af in audio_files:
            r_start = time.time()
            try:
                if af.startswith("synthetic"):
                    audio_data = _generate_synthetic_audio()
                    result_out = pipe({"raw": audio_data.numpy(), "sampling_rate": 16000})
                else:
                    result_out = pipe(af)
            except Exception as e:
                print(f"[Whisper] Error processing {af}: {e}")
                continue
            r_end = time.time()
            result.request_latencies.append((r_end - r_start) * 1000)
            if result_out and "text" in result_out:
                result.total_tokens_generated += len(result_out["text"].split())

        t_end = time.time()
        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start
    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(result.request_latencies) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = args.batch_size

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


def profile_api(args, result, gpu_mon, cpu_mon):
    """API-based speech recognition (e.g., OpenAI Whisper API)."""
    if not OPENAI_AVAILABLE:
        print("[ERROR] openai package required.")
        sys.exit(1)

    audio_files = _load_audio_files(args)
    client = OpenAI()

    print(f"[API] Benchmarking {len(audio_files)} audio files via Whisper API ...")
    t_start = time.time()

    for af in audio_files:
        if af.startswith("synthetic"):
            continue
        r_start = time.time()
        try:
            with open(af, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1", file=f,
                )
            r_end = time.time()
            result.request_latencies.append((r_end - r_start) * 1000)
            result.total_tokens_generated += len(resp.text.split())
        except Exception as e:
            print(f"[API] Error: {e}")

    t_end = time.time()
    total_time = t_end - t_start
    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(result.request_latencies) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    result = ProfilingResult()
    result.hw_summary = collect_hw_summary()
    result.sw_summary = collect_sw_summary()

    if args.task_script:
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()
        try:
            import subprocess
            subprocess.run(["bash", args.task_script], timeout=None)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()
        result.gpu_snapshots = list(gpu_mon.snapshots)
        result.cpu_snapshots = list(cpu_mon.snapshots)
        result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
        result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
        result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
        result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
        result.avg_gpu_util = gpu_mon.avg_util
        result.avg_cpu_util = cpu_mon.avg_util
    else:
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        dispatch = {"whisper": profile_whisper, "huggingface": profile_whisper, "api": profile_api}
        try:
            dispatch[args.framework](args, result, gpu_mon, cpu_mon)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()

    if args.langfuse:
        lf = LangFuseTracer(project_name="benchmark4lm-soundlm")
        result.langfuse_trace_url = lf.trace_generation(
            name="speech_benchmark",
            input_text=f"{len(_load_audio_files(args))} audio files",
            output_text=f"{result.total_tokens_generated} tokens transcribed",
            model=args.model,
            metadata={"framework": args.framework, "modality": "audio"},
        ) or ""
        lf.shutdown()

    generate_report(
        result,
        output_path=os.path.join(args.out_dir, "result.md"),
        scenario_name=f"Speech LLM Inference — {args.framework}",
        model_type="soundlm",
        task_type="inference",
    )


if __name__ == "__main__":
    main()
