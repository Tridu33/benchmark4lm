#!/usr/bin/env python3
"""
Video Generation LLM Inference Profiler
Targets: CogVideo, VideoCrafter, ModelScope, AnimateDiff, SVD, etc.

Monitors: Video generation FPS, frame-level latency, GPU memory spikes,
          temporal consistency overhead, PCIe bandwidth for frame I/O.

Usage:
    python profile_infer.py \
        --framework diffusers \
        --model THUDM/CogVideoX-2b \
        --prompt-file prompts.txt \
        --num-frames 49 \
        --num-videos 20 \
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import (
        CogVideoXPipeline,
        StableVideoDiffusionPipeline,
        AnimateDiffPipeline,
        AutoPipelineForImage2Video,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Video Generation LLM Inference Profiler")
    p.add_argument("--framework", choices=["diffusers", "cogvideo", "modelscope"], default="diffusers")
    p.add_argument("--model", type=str, default="THUDM/CogVideoX-2b")
    p.add_argument("--prompt-file", type=str, default="")
    p.add_argument("--num-frames", type=int, default=49,
                   help="Number of frames per video")
    p.add_argument("--fps", type=int, default=8,
                   help="Target FPS for video output")
    p.add_argument("--num-videos", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="./out")
    p.add_argument("--langfuse", action="store_true")
    p.add_argument("--task-script", type=str, default="")
    p.add_argument("--dtype", type=str, default="float16",
                   help="Model dtype: float16, float32, bfloat16")
    return p.parse_args()


def _load_prompts(args) -> List[str]:
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            return [l.strip() for l in f if l.strip()][:args.num_videos]
    defaults = [
        "A cat walking on a sunny beach at sunset.",
        "A drone shot flying over a dense forest.",
        "Time-lapse of a flower blooming.",
        "A car driving through a rainy city at night.",
        "An astronaut floating in space with Earth in background.",
        "Waves crashing on rocky cliffs with seagulls flying.",
        "A train crossing a bridge over a waterfall.",
        "Children playing in a field of sunflowers.",
    ]
    return (defaults * ((args.num_videos // len(defaults)) + 1))[:args.num_videos]


def profile_diffusers(args, result, gpu_mon, cpu_mon):
    """Diffusers-based video generation profiling."""
    if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
        print("[ERROR] diffusers and torch required.")
        sys.exit(1)

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.float16)
    device = f"cuda:{args.gpu_id}"

    # Try different pipeline types based on model
    try:
        pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
        pipe.enable_model_cpu_offload()
    except Exception:
        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(args_model, torch_dtype=dtype).to(device)
            pipe.enable_model_cpu_offload()
        except Exception:
            print(f"[ERROR] Cannot load model {args.model} as video pipeline")
            sys.exit(1)

    prompts = _load_prompts(args)
    print(f"[VideoGen] Generating {len(prompts)} videos ({args.num_frames} frames each) ...")

    t_start = time.time()
    with TorchProfilerCtx(output_dir=args.out_dir, name="video_gen_profile") as pt_prof:
        for i, prompt in enumerate(prompts):
            r_start = time.time()
            try:
                frames = pipe(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=50,
                ).frames[0]
                r_end = time.time()
                result.request_latencies.append((r_end - r_start) * 1000)
                result.total_tokens_generated += len(frames)

                # FPS calculation
                gen_time = r_end - r_start
                if gen_time > 0:
                    result.gen_fps = max(result.gen_fps, len(frames) / gen_time)

            except Exception as e:
                print(f"[VideoGen] Error on video {i}: {e}")

        t_end = time.time()
        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start
    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)  # frames/sec
    result.throughput_rps = len(result.request_latencies) / max(total_time, 1e-6)  # videos/sec
    result.max_concurrent_requests = 1  # Video gen is typically sequential

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util

    # FLOP estimation for video diffusion (rough: UNet steps * params * frames)
    num_unet_params = 1e9  # ~1B for typical video UNet
    total_unet_steps = len(prompts) * 50 * args.num_frames  # 50 inference steps
    result.flop_per_sec = 2 * num_unet_params * total_unet_steps / max(total_time, 1e-6)


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

        dispatch = {"diffusers": profile_diffusers, "cogvideo": profile_diffusers, "modelscope": profile_diffusers}
        try:
            dispatch[args.framework](args, result, gpu_mon, cpu_mon)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()

    if args.langfuse:
        lf = LangFuseTracer(project_name="benchmark4lm-videolm")
        result.langfuse_trace_url = lf.trace_generation(
            name="video_gen_benchmark",
            input_text=f"{len(_load_prompts(args))} video prompts",
            output_text=f"{result.total_tokens_generated} frames generated",
            model=args.model,
            metadata={"framework": args.framework, "modality": "video", "num_frames": args.num_frames},
        ) or ""
        lf.shutdown()

    generate_report(
        result,
        output_path=os.path.join(args.out_dir, "result.md"),
        scenario_name=f"Video Generation Inference — {args.framework}",
        model_type="videolm",
        task_type="inference",
    )


if __name__ == "__main__":
    main()
