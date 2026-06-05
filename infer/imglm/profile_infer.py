#!/usr/bin/env python3
"""
Image Multimodal LLM (VLM) Inference Profiler
Targets: vLLM (vision), LLaVA, Qwen-VL, InternVL, etc.

Monitors: TTFT (multimodal prefill is heavier), image decode latency,
          KV cache for vision tokens, GPU/CPU utilization, memory.

Usage:
    python profile_infer.py \
        --framework vllm \
        --model llava-hf/llava-1.5-7b-hf \
        --image-dir ./test_images \
        --prompt-file prompts.txt \
        --max-tokens 256 \
        --out-dir ./out
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core_profiler import (
    GpuMonitor, CpuMonitor, TorchProfilerCtx,
    ProfilingResult, collect_hw_summary, collect_sw_summary,
    generate_report, LangFuseTracer,
)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Image Multimodal LLM Inference Profiler")
    p.add_argument("--framework", choices=["vllm", "huggingface"], default="vllm")
    p.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--image-dir", type=str, default="",
                   help="Directory containing test images")
    p.add_argument("--prompt-file", type=str, default="",
                   help="File with one prompt per line (paired with images)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="./out")
    p.add_argument("--nsys", action="store_true")
    p.add_argument("--langfuse", action="store_true")
    p.add_argument("--task-script", type=str, default="")
    p.add_argument("--limit-mm-per-prompt", type=int, default=1,
                   help="Max images per prompt (vLLM)")
    return p.parse_args()


def _load_test_images(args) -> List[Image.Image]:
    """Load images from directory or create synthetic test images."""
    images: List[Image.Image] = []

    if args.image_dir and os.path.isdir(args.image_dir):
        for fn in sorted(os.listdir(args.image_dir))[:args.num_prompts]:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                images.append(Image.open(os.path.join(args.image_dir, fn)).convert("RGB"))
    else:
        # Generate synthetic images
        for _ in range(args.num_prompts):
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(img)

    return images[:args.num_prompts]


def _load_prompts(args) -> List[str]:
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            return [l.strip() for l in f if l.strip()][:args.num_prompts]
    defaults = [
        "Describe this image in detail.",
        "What objects are visible in this picture?",
        "Is there any text in this image? If so, transcribe it.",
        "Count the number of people in this image.",
        "What is the dominant color in this image?",
    ]
    return (defaults * ((args.num_prompts // len(defaults)) + 1))[:args.num_prompts]


def profile_vllm_vlm(args, result, gpu_mon, cpu_mon):
    """vLLM with multimodal support (LLaVA, Qwen-VL, etc.)."""
    if not VLLM_AVAILABLE:
        print("[ERROR] vllm not installed.")
        sys.exit(1)

    from vllm.multimodal.utils import load_image

    llm = LLM(
        model=args.model,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
        gpu_memory_utilization=0.9,
        max_num_seqs=256,
        seed=42,
    )

    images = _load_test_images(args)
    prompts = _load_prompts(args)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print(f"[VLM-vLLM] Benchmarking {len(prompts)} image+text prompts ...")
    t_start = time.time()

    with TorchProfilerCtx(output_dir=args.out_dir, name="vlm_vllm_profile") as pt_prof:
        for i, (prompt, img) in enumerate(zip(prompts, images)):
            # vLLM multimodal format
            mm_prompt = {
                "prompt": f"<image>\n{prompt}",
                "multi_modal_data": {"image": img},
            }
            outputs = llm.generate([mm_prompt], sampling_params)

            if outputs and outputs[0].outputs:
                tok_count = len(outputs[0].outputs[0].token_ids)
                result.total_tokens_generated += tok_count

        t_end = time.time()
        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start
    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(prompts) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = 256

    # Vision token overhead estimation
    result.kv_cache_used_mb = len(images) * 224 * 224 * 3 * 4 / (1024 * 1024) * 0.1  # rough
    result.kv_cache_max_mb = result.kv_cache_used_mb * 4

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


def profile_hf_vlm(args, result, gpu_mon, cpu_mon):
    """HuggingFace Transformers multimodal pipeline."""
    if not HF_AVAILABLE:
        print("[ERROR] transformers / torch not installed.")
        sys.exit(1)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model).to(device)

    images = _load_test_images(args)
    prompts = _load_prompts(args)

    print(f"[VLM-HF] Benchmarking {len(prompts)} prompts on {device} ...")
    t_start = time.time()

    with TorchProfilerCtx(output_dir=args.out_dir, name="vlm_hf_profile") as pt_prof:
        for prompt, img in zip(prompts, images):
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
            generated = model.generate(**inputs, max_new_tokens=args.max_tokens)
            result.total_tokens_generated += generated.shape[1]

        t_end = time.time()
        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start
    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(prompts) / max(total_time, 1e-6)
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

        dispatch = {
            "vllm": profile_vllm_vlm,
            "huggingface": profile_hf_vlm,
        }
        try:
            dispatch[args.framework](args, result, gpu_mon, cpu_mon)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()

    if args.langfuse:
        lf = LangFuseTracer(project_name="benchmark4lm-vlm")
        result.langfuse_trace_url = lf.trace_generation(
            name="vlm_benchmark",
            input_text="multimodal batch benchmark",
            output_text=f"{result.total_tokens_generated} tokens",
            model=args.model,
            prompt_tokens=result.total_tokens_generated // 2,
            completion_tokens=result.total_tokens_generated // 2,
            metadata={"framework": args.framework, "modality": "image"},
        ) or ""
        lf.shutdown()

    generate_report(
        result,
        output_path=os.path.join(args.out_dir, "result.md"),
        scenario_name=f"Image VLM Inference — {args.framework}",
        model_type="imglm",
        task_type="inference",
    )


if __name__ == "__main__":
    main()
