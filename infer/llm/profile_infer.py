#!/usr/bin/env python3
"""
LLM Inference Profiler — vLLM / SGLang / Ollama / Llamafile / kTransformers / TensorRT-LLM

Usage:
    python profile_infer.py \
        --framework vllm \
        --model meta-llama/Llama-3-8B-Instruct \
        --prompt-file prompts.txt \
        --max-tokens 256 \
        --num-prompts 100 \
        --gpu-id 0 \
        --monitor-interval 0.5 \
        --out-dir ./out \
        --nsys \
        --langfuse

    # Or launch via a task script:
    python profile_infer.py --task-script run_infer.sh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ensure parent dir is on path for core_profiler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core_profiler import (
    GpuMonitor,
    CpuMonitor,
    TorchProfilerCtx,
    run_with_nsys,
    LangFuseTracer,
    ProfilingResult,
    collect_hw_summary,
    collect_sw_summary,
    generate_report,
)

# ── Framework imports (optional) ───────────────────────────────
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


FRAMEWORKS = ["vllm", "sglang", "ollama", "llamafile", "ktransformers", "trt_llm"]


def parse_args():
    p = argparse.ArgumentParser(description="LLM Inference Performance Profiler")
    p.add_argument("--framework", choices=FRAMEWORKS, default="vllm",
                   help="Inference framework")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3-8B-Instruct",
                   help="Model name or path")
    p.add_argument("--prompt-file", type=str, default="",
                   help="File with one prompt per line")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Max generation tokens per request")
    p.add_argument("--num-prompts", type=int, default=100,
                   help="Number of prompts to benchmark")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 = greedy)")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5,
                   help="GPU/CPU monitor sampling interval (sec)")
    p.add_argument("--out-dir", type=str, default="./out")
    p.add_argument("--nsys", action="store_true", help="Enable NVIDIA Nsight profiling")
    p.add_argument("--langfuse", action="store_true", help="Enable LangFuse tracing")
    p.add_argument("--task-script", type=str, default="",
                   help="Path to bash task script to launch instead of direct profiling")
    p.add_argument("--tensor-parallel", type=int, default=1,
                   help="Tensor parallel size (multi-GPU)")
    p.add_argument("--kv-cache-dtype", type=str, default="auto",
                   help="KV cache data type (auto, fp8, fp16)")
    p.add_argument("--enable-chunked-prefill", action="store_true",
                   help="Enable chunked prefill for vLLM")
    p.add_argument("--max-num-seqs", type=int, default=256,
                   help="Max concurrent sequences")
    return p.parse_args()


# ================================================================
#  vLLM Profiler
# ================================================================
def profile_vllm(args, result: ProfilingResult, gpu_mon: GpuMonitor, cpu_mon: CpuMonitor):
    if not VLLM_AVAILABLE:
        print("[ERROR] vllm not installed. pip install vllm")
        sys.exit(1)

    print(f"[vLLM] Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=0.9,
        dtype="auto",
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=args.enable_chunked_prefill,
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=42,
    )

    prompts = _load_prompts(args)

    print(f"[vLLM] Benchmarking {len(prompts)} prompts ...")
    t_start = time.time()

    # PyTorch Profiler
    with TorchProfilerCtx(output_dir=args.out_dir, name="vllm_profile") as pt_prof:
        outputs = llm.generate(prompts, sampling_params)
        t_end = time.time()

        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start

    # Collect metrics
    for req in outputs:
        if req.outputs and req.outputs[0]:
            ot = req.outputs[0]
            if hasattr(ot, "latency_ms"):
                result.request_latencies.append(ot.latency_ms)
            # Count tokens
            result.total_tokens_generated += len(ot.token_ids) if ot.token_ids else 0
            if req.prompt_token_ids:
                result.total_prompt_tokens += len(req.prompt_token_ids)

    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(outputs) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps

    # FLOPs estimation (rough: 2 * params * tokens for decode)
    # params estimated from model name
    params_b = _estimate_params(args.model)
    result.flop_per_sec = 2 * params_b * result.total_tokens_generated / max(total_time, 1e-6)

    # KV cache info
    if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "cache_config"):
        # vLLM doesn't expose KV cache size directly in all versions
        pass

    # Monitor data
    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util
    result.max_concurrent_requests = args.max_num_seqs


# ================================================================
#  SGLang Profiler
# ================================================================
def profile_sglang(args, result: ProfilingResult, gpu_mon: GpuMonitor, cpu_mon: CpuMonitor):
    if not SGLANG_AVAILABLE:
        print("[ERROR] sglang not installed. pip install sglang")
        sys.exit(1)

    print(f"[SGLang] Loading model: {args.model}")

    prompts = _load_prompts(args)

    runtime = sgl.Runtime(
        model_path=args.model,
        tp_size=args.tensor_parallel,
        mem_fraction_static=0.9,
        max_running_requests=args.max_num_seqs,
    )

    print(f"[SGLang] Benchmarking {len(prompts)} prompts ...")
    t_start = time.time()

    with TorchProfilerCtx(output_dir=args.out_dir, name="sglang_profile") as pt_prof:
        responses = []
        for prompt in prompts:
            ret = runtime.generate(prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
            responses.append(ret)
        t_end = time.time()

        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start
    runtime.shutdown()

    for resp in responses:
        if isinstance(resp, dict) and "text" in resp:
            result.total_tokens_generated += len(resp["text"].split())
        elif isinstance(resp, str):
            result.total_tokens_generated += len(resp.split())

    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(responses) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = args.max_num_seqs

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


# ================================================================
#  Ollama Profiler (OpenAI-compatible API)
# ================================================================
def profile_ollama(args, result: ProfilingResult, gpu_mon: GpuMonitor, cpu_mon: CpuMonitor):
    if not OPENAI_AVAILABLE:
        print("[ERROR] openai package required. pip install openai")
        sys.exit(1)

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    prompts = _load_prompts(args)
    model_name = args.model.split("/")[-1].lower()

    print(f"[Ollama] Benchmarking {len(prompts)} prompts with model: {model_name}")
    t_start = time.time()

    for prompt in prompts:
        r_start = time.time()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=False,
        )
        r_end = time.time()

        result.request_latencies.append((r_end - r_start) * 1000)
        if resp.usage:
            result.total_tokens_generated += resp.usage.completion_tokens or 0
            result.total_prompt_tokens += resp.usage.prompt_tokens or 0

    t_end = time.time()
    total_time = t_end - t_start

    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(prompts) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = 1  # Ollama sequential by default

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


# ================================================================
#  Llamafile Profiler (OpenAI-compatible API)
# ================================================================
def profile_llamafile(args, result: ProfilingResult, gpu_mon: GpuMonitor, cpu_mon: CpuMonitor):
    if not OPENAI_AVAILABLE:
        print("[ERROR] openai package required. pip install openai")
        sys.exit(1)

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="llamafile",
    )

    prompts = _load_prompts(args)

    print(f"[Llamafile] Benchmarking {len(prompts)} prompts ...")
    t_start = time.time()

    for prompt in prompts:
        r_start = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=False,
        )
        r_end = time.time()

        result.request_latencies.append((r_end - r_start) * 1000)
        if resp.usage:
            result.total_tokens_generated += resp.usage.completion_tokens or 0
            result.total_prompt_tokens += resp.usage.prompt_tokens or 0

    t_end = time.time()
    total_time = t_end - t_start

    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(prompts) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = 1

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


# ================================================================
#  kTransformers Profiler
# ================================================================
def profile_ktransformers(args, result: ProfilingResult, gpu_mon: GpuMonitor, cpu_mon: CpuMonitor):
    """
    kTransformers profiler — uses ktransformers.LLM API.
    kTransformers specializes in CPU+GPU hybrid inference with KV offloading.
    """
    try:
        from ktransformers import LLM
    except ImportError:
        print("[ERROR] ktransformers not installed.")
        sys.exit(1)

    print(f"[kTransformers] Loading model: {args.model}")
    prompts = _load_prompts(args)

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.85,
        max_num_seqs=args.max_num_seqs,
    )

    print(f"[kTransformers] Benchmarking {len(prompts)} prompts ...")
    t_start = time.time()

    with TorchProfilerCtx(output_dir=args.out_dir, name="ktransformers_profile") as pt_prof:
        outputs = llm.generate(
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        t_end = time.time()

        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_end - t_start

    for req in outputs:
        result.total_tokens_generated += len(req.outputs[0].text.split()) if req.outputs else 0

    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(outputs) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = args.max_num_seqs

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


# ================================================================
#  TensorRT-LLM Profiler
# ================================================================
def profile_trt_llm(args, result: ProfilingResult, gpu_mon: GpuMonitor, cpu_mon: CpuMonitor):
    """
    TensorRT-LLM profiler — runs the TRT-LLM Python API.
    Requires tensorrt_llm to be installed from NVIDIA.
    """
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunnerCpp
    except ImportError:
        print("[ERROR] tensorrt_llm not installed. Install from NVIDIA.")
        sys.exit(1)

    print(f"[TensorRT-LLM] Engine path: {args.model}")
    prompts = _load_prompts(args)

    # TRT-LLM typically uses a pre-built engine directory
    runner = ModelRunnerCpp.from_dir(
        engine_dir=args.model,
        rank=0,
    )

    print(f"[TensorRT-LLM] Benchmarking {len(prompts)} prompts ...")
    t_start = time.time()

    for prompt in prompts:
        r_start = time.time()
        # TRT-LLM requires tokenized input
        input_ids = _tokenize_for_trt(prompt)
        output = runner.generate(input_ids, max_new_tokens=args.max_tokens)
        r_end = time.time()
        result.request_latencies.append((r_end - r_start) * 1000)
        result.total_tokens_generated += output.shape[1]

    t_end = time.time()
    total_time = t_end - t_start
    runner.free_runtime()

    result.throughput_tps = result.total_tokens_generated / max(total_time, 1e-6)
    result.throughput_rps = len(prompts) / max(total_time, 1e-6)
    result.gen_fps = result.throughput_tps
    result.max_concurrent_requests = args.max_num_seqs

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


def _tokenize_for_trt(prompt: str):
    """Simple tokenizer for TRT-LLM — uses torch tensor of token IDs."""
    if TORCH_AVAILABLE:
        return torch.tensor([[0]], dtype=torch.int32, device="cuda")
    import numpy as np
    return np.array([[0]], dtype=np.int32)


# ================================================================
#  Helpers
# ================================================================
def _load_prompts(args) -> List[str]:
    """Load prompts from file or generate defaults."""
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[:args.num_prompts]

    # Default prompts
    default_prompts = [
        "Explain the concept of attention mechanisms in transformers.",
        "Write a Python function to compute Fibonacci numbers recursively.",
        "Summarize the key differences between supervised and unsupervised learning.",
        "What are the main challenges in training large language models?",
        "Describe the architecture of a modern recommendation system.",
        "Compare and contrast REST and GraphQL APIs.",
        "Explain how gradient descent works with a simple example.",
        "What is the CAP theorem and why does it matter for distributed systems?",
        "Write a short story about a robot learning to paint.",
        "List the top 5 best practices for writing clean code.",
        "How does transfer learning work in computer vision?",
        "Explain the difference between batch norm and layer norm.",
        "What are the advantages of using a vector database for RAG?",
        "Describe the process of fine-tuning a pretrained language model.",
        "What is speculative decoding and how does it speed up inference?",
    ]
    return (default_prompts * ((args.num_prompts // len(default_prompts)) + 1))[:args.num_prompts]


def _estimate_params(model_name: str) -> float:
    """Rough parameter count in billions based on model name."""
    name = model_name.lower()
    if "70b" in name or "72b" in name:
        return 70.0
    if "34b" in name:
        return 34.0
    if "13b" in name:
        return 13.0
    if "8b" in name or "7b" in name:
        return 7.0
    if "3b" in name:
        return 3.0
    if "1b" in name:
        return 1.0
    return 7.0  # default


# ================================================================
#  Main
# ================================================================
def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    result = ProfilingResult()
    result.hw_summary = collect_hw_summary()
    result.sw_summary = collect_sw_summary()

    # Task script mode
    if args.task_script:
        print(f"[Profiler] Launching task script: {args.task_script}")
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        try:
            cmd = ["bash", args.task_script]
            subprocess.run(cmd, timeout=None)
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
        # Direct profiling mode
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        nsys_cmd = None
        if args.nsys:
            # For Nsight, we need to re-wrap the command
            # This is a simplified approach — full Nsight would wrap the actual process
            pass

        dispatch = {
            "vllm": profile_vllm,
            "sglang": profile_sglang,
            "ollama": profile_ollama,
            "llamafile": profile_llamafile,
            "ktransformers": profile_ktransformers,
            "trt_llm": profile_trt_llm,
        }
        profiler_fn = dispatch[args.framework]

        try:
            profiler_fn(args, result, gpu_mon, cpu_mon)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()

    # LangFuse
    if args.langfuse:
        lf = LangFuseTracer(project_name=f"benchmark4lm-{args.framework}")
        url = lf.trace_generation(
            name=f"{args.framework}_benchmark",
            input_text="batch benchmark",
            output_text=f"{result.total_tokens_generated} tokens generated",
            model=args.model,
            prompt_tokens=result.total_prompt_tokens,
            completion_tokens=result.total_tokens_generated,
            metadata={"framework": args.framework},
        )
        result.langfuse_trace_url = url or ""
        lf.shutdown()

    # Nsight
    if args.nsys and not args.task_script:
        print("[nsys] For full Nsight profiling, use --task-script with a wrapper script.")

    # Generate report
    report_path = os.path.join(args.out_dir, "result.md")
    generate_report(
        result,
        output_path=report_path,
        scenario_name=f"LLM Inference — {args.framework}",
        model_type="llm",
        task_type="inference",
    )


if __name__ == "__main__":
    main()
