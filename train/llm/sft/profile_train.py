#!/usr/bin/env python3
"""
LLM Full Supervised Fine-Tuning (SFT) Profiler

Targets: Hugging Face Transformers + Accelerate + Datasets, Megatron-LM,
         LLaMA-Factory, Axolotl, DeepSpeed, TGI, LMDeploy, Colossal-AI.

Monitors: Training convergence, loss curves, gradient norms, learning rate schedule,
          steps/sec, samples/sec, GPU/CPU utilization, peak memory, KV cache behavior,
          PCIe bandwidth, thread scheduling, PyTorch profiler operator breakdown,
          Nsight Systems GPU report (.nsys-rep), LangFuse cost tracking.

Usage:
    python profile_train.py \
        --framework deepspeed \
        --model meta-llama/Llama-3-8B \
        --train-dataset train.jsonl \
        --epochs 3 \
        --batch-size 4 \
        --grad-accum 4 \
        --lr 2e-5 \
        --max-steps 1000 \
        --gpu-id 0 \
        --out-dir ./out \
        --nsys \
        --langfuse

    # Or via task script:
    python profile_train.py --task-script run_train.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from core_profiler import (
    GpuMonitor, CpuMonitor, TorchProfilerCtx,
    run_with_nsys, LangFuseTracer,
    ProfilingResult, collect_hw_summary, collect_sw_summary,
    generate_report, CsvTimeSeries,
)

# ── PyTorch ecosystem ─────────────────────────────────────────
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForSeq2Seq,
    )
    from accelerate import Accelerator
    from datasets import load_dataset, Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


FRAMEWORKS = [
    "huggingface", "deepspeed", "megatron", "llama_factory",
    "axolotl", "colossalai", "lmdeploy", "tgi"
]


def parse_args():
    p = argparse.ArgumentParser(description="LLM SFT Training Profiler")
    p.add_argument("--framework", choices=FRAMEWORKS, default="huggingface")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3-8B")
    p.add_argument("--train-dataset", type=str, default="",
                   help="Path to training data (JSONL or HF dataset name)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Max training steps (overrides epochs if set)")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--mixed-precision", type=str, default="bf16",
                   choices=["fp16", "bf16", "fp32", "none"])
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="./out")
    p.add_argument("--nsys", action="store_true",
                   help="Profile with NVIDIA Nsight Systems")
    p.add_argument("--langfuse", action="store_true",
                   help="Track training with LangFuse")
    p.add_argument("--task-script", type=str, default="",
                   help="Bash task script to launch (distributed training)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ================================================================
#  Synthetic Dataset
# ================================================================
class SyntheticTokenDataset(Dataset):
    """Synthetic tokenized dataset for benchmarking when no real data provided."""

    def __init__(self, num_samples: int = 10000, seq_length: int = 512, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# ================================================================
#  HuggingFace SFT Profiler
# ================================================================
def profile_hf_sft(args, result, gpu_mon, cpu_mon, csv_logger):
    """HuggingFace Transformers + Accelerate training loop."""
    if not TORCH_AVAILABLE:
        print("[ERROR] torch required.")
        sys.exit(1)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    print(f"[HF-SFT] Loading model: {args.model}")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "none": torch.float32}
    torch_dtype = dtype_map.get(args.mixed_precision, torch.float32)

    # For benchmarking, use a small model if target is unavailable
    try:
        if HF_AVAILABLE and args.train_dataset:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        else:
            # Use a tiny model for benchmarking
            print("[HF-SFT] Using synthetic small model for benchmark")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                "gpt2",
                n_layer=4,
                n_head=4,
                n_embd=256,
                vocab_size=1000,
            )
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel(config).to(device)
            tokenizer = None
    except Exception as e:
        print(f"[HF-SFT] Model load error: {e}, falling back to synthetic model")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("gpt2", n_layer=2, n_head=2, n_embd=128, vocab_size=500)
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel(config).to(device)
        tokenizer = None

    # Dataset
    if HF_AVAILABLE and args.train_dataset and os.path.exists(args.train_dataset):
        dataset = HFDataset.from_json(args.train_dataset)
        if tokenizer:
            def tokenize_fn(batch):
                return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length)
            dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    else:
        dataset = SyntheticTokenDataset(num_samples=args.max_steps * args.batch_size * args.grad_accum,
                                        seq_length=args.max_seq_length)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"[HF-SFT] Training for {total_steps} steps ...")
    t_train_start = time.time()
    step_times: List[float] = []
    grad_norms: List[float] = []

    with TorchProfilerCtx(output_dir=args.out_dir, name="hf_sft_profile") as pt_prof:
        for step in range(total_steps):
            step_start = time.time()

            # Forward pass with synthetic data
            batch = dataset[step % len(dataset)]
            if isinstance(batch, dict):
                input_ids = batch["input_ids"] if isinstance(batch["input_ids"], torch.Tensor) else torch.tensor(batch["input_ids"])
            else:
                input_ids = batch["input_ids"]

            if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward
            loss.backward()

            # Gradient clipping
            if args.gradient_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Track metrics
            result.train_loss_list.append(loss.item())
            result.learning_rate_list.append(scheduler.get_last_lr()[0])

            # CSV time-series
            if csv_logger:
                csv_logger.log({
                    "step": step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step_time_ms": step_time * 1000,
                    "gpu_mem_mb": gpu_mon.peak_mem_mb if gpu_mon.snapshots else 0,
                    "gpu_util": gpu_mon.snapshots[-1].gpu_util_pct if gpu_mon.snapshots else 0,
                })

            if step % 100 == 0:
                avg_loss = np.mean(result.train_loss_list[-100:])
                avg_step_time = np.mean(step_times[-100:])
                print(f"  Step {step}: loss={avg_loss:.4f}, step_time={avg_step_time*1000:.1f}ms, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            if pt_prof.is_active:
                pt_prof.step()

        t_train_end = time.time()

        if pt_prof.is_active:
            result.pt_profiler_summary = pt_prof.summary

    total_time = t_train_end - t_train_start

    # Aggregate results
    if step_times:
        result.steps_per_sec = len(step_times) / max(total_time, 1e-6)
        result.samples_per_sec = result.steps_per_sec * args.batch_size * args.grad_accum

    # Convergence detection
    if len(result.train_loss_list) > 50:
        early = np.mean(result.train_loss_list[:50])
        late = np.mean(result.train_loss_list[-50:])
        if late < early * 0.8:
            result.convergence_status = "converging"
        elif late > early * 1.2:
            result.convergence_status = "diverging"
        else:
            result.convergence_status = "stable"

    result.grad_norm_list = grad_norms

    # Monitor data
    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


# ================================================================
#  DeepSpeed SFT Profiler
# ================================================================
def profile_deepspeed_sft(args, result, gpu_mon, cpu_mon, csv_logger):
    """DeepSpeed distributed training profiling."""
    if not TORCH_AVAILABLE:
        print("[ERROR] torch required.")
        sys.exit(1)

    if not DEEPSPEED_AVAILABLE:
        print("[WARN] deepspeed not installed, falling back to HF profiler")
        profile_hf_sft(args, result, gpu_mon, cpu_mon, csv_logger)
        return

    device = f"cuda:{args.gpu_id}"
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "none": torch.float32}

    # Simple model for benchmarking
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("gpt2", n_layer=4, n_head=4, n_embd=256, vocab_size=1000)
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel(config)

    # Initialize DeepSpeed
    ds_config = {
        "train_batch_size": args.batch_size * args.grad_accum,
        "gradient_accumulation_steps": args.grad_accum,
        "fp16": {"enabled": args.mixed_precision == "fp16"},
        "bf16": {"enabled": args.mixed_precision == "bf16"},
        "zero_optimization": {"stage": 2},
        "gradient_clipping": args.gradient_clip,
        "steps_per_print": 100,
        "wall_clock_breakdown": True,
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )

    model_engine.train()
    dataset = SyntheticTokenDataset(num_samples=args.max_steps * args.batch_size * args.grad_accum,
                                    seq_length=args.max_seq_length)

    print(f"[DeepSpeed-SFT] Training for {args.max_steps} steps (ZeRO-2) ...")
    t_train_start = time.time()
    step_times: List[float] = []

    for step in range(args.max_steps):
        step_start = time.time()

        batch = dataset[step % len(dataset)]
        input_ids = batch["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(model_engine.device)

        loss = model_engine(input_ids=input_ids, labels=input_ids).loss
        model_engine.backward(loss)
        model_engine.step()

        step_time = time.time() - step_start
        step_times.append(step_time)
        result.train_loss_list.append(loss.item())

        if csv_logger:
            csv_logger.log({"step": step, "loss": loss.item(), "step_time_ms": step_time * 1000})

        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, step_time={step_time*1000:.1f}ms")

    t_train_end = time.time()
    total_time = t_train_end - t_train_start

    if step_times:
        result.steps_per_sec = len(step_times) / max(total_time, 1e-6)
        result.samples_per_sec = result.steps_per_sec * args.batch_size * args.grad_accum

    result.convergence_status = "converging" if result.train_loss_list[-1] < result.train_loss_list[0] else "stable"

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util


# ================================================================
#  Task Script Launcher (for Megatron, LLaMA-Factory, etc.)
# ================================================================
def profile_via_task_script(args, result, gpu_mon, cpu_mon, csv_logger):
    """Launch external training via bash script and monitor."""
    import subprocess

    print(f"[Profiler] Launching task script: {args.task_script}")
    t_start = time.time()

    try:
        proc = subprocess.run(["bash", args.task_script], timeout=None)
    except subprocess.TimeoutExpired:
        print("[Profiler] Task script timed out")
    finally:
        t_end = time.time()

    total_time = t_end - t_start
    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util

    print(f"[Profiler] Task completed in {total_time:.1f}s")


# ================================================================
#  Main
# ================================================================
def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)

    result = ProfilingResult()
    result.hw_summary = collect_hw_summary()
    result.sw_summary = collect_sw_summary()

    csv_logger = CsvTimeSeries(output_dir=args.out_dir, filename=f"{args.framework}_sft_timeseries.csv")

    if args.task_script:
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        if args.nsys:
            nsys_path = run_with_nsys(
                ["bash", args.task_script],
                output_dir=args.out_dir,
                report_name=f"nsys_{args.framework}_sft",
            )
            result.nsys_report_path = nsys_path
        else:
            profile_via_task_script(args, result, gpu_mon, cpu_mon, csv_logger)

        gpu_mon.stop()
        cpu_mon.stop()
    else:
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        try:
            if args.nsys:
                nsys_path = run_with_nsys(
                    [sys.executable, __file__, "--framework", args.framework,
                     "--model", args.model, "--max-steps", str(args.max_steps)],
                    output_dir=args.out_dir,
                    report_name=f"nsys_{args.framework}_sft",
                )
                result.nsys_report_path = nsys_path
            else:
                dispatch = {
                    "huggingface": profile_hf_sft,
                    "deepspeed": profile_deepspeed_sft,
                    "megatron": profile_via_task_script,
                    "llama_factory": profile_via_task_script,
                    "axolotl": profile_via_task_script,
                    "colossalai": profile_via_task_script,
                    "lmdeploy": profile_via_task_script,
                    "tgi": profile_via_task_script,
                }
                dispatch[args.framework](args, result, gpu_mon, cpu_mon, csv_logger)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()

    csv_logger.close()

    # LangFuse
    if args.langfuse:
        lf = LangFuseTracer(project_name="benchmark4lm-train")
        result.langfuse_trace_url = lf.trace_generation(
            name=f"{args.framework}_sft_training",
            input_text=f"SFT training: {args.max_steps} steps",
            output_text=f"Final loss: {result.train_loss_list[-1] if result.train_loss_list else 'N/A'}",
            model=args.model,
            metadata={"framework": args.framework, "epochs": args.epochs,
                      "batch_size": args.batch_size, "lr": args.lr},
        ) or ""
        lf.shutdown()

    generate_report(
        result,
        output_path=os.path.join(args.out_dir, "result.md"),
        scenario_name=f"LLM SFT Training — {args.framework}",
        model_type="llm",
        task_type="training",
    )


if __name__ == "__main__":
    main()
