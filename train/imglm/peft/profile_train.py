#!/usr/bin/env python3
"""
Image Multimodal LLM PEFT Training Profiler
LoRA/QLoRA for VLMs (LLaVA, Qwen-VL, etc.)
"""
from __future__ import annotations

import argparse, os, random, sys, time
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from core_profiler import (
    GpuMonitor, CpuMonitor, TorchProfilerCtx,
    ProfilingResult, collect_hw_summary, collect_sw_summary,
    generate_report, CsvTimeSeries, LangFuseTracer,
)

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--peft-method", default="lora")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--out-dir", default="./out")
    p.add_argument("--task-script", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SyntheticVisionDataset(Dataset):
    def __init__(self, n=5000, seq_len=512, vocab=32000):
        self.n, self.seq_len, self.vocab = n, seq_len, vocab
    def __len__(self): return self.n
    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab, (self.seq_len,))
        return {"input_ids": ids, "labels": ids.clone(),
                "pixel_values": torch.randn(3, 336, 336)}


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    if TORCH_AVAILABLE: torch.manual_seed(args.seed)

    if not TORCH_AVAILABLE:
        print("[ERROR] torch required"); sys.exit(1)

    result = ProfilingResult()
    result.hw_summary = collect_hw_summary()
    result.sw_summary = collect_sw_summary()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    csv_logger = CsvTimeSeries(output_dir=args.out_dir, filename="vlm_peft_timeseries.csv")
    gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
    cpu_mon = CpuMonitor(args.monitor_interval)
    gpu_mon.start(); cpu_mon.start()

    try:
        if args.task_script:
            import subprocess
            subprocess.run(["bash", args.task_script], timeout=None)
        else:
            from transformers import GPT2LMHeadModel, GPT2Config
            config = GPT2Config(n_layer=4, n_head=4, n_embd=256, vocab_size=1000)
            model = GPT2LMHeadModel(config).to(device)

            if PEFT_AVAILABLE:
                peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                                         target_modules=["c_attn","c_proj","c_fc"],
                                         task_type="CAUSAL_LM")
                model = get_peft_model(model, peft_config)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"[VLM-PEFT] Trainable: {trainable:,}/{total:,} ({100*trainable/max(total,1):.2f}%)")

            model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            dataset = SyntheticVisionDataset(n=args.max_steps*args.batch_size*4)

            print(f"[VLM-PEFT] Training {args.max_steps} steps ...")
            t0 = time.time()
            step_times = []
            with TorchProfilerCtx(output_dir=args.out_dir, name="vlm_peft_profile") as pt_prof:
                for step in range(args.max_steps):
                    ts = time.time()
                    batch = dataset[step % len(dataset)]
                    input_ids = batch["input_ids"].unsqueeze(0).to(device)
                    loss = model(input_ids=input_ids, labels=input_ids).loss
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    dt = time.time() - ts
                    step_times.append(dt)
                    result.train_loss_list.append(loss.item())
                    if csv_logger: csv_logger.log({"step": step, "loss": loss.item(), "step_time_ms": dt*1000})
                    if step % 50 == 0: print(f"  Step {step}: loss={loss.item():.4f}, time={dt*1000:.1f}ms")
                    if pt_prof.is_active: pt_prof.step()

                t1 = time.time()
                if pt_prof.is_active: result.pt_profiler_summary = pt_prof.summary

            total_time = t1 - t0
            if step_times:
                result.steps_per_sec = len(step_times)/max(total_time,1e-6)
                result.samples_per_sec = result.steps_per_sec * args.batch_size * 4
            result.convergence_status = "converging" if result.train_loss_list[-1] < result.train_loss_list[0] else "stable"
    finally:
        gpu_mon.stop(); cpu_mon.stop()

    csv_logger.close()
    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb; result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb; result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util; result.avg_cpu_util = cpu_mon.avg_util

    generate_report(result, output_path=os.path.join(args.out_dir, "result.md"),
                    scenario_name=f"Image VLM PEFT — {args.peft_method}", model_type="imglm", task_type="training")


if __name__ == "__main__":
    main()
