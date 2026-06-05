#!/usr/bin/env python3
"""Video Generation LLM PEFT Training Profiler (LoRA for video UNet models)"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from core_profiler import (
    GpuMonitor, CpuMonitor, TorchProfilerCtx,
    ProfilingResult, collect_hw_summary, collect_sw_summary,
    generate_report, CsvTimeSeries,
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
    p.add_argument("--model", default="THUDM/CogVideoX-2b")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--out-dir", default="./out")
    p.add_argument("--task-script", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SyntheticVideoDataset(Dataset):
    def __init__(self, n=1000, nf=16, fs=64):
        self.n, self.nf, self.fs = n, nf, fs
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return {"video": torch.randn(3, self.nf, self.fs, self.fs),
                "labels": torch.randint(0, 1000, (64,))}


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
    csv_logger = CsvTimeSeries(output_dir=args.out_dir, filename="videolm_peft_timeseries.csv")
    gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
    cpu_mon = CpuMonitor(args.monitor_interval)
    gpu_mon.start(); cpu_mon.start()

    try:
        if args.task_script:
            import subprocess
            subprocess.run(["bash", args.task_script], timeout=None)
        else:
            from diffusers import UNet2DConditionModel
            model = UNet2DConditionModel(
                sample_size=args.num_frames//4, in_channels=3, out_channels=3,
                layers_per_block=1, block_out_channels=(64, 128),
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=64, attention_head_dim=4,
            ).to(device)

            if PEFT_AVAILABLE:
                peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                                         target_modules=["to_q","to_k","to_v","to_out.0"],
                                         task_type=None)
                model = get_peft_model(model, peft_config)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"[VideoLM-PEFT] Trainable: {trainable:,}/{total:,} ({100*trainable/max(total,1):.2f}%)")

            model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            dataset = SyntheticVideoDataset(n=args.max_steps*args.batch_size*4, nf=args.num_frames)

            print(f"[VideoLM-PEFT] Training {args.max_steps} steps ...")
            t0 = time.time()
            step_times = []
            with TorchProfilerCtx(output_dir=args.out_dir, name="videolm_peft_profile") as pt_prof:
                for step in range(args.max_steps):
                    ts = time.time()
                    batch = dataset[step % len(dataset)]
                    video = batch["video"].unsqueeze(0).to(device)
                    noise = torch.randn_like(video)
                    t = torch.tensor([500], device=device, dtype=torch.long)
                    enc = torch.randn(1, 1, 64, device=device)
                    pred = model(video + 0.1*noise, t, encoder_hidden_states=enc).sample
                    loss = torch.nn.functional.mse_loss(pred, noise)
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    dt = time.time() - ts
                    step_times.append(dt)
                    result.train_loss_list.append(loss.item())
                    if csv_logger: csv_logger.log({"step": step, "loss": loss.item(), "step_time_ms": dt*1000})
                    if step % 20 == 0: print(f"  Step {step}: loss={loss.item():.4f}, time={dt*1000:.1f}ms")
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
    result.gpu_snapshots = list(gpu_mon.snapshots); result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb; result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb; result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util; result.avg_cpu_util = cpu_mon.avg_util

    generate_report(result, output_path=os.path.join(args.out_dir, "result.md"),
                    scenario_name=f"Video PEFT — {args.peft_method}", model_type="videolm", task_type="training")


if __name__ == "__main__":
    main()
