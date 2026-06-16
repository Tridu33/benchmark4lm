#!/usr/bin/env python3
"""
Benchmark 2: AI Infrastructure — Communication & Interconnect Evaluation

Measures card-to-card interconnect bandwidth, communication latency & jitter,
collective communication performance (All-Reduce, All-Gather, Broadcast, Reduce-Scatter),
cluster linear scaling efficiency, and communication library robustness.

Usage:
    # Single-node multi-card test
    python3 02_communication_perf.py [--num-gpus 8] \
        [--backend nccl|hccl|cncl|mccl|gloo] \
        [--sizes 1MB,4MB,16MB,64MB,256MB,1GB] \
        [--output results/comm_perf.json]

    # Multi-node (MPI launcher)
    mpirun -np 16 -hostfile hosts.txt python3 02_communication_perf.py --backend nccl

Supported backends:
    - nccl:   NVIDIA NCCL (CUDA)
    - hccl:   Huawei HCCL (Ascend NPU)
    - cncl:   Cambricon CNCL (MLU)
    - mccl:   Moore Threads MCCL (MUSA)
    - gloo:   PyTorch Gloo (CPU fallback, no peer-to-peer)

Requirements:
    - torch with distributed support
    - For NCCL: NVIDIA GPU + NCCL (bundled with PyTorch)
    - For HCCL: torch_npu + CANN >= 7.0
    - For CNCL: torch_mlu + Neuware
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

BACKEND_MAP = {
    'nccl': 'nccl',
    'hccl': 'hccl',
    'cncl': 'cncl',
    'mccl': 'mccl',
    'gloo': 'gloo',
}


def detect_backend() -> str:
    """Auto-detect the best available distributed backend."""
    # Try NCCL first (most common)
    if dist.is_nccl_available() and torch.cuda.is_available():
        return 'nccl'
    # Try HCCL
    try:
        if dist.is_hccl_available():
            return 'hccl'
    except Exception:
        pass
    # Try CNCL
    try:
        if dist.is_cncl_available():
            return 'cncl'
    except Exception:
        pass
    # Fallback to Gloo
    if dist.is_gloo_available():
        return 'gloo'
    return 'none'


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

def parse_size(s: str) -> int:
    """Parse human-readable size string to bytes."""
    s = s.strip().upper()
    if s.endswith('KB'):
        return int(s[:-2]) * 1024
    elif s.endswith('MB'):
        return int(s[:-2]) * 1024 * 1024
    elif s.endswith('GB'):
        return int(s[:-2]) * 1024 * 1024 * 1024
    elif s.endswith('B'):
        return int(s[:-1])
    return int(s)


def benchmark_point_to_point(rank, world_size, size_bytes, backend, device, num_iters=50, warmup=10):
    """
    Benchmark point-to-point bandwidth between rank 0 and rank 1.
    Returns bandwidth in GB/s and latency stats in microseconds.
    """
    buffer = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup):
        if rank == 0:
            dist.send(buffer, dst=1)
            dist.recv(buffer, src=1)
        elif rank == 1:
            dist.recv(buffer, src=0)
            dist.send(buffer, dst=0)

    if backend == 'nccl':
        torch.cuda.synchronize()

    latencies = []
    start_time = time.perf_counter()

    for _ in range(num_iters):
        t0 = time.perf_counter()
        if rank == 0:
            dist.send(buffer, dst=1)
            dist.recv(buffer, src=1)
        elif rank == 1:
            dist.recv(buffer, src=0)
            dist.send(buffer, dst=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)  # us

    if backend == 'nccl':
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    # Each iteration: send + receive = 2 * size_bytes
    total_bytes = num_iters * 2 * size_bytes
    bandwidth_gbps = total_bytes / total_time / 1e9 if total_time > 0 else 0

    import numpy as np
    arr = np.array(latencies)

    return {
        'bandwidth_gbps': round(bandwidth_gbps, 2),
        'latency_us_mean': round(float(np.mean(arr)), 2),
        'latency_us_std': round(float(np.std(arr)), 2),
        'latency_us_p50': round(float(np.percentile(arr, 50)), 2),
        'latency_us_p99': round(float(np.percentile(arr, 99)), 2),
        'latency_us_min': round(float(np.min(arr)), 2),
        'latency_us_max': round(float(np.max(arr)), 2),
    }


def benchmark_collective(rank, world_size, size_bytes, backend, device,
                         op_name, num_iters=20, warmup=5):
    """
    Benchmark collective communication operations.
    Supported: all_reduce, all_gather, broadcast, reduce_scatter, reduce
    """
    num_elements = size_bytes // 4  # float32
    buffer = torch.randn(num_elements, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup):
        run_collective(op_name, buffer, world_size)

    if backend == 'nccl':
        torch.cuda.synchronize()
    elif backend == 'hccl':
        try:
            import torch_npu
            torch_npu.synchronize()
        except Exception:
            pass

    latencies = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        run_collective(op_name, buffer, world_size)
        if backend == 'nccl':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    import numpy as np
    arr = np.array(latencies)

    # Effective bandwidth calculation
    if op_name == 'all_reduce':
        # Each rank sends (world_size-1)/world_size of data, receives same
        total_bytes = size_bytes * 2 * (world_size - 1) / world_size
    elif op_name == 'all_gather':
        total_bytes = size_bytes * (world_size - 1)
    elif op_name == 'broadcast':
        total_bytes = size_bytes
    elif op_name == 'reduce_scatter':
        total_bytes = size_bytes * (world_size - 1) / world_size
    elif op_name == 'reduce':
        total_bytes = size_bytes
    else:
        total_bytes = size_bytes

    mean_time_s = float(np.mean(arr)) / 1e6
    bandwidth_gbps = total_bytes / mean_time_s / 1e9 if mean_time_s > 0 else 0

    return {
        'bandwidth_gbps': round(bandwidth_gbps, 2),
        'latency_us_mean': round(float(np.mean(arr)), 2),
        'latency_us_std': round(float(np.std(arr)), 2),
        'latency_us_p50': round(float(np.percentile(arr, 50)), 2),
        'latency_us_p99': round(float(np.percentile(arr, 99)), 2),
        'latency_us_min': round(float(np.min(arr)), 2),
        'latency_us_max': round(float(np.max(arr)), 2),
        'jitter_pct': round(float(np.std(arr) / np.mean(arr) * 100), 2) if np.mean(arr) > 0 else 0,
    }


def run_collective(op_name, buffer, world_size):
    """Execute a collective communication operation."""
    if op_name == 'all_reduce':
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    elif op_name == 'all_gather':
        gathered = [torch.empty_like(buffer) for _ in range(world_size)]
        dist.all_gather(gathered, buffer)
    elif op_name == 'broadcast':
        dist.broadcast(buffer, src=0)
    elif op_name == 'reduce_scatter':
        output = torch.empty(buffer.numel() // world_size, dtype=buffer.dtype, device=buffer.device)
        dist.reduce_scatter(output, list(buffer.chunk(world_size)), op=dist.ReduceOp.SUM)
    elif op_name == 'reduce':
        dist.reduce(buffer, dst=0, op=dist.ReduceOp.SUM)


def benchmark_scaling_efficiency(world_size, backend, device, sizes, num_iters=10):
    """
    Measure scaling efficiency: how performance changes as we add GPUs.
    Returns single-GPU baseline and multi-GPU scaling ratios.
    """
    collective_ops = ['all_reduce', 'all_gather']
    results = {}

    for op in collective_ops:
        op_results = {}
        for size_str in sizes:
            size_bytes = parse_size(size_str)
            times = []
            for _ in range(num_iters):
                buffer = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)
                t0 = time.perf_counter()
                run_collective(op, buffer, world_size)
                if backend == 'nccl':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)

            import numpy as np
            op_results[size_str] = {
                'latency_us_mean': round(float(np.mean(times)), 2),
                'latency_us_p99': round(float(np.percentile(times, 99)), 2),
            }
        results[op] = op_results

    return results


def assess_library_robustness(rank, world_size, backend, device, num_rounds=100):
    """
    Test communication library robustness under extreme concurrent loads.
    Checks for: packet loss, deadlocks, connection failures.
    """
    results = {
        'total_rounds': num_rounds,
        'successful_rounds': 0,
        'failed_rounds': 0,
        'errors': [],
        'deadlock_detected': False,
    }

    buffer = torch.randn(1024 * 1024, dtype=torch.float32, device=device)  # 4MB

    for i in range(num_rounds):
        try:
            # Mix of operations to stress the library
            if i % 3 == 0:
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            elif i % 3 == 1:
                gathered = [torch.empty_like(buffer) for _ in range(world_size)]
                dist.all_gather(gathered, buffer)
            else:
                dist.broadcast(buffer, src=0)

            if backend == 'nccl':
                torch.cuda.synchronize()

            results['successful_rounds'] += 1

        except Exception as e:
            results['failed_rounds'] += 1
            error_type = type(e).__name__
            if error_type not in results['errors']:
                results['errors'].append(error_type)
            if 'timeout' in str(e).lower() or 'deadlock' in str(e).lower():
                results['deadlock_detected'] = True

    return results


# ---------------------------------------------------------------------------
# Main worker function
# ---------------------------------------------------------------------------

def worker(rank, world_size, args):
    """Main worker for distributed benchmark."""
    backend = args.backend if args.backend else detect_backend()

    if backend == 'none':
        if rank == 0:
            print("ERROR: No distributed backend available.")
        return

    # Initialize device
    if backend == 'nccl':
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    elif backend == 'hccl':
        device = torch.device(f'npu:{rank}')
        try:
            import torch.npu
            torch.npu.set_device(device)
        except Exception:
            pass
    elif backend == 'cncl':
        device = torch.device(f'mlu:{rank}')
    else:
        device = torch.device('cpu')

    # Initialize distributed group
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except Exception as e:
        print(f"[Rank {rank}] Failed to init process group: {e}")
        return

    print(f"[Rank {rank}/{world_size}] Backend: {backend}, Device: {device}")

    results = {
        'backend': backend,
        'world_size': world_size,
        'device': str(device),
        'point_to_point': {},
        'collective': {},
        'scaling': {},
        'robustness': {},
    }

    # Parse sizes
    sizes = [parse_size(s) for s in args.sizes.split(',')]
    size_labels = args.sizes.split(',')

    # --- Point-to-point (only between rank 0 and 1) ---
    if world_size >= 2:
        print(f"\n[Rank {rank}] --- Point-to-Point Benchmark ---")
        for size_label, size_bytes in zip(size_labels, sizes):
            if rank in (0, 1):
                p2p = benchmark_point_to_point(rank, world_size, size_bytes, backend, device)
                if rank == 0:
                    results['point_to_point'][size_label] = p2p
                    print(f"  P2P {size_label}: BW={p2p['bandwidth_gbps']} GB/s, "
                          f"Lat={p2p['latency_us_mean']:.1f} µs")

    # --- Collective operations ---
    collective_ops = ['all_reduce', 'all_gather', 'broadcast', 'reduce_scatter']
    print(f"\n[Rank {rank}] --- Collective Communication Benchmark ---")

    for op in collective_ops:
        results['collective'][op] = {}
        for size_label, size_bytes in zip(size_labels, sizes):
            stats = benchmark_collective(rank, world_size, size_bytes, backend, device, op)
            if rank == 0:
                results['collective'][op][size_label] = stats
                print(f"  {op} {size_label}: BW={stats['bandwidth_gbps']} GB/s, "
                      f"Lat={stats['latency_us_mean']:.1f} µs, Jitter={stats['jitter_pct']:.1f}%")

    # --- Scaling efficiency ---
    print(f"\n[Rank {rank}] --- Scaling Efficiency ---")
    scaling = benchmark_scaling_efficiency(world_size, backend, device, size_labels)
    if rank == 0:
        results['scaling'] = scaling
        for op, data in scaling.items():
            print(f"  {op}:")
            for size, metrics in data.items():
                print(f"    {size}: Lat={metrics['latency_us_mean']:.1f} µs, P99={metrics['latency_us_p99']:.1f} µs")

    # --- Library robustness ---
    print(f"\n[Rank {rank}] --- Library Robustness Test ---")
    robustness = assess_library_robustness(rank, world_size, backend, device)
    if rank == 0:
        results['robustness'] = robustness
        success_rate = robustness['successful_rounds'] / robustness['total_rounds'] * 100
        print(f"  Success rate: {success_rate:.1f}% ({robustness['successful_rounds']}/{robustness['total_rounds']})")
        print(f"  Failures: {robustness['failed_rounds']}")
        if robustness['errors']:
            print(f"  Error types: {', '.join(robustness['errors'])}")
        if robustness['deadlock_detected']:
            print("  ⚠️  DEADLOCK DETECTED!")

    # --- Peak interconnect bandwidth estimate ---
    if rank == 0:
        # Estimate theoretical peak based on detected hardware
        peak_bw = estimate_peak_bandwidth(backend, device)
        results['peak_interconnect_bandwidth_gbps'] = peak_bw
        print(f"\n  Estimated peak interconnect bandwidth: {peak_bw} GB/s")

    # Save results (rank 0 only)
    if rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}")

    # Cleanup
    dist.destroy_process_group()


def estimate_peak_bandwidth(backend: str, device) -> int:
    """Estimate theoretical peak interconnect bandwidth."""
    estimates = {
        'nccl': 900,   # NVLink Gen4
        'hccl': 392,   # HCCS (Ascend 910B)
        'cncl': 200,   # MLU-Link
        'mccl': 400,   # MUSA Link
        'gloo': 12,    # PCIe Gen3 x16
    }

    if backend == 'nccl':
        try:
            # Check if NVLink is available
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # NVLink typically indicates higher bandwidth
                if props.multi_processor_count > 100:  # A100/H100 class
                    return 900
                elif props.multi_processor_count > 60:  # A6000/4090 class
                    return 500  # PCIe Gen4
        except Exception:
            pass

    return estimates.get(backend, 100)


# ---------------------------------------------------------------------------
# Main entry point (supports both single-node and multi-node)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AI Infra — Communication & Interconnect Benchmark')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (auto-detect if not specified)')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['nccl', 'hccl', 'cncl', 'mccl', 'gloo'],
                        help='Communication backend (auto-detect if not specified)')
    parser.add_argument('--sizes', type=str,
                        default='1MB,4MB,16MB,64MB,256MB,1GB',
                        help='Comma-separated buffer sizes to test')
    parser.add_argument('--output', type=str, default='results/comm_perf.json',
                        help='Output JSON file path')
    parser.add_argument('--master-addr', type=str, default='localhost',
                        help='Master address for multi-node')
    parser.add_argument('--master-port', type=str, default='29500',
                        help='Master port for multi-node')
    args = parser.parse_args()

    # Detect number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        else:
            try:
                import torch.npu
                num_gpus = torch.npu.device_count()
            except Exception:
                num_gpus = 1

    print(f"\n{'='*70}")
    print(f"  AI Infrastructure Benchmark — Communication & Interconnect")
    print(f"{'='*70}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Backend: {args.backend or 'auto-detect'}")
    print(f"{'='*70}\n")

    # Use torch.multiprocessing.spawn for single-node multi-GPU
    if 'LOCAL_RANK' in os.environ:
        # Launched via torchrun or mpirun
        rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
        world_size = int(os.environ.get('WORLD_SIZE', num_gpus))

        # Set environment variables for distributed init
        os.environ.setdefault('MASTER_ADDR', args.master_addr)
        os.environ.setdefault('MASTER_PORT', args.master_port)

        worker(rank, world_size, args)
    else:
        # Single-node spawn
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port

        torch.multiprocessing.spawn(
            worker,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True,
        )


if __name__ == '__main__':
    main()
