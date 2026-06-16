#!/usr/bin/env python3
"""
Benchmark 1: AI Infrastructure — Operator Performance Evaluation

Compares domestic AI accelerators (Alibaba PPU, Huawei Ascend, Cambricon, Hygon DCU, etc.)
vs NVIDIA GPUs on core deep learning operator latency, throughput, memory bandwidth utilization,
compute-to-memory ratio, instruction-level parallelism, and mixed-precision/quantization support.

Usage:
    python3 01_operator_perf.py [--device cuda|ppu|ascend|cambricon|dcu] \
        [--batch-sizes 1,2,4,8,16,32,64] [--hidden-sizes 4096,8192] \
        [--dtypes bf16,fp16] [--output results/operator_perf.json]

Requirements per device:
    - NVIDIA (cuda):   torch with CUDA, flash-attn (optional)
    - Alibaba PPU:     torch + PPU backend plugin (torch_ppu)
    - Huawei Ascend:   torch + torch_npu (CANN >= 7.0)
    - Cambricon:       torch + torch_mlu (Neuware)
    - Hygon DCU:       torch + ROCm/HIP backend
    - Moore Threads:   torch + musa backend
    - Biren:           torch + biren backend
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Device detection & backend initialization
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect available accelerator backend."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0).lower()
            if 'nvidia' in dev or 'a100' in dev or 'h100' in dev or 'h20' in dev or 'l40' in dev or '4090' in dev:
                return 'cuda'
            # Some domestic chips expose via CUDA-compatible API
        # Try PPU
        try:
            import torch_ppu
            if torch_ppu.is_available():
                return 'ppu'
        except ImportError:
            pass
        # Try Ascend NPU
        try:
            import torch_npu
            if torch.npu.is_available():
                return 'ascend'
        except ImportError:
            pass
        # Try Cambricon MLU
        try:
            import torch_mlu
            if torch_mlu.is_mlu_available():
                return 'cambricon'
        except ImportError:
            pass
        # Try Moore Threads MUSA
        try:
            import torch_musa
            if torch_musa.is_available():
                return 'musa'
        except ImportError:
            pass
        # Try Hygon DCU (ROCm path)
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_built():
            return 'dcu'
    except ImportError:
        pass
    return 'cpu'


def get_device_module(device: str):
    """Return the torch device handle for the given backend."""
    import torch
    mapping = {
        'cuda': 'cuda',
        'ppu': 'ppu',
        'ascend': 'npu',
        'cambricon': 'mlu',
        'dcu': 'cuda',  # DCU uses HIP but often exposes as cuda
        'musa': 'musa',
    }
    return torch.device(mapping.get(device, 'cpu'))


def get_device_info(device: str) -> dict:
    """Collect hardware metadata."""
    info = {'device_type': device, 'device_name': 'unknown', 'memory_total_gb': 0}
    try:
        import torch
        dev = get_device_module(device)
        if device == 'cuda':
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info['compute_capability'] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            info['sm_count'] = torch.cuda.get_device_properties(0).multi_processor_count
            info['memory_bandwidth_gbps'] = round(torch.cuda.get_device_properties(0).memory_clock_rate * 2 * torch.cuda.get_device_properties(0).memory_bus_width / 8 / 1e6 * 2, 0)
        elif device == 'ppu':
            try:
                import torch_ppu
                info['device_name'] = torch_ppu.get_device_name(0)
                info['memory_total_gb'] = round(torch_ppu.get_device_properties(0).total_memory / 1e9, 1)
            except Exception:
                pass
        elif device == 'ascend':
            import torch.npu
            info['device_name'] = torch.npu.get_device_name(0)
            try:
                import torch_npu
                info['memory_total_gb'] = round(torch_npu.get_device_properties(0).total_memory / 1e9, 1)
            except Exception:
                pass
        elif device == 'cambricon':
            try:
                import torch_mlu
                info['device_name'] = torch_mlu.get_device_name(0)
                info['memory_total_gb'] = round(torch_mlu.get_device_properties(0).total_memory / 1e9, 1)
            except Exception:
                pass
        elif device == 'musa':
            try:
                import torch_musa
                info['device_name'] = torch_musa.get_device_name(0)
            except Exception:
                pass
    except Exception as e:
        info['error'] = str(e)
    return info

# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    operator: str
    dtype: str
    batch_size: int
    hidden_size: int
    seq_len: int
    latency_us_mean: float
    latency_us_std: float
    latency_us_p50: float
    latency_us_p99: float
    throughput_tflops: float
    memory_bandwidth_gbps: float
    memory_bandwidth_utilization_pct: float
    calc_mem_ratio: float  # FLOPs per byte
    warmup_runs: int
    measure_runs: int
    notes: str = ""


def benchmark_kernel(fn, args, warmup: int = 5, measure: int = 20, device=None):
    """
    Benchmark a callable `fn(*args)` and return latency stats in microseconds.
    Uses CUDA/PPU events if available, falls back to time.perf_counter.
    """
    import torch

    # Warmup
    for _ in range(warmup):
        fn(*args)

    if device is not None and str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elif device is not None and str(device).startswith('npu'):
        import torch_npu
        torch_npu.synchronize()

    latencies = []
    for _ in range(measure):
        if device is not None and str(device).startswith('cuda'):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end) * 1000)  # ms -> us
        else:
            start = time.perf_counter()
            fn(*args)
            if device is not None:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            latencies.append((time.perf_counter() - start) * 1e6)  # seconds -> us

    import numpy as np
    arr = np.array(latencies)
    return {
        'mean_us': float(np.mean(arr)),
        'std_us': float(np.std(arr)),
        'p50_us': float(np.percentile(arr, 50)),
        'p99_us': float(np.percentile(arr, 99)),
    }


def get_peak_bandwidth_gbps(device: str) -> float:
    """Theoretical peak memory bandwidth in GB/s for common devices."""
    # Default estimates — override with actual hardware specs
    estimates = {
        'cuda': 2000,    # A100 ~2039 GB/s HBM2e
        'ppu': 1200,     # PPU estimate
        'ascend': 1200,  # Ascend 910B ~1200 GB/s
        'cambricon': 800,
        'dcu': 1600,     # DCU Z100
        'musa': 1000,
    }
    try:
        import torch
        if device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            # bandwidth = memory_clock * 2 * bus_width / 8
            bw = props.memory_clock_rate * 2 * props.memory_bus_width / 8 / 1e6  # MHz * bit -> GB/s
            if bw > 100:
                return bw
    except Exception:
        pass
    return estimates.get(device, 500)


# ---------------------------------------------------------------------------
# Operator benchmarks
# ---------------------------------------------------------------------------

def bench_gemm(device, dtype_str, batch, hidden, seq):
    """General Matrix Multiply: the fundamental compute kernel."""
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    M = batch * seq
    K = hidden
    N = hidden

    A = torch.randn(M, K, device=dev, dtype=dtype)
    B = torch.randn(K, N, device=dev, dtype=dtype)
    C = torch.empty(M, N, device=dev, dtype=dtype)

    def kernel():
        torch.mm(A, B, out=C)

    stats = benchmark_kernel(kernel, (), device=dev)

    # FLOPs for GEMM: 2 * M * K * N
    flops = 2 * M * K * N
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12

    # Memory: read A + read B + write C
    bytes_moved = (M * K + K * N + M * N) * dtype.itemsize
    bandwidth_gbps = bytes_moved / latency_s / 1e9
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    return BenchmarkResult(
        operator='GEMM', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


def bench_flash_attention(device, dtype_str, batch, hidden, seq):
    """
    FlashAttention (scaled dot-product attention).
    Tries torch's fused SDPA first, then flash-attn if available.
    """
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    head_dim = 128
    num_heads = hidden // head_dim

    Q = torch.randn(batch, num_heads, seq, head_dim, device=dev, dtype=dtype)
    K = torch.randn(batch, num_heads, seq, head_dim, device=dev, dtype=dtype)
    V = torch.randn(batch, num_heads, seq, head_dim, device=dev, dtype=dtype)

    fa_available = False
    try:
        from flash_attn import flash_attn_func
        fa_available = True
    except ImportError:
        pass

    if fa_available:
        def kernel():
            flash_attn_func(Q, K, V, dropout_p=0.0, causal=True)
    else:
        def kernel():
            torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    stats = benchmark_kernel(kernel, (), device=dev)

    # FLOPs for attention: ~4 * B * num_heads * seq^2 * head_dim (forward, causal ~half)
    flops = 4 * batch * num_heads * seq * seq * head_dim * 0.5
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12

    # Memory: read QKV + write O
    bytes_moved = 3 * (batch * num_heads * seq * head_dim * dtype.itemsize) + \
                  (batch * num_heads * seq * head_dim * dtype.itemsize)
    bandwidth_gbps = bytes_moved / latency_s / 1e9
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    impl = 'flash_attn' if fa_available else 'sdpa_fused'
    return BenchmarkResult(
        operator=f'FlashAttention ({impl})', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


def bench_layer_norm(device, dtype_str, batch, hidden, seq):
    """LayerNorm: common in Transformer encoder/decoder blocks."""
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    x = torch.randn(batch, seq, hidden, device=dev, dtype=dtype)
    ln = torch.nn.LayerNorm(hidden, device=dev, dtype=dtype)

    def kernel():
        ln(x)

    stats = benchmark_kernel(kernel, (), device=dev)

    # FLOPs: ~5 * B * seq * hidden (mean, var, normalize, scale, shift)
    flops = 5 * batch * seq * hidden
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12 if latency_s > 0 else 0

    # Memory: read x + write output + read weight+bias
    bytes_moved = (batch * seq * hidden * dtype.itemsize * 2) + (hidden * dtype.itemsize * 2)
    bandwidth_gbps = bytes_moved / latency_s / 1e9 if latency_s > 0 else 0
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    return BenchmarkResult(
        operator='LayerNorm', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


def bench_rms_norm(device, dtype_str, batch, hidden, seq):
    """RMSNorm: popular in LLaMA-family models."""
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    x = torch.randn(batch, seq, hidden, device=dev, dtype=dtype)
    weight = torch.ones(hidden, device=dev, dtype=dtype)
    eps = 1e-6

    def rms_norm(x, weight, eps):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return norm * weight

    def kernel():
        rms_norm(x, weight, eps)

    stats = benchmark_kernel(kernel, (), device=dev)

    flops = 4 * batch * seq * hidden  # pow2, mean, rsqrt, mul
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12 if latency_s > 0 else 0

    bytes_moved = 2 * batch * seq * hidden * dtype.itemsize + hidden * dtype.itemsize
    bandwidth_gbps = bytes_moved / latency_s / 1e9 if latency_s > 0 else 0
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    return BenchmarkResult(
        operator='RMSNorm', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


def bench_rope(device, dtype_str, batch, hidden, seq):
    """Rotary Position Embedding (RoPE): used in most modern LLMs."""
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    head_dim = 128
    num_heads = hidden // head_dim

    x = torch.randn(batch, num_heads, seq, head_dim, device=dev, dtype=dtype)
    cos = torch.randn(seq, head_dim, device=dev, dtype=dtype)
    sin = torch.randn(seq, head_dim, device=dev, dtype=dtype)

    def apply_rope(x, cos, sin):
        x_cos = x * cos - x[..., 1::2] * sin
        x_sin = x[..., 1::2] * cos + x[..., ::2] * sin
        result = x.clone()
        result[..., ::2] = x_cos
        result[..., 1::2] = x_sin
        return result

    def kernel():
        apply_rope(x, cos, sin)

    stats = benchmark_kernel(kernel, (), device=dev)

    flops = 6 * batch * num_heads * seq * head_dim  # mul, mul, add, mul, add, assign
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12 if latency_s > 0 else 0

    bytes_moved = (batch * num_heads * seq * head_dim * 2 + seq * head_dim * 2) * dtype.itemsize
    bandwidth_gbps = bytes_moved / latency_s / 1e9 if latency_s > 0 else 0
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    return BenchmarkResult(
        operator='RoPE', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


def bench_swiglu(device, dtype_str, batch, hidden, seq):
    """SwiGLU activation: common in LLaMA, Mixtral, etc."""
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    x = torch.randn(batch, seq, hidden, device=dev, dtype=dtype)
    w1 = torch.randn(hidden, hidden, device=dev, dtype=dtype)
    w3 = torch.randn(hidden, hidden, device=dev, dtype=dtype)
    w2 = torch.randn(hidden, hidden, device=dev, dtype=dtype)

    def swiglu(x, w1, w2, w3):
        gate = torch.nn.functional.silu(x @ w1)
        up = x @ w3
        return (gate * up) @ w2

    def kernel():
        swiglu(x, w1, w2, w3)

    stats = benchmark_kernel(kernel, (), device=dev)

    # 3 GEMMs + SiLU + element-wise mul
    flops = 3 * (2 * batch * seq * hidden * hidden) + 4 * batch * seq * hidden
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12 if latency_s > 0 else 0

    bytes_moved = 3 * (batch * seq * hidden + hidden * hidden) * dtype.itemsize + \
                  batch * seq * hidden * dtype.itemsize
    bandwidth_gbps = bytes_moved / latency_s / 1e9 if latency_s > 0 else 0
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    return BenchmarkResult(
        operator='SwiGLU', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


def bench_softmax(device, dtype_str, batch, hidden, seq):
    """Softmax: attention output normalization."""
    import torch
    dev = get_device_module(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    x = torch.randn(batch, seq, hidden, device=dev, dtype=dtype)
    sm = torch.nn.Softmax(dim=-1)

    def kernel():
        sm(x)

    stats = benchmark_kernel(kernel, (), device=dev)

    flops = 4 * batch * seq * hidden  # exp, sum, div
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12 if latency_s > 0 else 0

    bytes_moved = 2 * batch * seq * hidden * dtype.itemsize
    bandwidth_gbps = bytes_moved / latency_s / 1e9 if latency_s > 0 else 0
    peak_bw = get_peak_bandwidth_gbps(device)
    bw_util = bandwidth_gbps / peak_bw * 100 if peak_bw > 0 else 0

    return BenchmarkResult(
        operator='Softmax', dtype=dtype_str, batch_size=batch,
        hidden_size=hidden, seq_len=seq,
        latency_us_mean=stats['mean_us'], latency_us_std=stats['std_us'],
        latency_us_p50=stats['p50_us'], latency_us_p99=stats['p99_us'],
        throughput_tflops=tflops, memory_bandwidth_gbps=bandwidth_gbps,
        memory_bandwidth_utilization_pct=bw_util,
        calc_mem_ratio=flops / bytes_moved if bytes_moved > 0 else 0,
        warmup_runs=5, measure_runs=20,
    )


# ---------------------------------------------------------------------------
# Quantization support assessment
# ---------------------------------------------------------------------------

def assess_quantization_support(device: str) -> dict:
    """Test hardware acceleration for FP8, INT8, INT4 quantization formats."""
    import torch
    dev = get_device_module(device)
    results = {
        'fp32': {'supported': True, 'hw_accelerated': True},
        'fp16': {'supported': True, 'hw_accelerated': True},
        'bf16': {'supported': True, 'hw_accelerated': True},
        'fp8_e4m3': {'supported': False, 'hw_accelerated': False},
        'fp8_e5m2': {'supported': False, 'hw_accelerated': False},
        'int8': {'supported': False, 'hw_accelerated': False},
        'int4': {'supported': False, 'hw_accelerated': False},
    }

    # Test FP8
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            x = torch.randn(256, 256, device=dev, dtype=torch.float8_e4m3fn)
            y = torch.randn(256, 256, device=dev, dtype=torch.float8_e4m3fn)
            torch.mm(x, y)
            results['fp8_e4m3']['supported'] = True
            results['fp8_e4m3']['hw_accelerated'] = True
        except Exception:
            pass

    if hasattr(torch, 'float8_e5m2'):
        try:
            x = torch.randn(256, 256, device=dev, dtype=torch.float8_e5m2)
            results['fp8_e5m2']['supported'] = True
        except Exception:
            pass

    # Test INT8
    try:
        x = torch.randint(-128, 127, (256, 256), device=dev, dtype=torch.int8)
        y = torch.randint(-128, 127, (256, 256), device=dev, dtype=torch.int8)
        if hasattr(torch, '_int_mm'):
            torch._int_mm(x, y)
            results['int8']['hw_accelerated'] = True
        results['int8']['supported'] = True
    except Exception:
        results['int8']['supported'] = 'partial'

    # Test INT4 (if available)
    try:
        if hasattr(torch, 'quint4x2'):
            results['int4']['supported'] = True
    except Exception:
        pass

    # Device-specific notes
    if device == 'cuda':
        try:
            import torch
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 9:  # Hopper
                results['fp8_e4m3']['supported'] = True
                results['fp8_e4m3']['hw_accelerated'] = True
                results['fp8_e5m2']['supported'] = True
            if cc[0] >= 8:  # Ampere+
                results['int8']['hw_accelerated'] = True
        except Exception:
            pass
    elif device == 'ascend':
        results['int8']['supported'] = True
        results['int8']['hw_accelerated'] = True  # Ascend has INT8 matrix units
    elif device == 'ppu':
        results['int8']['supported'] = True
        results['int8']['hw_accelerated'] = True

    return results


# ---------------------------------------------------------------------------
# SIMD / instruction-level parallelism assessment
# ---------------------------------------------------------------------------

def assess_ilp(device: str) -> dict:
    """
    Estimate instruction-level parallelism by running independent ops
    concurrently and measuring speedup vs sequential execution.
    """
    import torch
    dev = get_device_module(device)
    dtype = torch.float16

    sizes = [(1024, 1024), (2048, 2048)]
    results = {}

    for M, K in sizes:
        A = torch.randn(M, K, device=dev, dtype=dtype)
        B = torch.randn(K, M, device=dev, dtype=dtype)
        C = torch.randn(M, K, device=dev, dtype=dtype)
        D = torch.randn(K, M, device=dev, dtype=dtype)

        # Sequential
        def seq():
            r1 = A @ B
            r2 = C @ D
            return r1 + r2

        seq_stats = benchmark_kernel(seq, (), device=dev)

        # Parallel (two streams if supported)
        par_time = None
        try:
            if str(dev).startswith('cuda'):
                s1 = torch.cuda.Stream()
                s2 = torch.cuda.Stream()

                def par():
                    with torch.cuda.stream(s1):
                        r1 = A @ B
                    with torch.cuda.stream(s2):
                        r2 = C @ D
                    torch.cuda.synchronize()
                    return r1 + r2

                par_stats = benchmark_kernel(par, (), device=dev)
                par_time = par_stats['mean_us']
        except Exception:
            pass

        speedup = seq_stats['mean_us'] / par_time if par_time and par_time > 0 else 1.0

        results[f'{M}x{K}'] = {
            'sequential_us': seq_stats['mean_us'],
            'parallel_us': par_time,
            'speedup': round(speedup, 3),
            'simd_efficiency': round(min(speedup / 2.0, 1.0) * 100, 1),
        }

    return results


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_benchmarks(args):
    import torch
    print(f"\n{'='*70}")
    print(f"  AI Infrastructure Benchmark — Operator Performance")
    print(f"{'='*70}\n")

    device = args.device if args.device else detect_device()
    print(f"Detected device: {device}")

    dev_info = get_device_info(device)
    print(f"Device info: {json.dumps(dev_info, indent=2, ensure_ascii=False)}\n")

    bench_fn_map = {
        'gemm': bench_gemm,
        'flash_attention': bench_flash_attention,
        'layer_norm': bench_layer_norm,
        'rms_norm': bench_rms_norm,
        'rope': bench_rope,
        'swiglu': bench_swiglu,
        'softmax': bench_softmax,
    }

    operators = args.operators.split(',') if args.operators else list(bench_fn_map.keys())
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
    dtypes = args.dtypes.split(',')
    seq_len = args.seq_len

    all_results = []

    for op_name in operators:
        if op_name not in bench_fn_map:
            print(f"  [SKIP] Unknown operator: {op_name}")
            continue

        bench_fn = bench_fn_map[op_name]
        print(f"\n--- Benchmark: {op_name} ---")

        for dtype_str in dtypes:
            for batch in batch_sizes:
                for hidden in hidden_sizes:
                    label = f"{op_name} | {dtype_str} | B={batch} | H={hidden} | S={seq_len}"
                    print(f"  Running: {label}", end=' ', flush=True)
                    try:
                        result = bench_fn(device, dtype_str, batch, hidden, seq_len)
                        all_results.append(result)
                        print(f"✅ {result.latency_us_mean:.1f} µs, {result.throughput_tflops:.2f} TFLOPS, "
                              f"BW={result.memory_bandwidth_utilization_pct:.1f}%")
                    except Exception as e:
                        print(f"❌ {type(e).__name__}: {e}")
                        all_results.append(BenchmarkResult(
                            operator=op_name, dtype=dtype_str, batch_size=batch,
                            hidden_size=hidden, seq_len=seq_len,
                            latency_us_mean=-1, latency_us_std=-1,
                            latency_us_p50=-1, latency_us_p99=-1,
                            throughput_tflops=-1, memory_bandwidth_gbps=-1,
                            memory_bandwidth_utilization_pct=-1,
                            calc_mem_ratio=-1, warmup_runs=0, measure_runs=0,
                            notes=f"Error: {type(e).__name__}: {e}",
                        ))

    # Quantization assessment
    print(f"\n--- Quantization Support Assessment ---")
    quant_results = assess_quantization_support(device)
    for fmt, info in quant_results.items():
        status = "✅ HW accelerated" if info.get('hw_accelerated') else ("✅ Supported" if info.get('supported') else "❌ Not supported")
        print(f"  {fmt}: {status}")

    # ILP assessment
    print(f"\n--- Instruction-Level Parallelism Assessment ---")
    ilp_results = assess_ilp(device)
    for shape, info in ilp_results.items():
        print(f"  {shape}: seq={info['sequential_us']:.1f}µs, par={info['parallel_us']:.1f if info['parallel_us'] else 'N/A'}µs, "
              f"speedup={info['speedup']:.2f}x, SIMD eff={info['simd_efficiency']:.1f}%")

    # Compile final report
    report = {
        'device_info': dev_info,
        'benchmark_config': {
            'operators': operators,
            'batch_sizes': batch_sizes,
            'hidden_sizes': hidden_sizes,
            'dtypes': dtypes,
            'seq_len': seq_len,
        },
        'operator_results': [asdict(r) for r in all_results],
        'quantization_support': quant_results,
        'ilp_assessment': ilp_results,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Infra — Operator Performance Benchmark')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'ppu', 'ascend', 'cambricon', 'dcu', 'musa'],
                        help='Device backend (auto-detect if not specified)')
    parser.add_argument('--operators', type=str, default=None,
                        help='Comma-separated operator names (default: all)')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16,32',
                        help='Comma-separated batch sizes')
    parser.add_argument('--hidden-sizes', type=str, default='4096,8192',
                        help='Comma-separated hidden dimensions')
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Sequence length')
    parser.add_argument('--dtypes', type=str, default='bf16,fp16',
                        help='Comma-separated data types')
    parser.add_argument('--output', type=str, default='results/operator_perf.json',
                        help='Output JSON file path')
    args = parser.parse_args()
    run_benchmarks(args)
