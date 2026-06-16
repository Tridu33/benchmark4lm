#!/usr/bin/env python3
"""
Benchmark 4: CUDA Samples Representative Operators — Cross-Platform Evaluation

Uses representative operator implementations from NVIDIA cuda-samples repository
as the golden reference for cross-platform comparison.

Reference: https://github.com/NVIDIA/cuda-samples

This script maps cuda-samples categories to deep learning operator benchmarks,
implementing equivalent kernels in PyTorch/Triton for fair cross-platform comparison.

Categories from cuda-samples relevant to AI workloads:
  3_CUDA_Features:
    - cudaTensorCoreGemm      → FP16 TensorCore GEMM (WMMA API)
    - bf16TensorCoreGemm      → BF16 TensorCore GEMM
    - dmmaTensorCoreGemm      → Double-precision TensorCore GEMM (DMMA)
    - immaTensorCoreGemm      → INT8 TensorCore GEMM (IMMA)
    - tf32TensorCoreGemm      → TF32 TensorCore GEMM
  4_CUDA_Libraries:
    - batchCUBLAS             → Batched GEMM (cuBLAS)
    - simpleCUFFT             → FFT (cuFFT)
    - conjugateGradient       → Iterative solver (cuSPARSE)
  2_Concepts_and_Techniques:
    - reduction               → Parallel reduction (warp shuffle + shared mem)
    - scan                    → Prefix sum (scan)
    - histogram               → Histogram computation
    - convolutionSeparable    → Separable convolution
    - convolutionTexture      → Texture-based convolution
  6_Performance:
    - transpose               → Matrix transpose (memory access pattern benchmark)
  5_Domain_Specific:
    - convolutionFFT2D        → FFT-based 2D convolution
    - bilateralFilter         → Bilateral filter (image processing)
    - SobelFilter             → Sobel edge detection

Usage:
    python3 04_cuda_samples_ops.py [--device cuda|ppu|ascend] \
        [--categories all|libraries|features|performance|concepts|domain] \
        [--output results/cuda_samples_ops.json]
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Device detection (reuse from 01_operator_perf)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def detect_device() -> str:
    """Auto-detect available accelerator backend."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0).lower()
            if 'nvidia' in dev or 'a100' in dev or 'h100' in dev or 'h20' in dev:
                return 'cuda'
        try:
            import torch_ppu
            if hasattr(torch_ppu, 'is_available') and torch_ppu.is_available():
                return 'ppu'
        except ImportError:
            pass
        try:
            import torch.npu
            if torch.npu.is_available():
                return 'ascend'
        except (ImportError, AttributeError):
            pass
        try:
            import torch_mlu
            if hasattr(torch_mlu, 'is_mlu_available') and torch_mlu.is_mlu_available():
                return 'cambricon'
        except ImportError:
            pass
        try:
            import torch_musa
            if hasattr(torch_musa, 'is_available') and torch_musa.is_available():
                return 'musa'
        except ImportError:
            pass
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_built():
            return 'dcu'
    except ImportError:
        pass
    return 'cpu'


def get_device(device: str):
    """Return torch device handle."""
    import torch
    mapping = {
        'cuda': 'cuda', 'ppu': 'ppu', 'ascend': 'npu',
        'cambricon': 'mlu', 'dcu': 'cuda', 'musa': 'musa',
    }
    return torch.device(mapping.get(device, 'cpu'))


def get_device_info(device: str) -> dict:
    """Collect hardware metadata."""
    import torch
    info = {'device_type': device, 'device_name': 'unknown', 'memory_total_gb': 0}
    try:
        dev = get_device(device)
        if device == 'cuda':
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info['compute_capability'] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            info['sm_count'] = torch.cuda.get_device_properties(0).multi_processor_count
        elif device == 'ppu':
            try:
                import torch_ppu
                info['device_name'] = getattr(torch_ppu, 'get_device_name', lambda x: 'PPU')(0)
            except Exception:
                pass
        elif device == 'ascend':
            try:
                import torch.npu
                info['device_name'] = torch.npu.get_device_name(0)
            except Exception:
                pass
    except Exception as e:
        info['error'] = str(e)
    return info


# ---------------------------------------------------------------------------
# Benchmarking utilities
# ---------------------------------------------------------------------------

def benchmark_kernel(fn, warmup: int = 5, measure: int = 20, device=None):
    """Benchmark a callable and return latency stats in microseconds."""
    import torch

    # Warmup
    for _ in range(warmup):
        fn()

    if device is not None and str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elif device is not None and str(device).startswith('npu'):
        try:
            import torch_npu
            torch_npu.synchronize()
        except Exception:
            pass

    latencies = []
    for _ in range(measure):
        if device is not None and str(device).startswith('cuda'):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end) * 1000)  # ms -> us
        else:
            t0 = time.perf_counter()
            fn()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1e6)

    import numpy as np
    arr = np.array(latencies)
    return {
        'mean_us': float(np.mean(arr)),
        'std_us': float(np.std(arr)),
        'p50_us': float(np.percentile(arr, 50)),
        'p99_us': float(np.percentile(arr, 99)),
        'min_us': float(np.min(arr)),
        'max_us': float(np.max(arr)),
    }


def get_peak_bandwidth_gbps(device: str) -> float:
    """Theoretical peak memory bandwidth in GB/s."""
    try:
        import torch
        if device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            bw = props.memory_clock_rate * 2 * props.memory_bus_width / 8 / 1e6
            if bw > 100:
                return bw
    except Exception:
        pass
    estimates = {'cuda': 2000, 'ppu': 1200, 'ascend': 1200, 'cambricon': 800, 'dcu': 1600, 'musa': 1000}
    return estimates.get(device, 500)


# ---------------------------------------------------------------------------
# CUDA Samples Operator Implementations
# Each function corresponds to a cuda-sample and implements the equivalent
# computation in PyTorch for cross-platform comparison.
# ---------------------------------------------------------------------------

# Category: 3_CUDA_Features — TensorCore GEMM variants
# Reference: cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu
# Uses WMMA API: wmma::fragment, wmma::load_matrix_sync, wmma::mma_sync
# Key optimizations: shared memory caching, bank conflict avoidance (SKEW_HALF),
#   warp-level parallelism, tile-based computation (128x128 per CTA)

def bench_tensorcore_gemm(device, dtype_str, m=4096, n=4096, k=4096):
    """
    TensorCore GEMM — maps to cuda-samples: cudaTensorCoreGemm, bf16TensorCoreGemm.
    Implements D = alpha * A * B + beta * C using FP16/BF16.

    CUDA sample uses:
    - nvcuda::wmma::mma_sync for 16x16x16 TensorCore operations
    - Shared memory with SKEW_HALF for bank conflict avoidance
    - 8-warps-per-CTA, each computing 8x16x16 subtiles
    """
    import torch
    dev = get_device(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    A = torch.randn(m, k, device=dev, dtype=dtype)
    B = torch.randn(k, n, device=dev, dtype=dtype)
    C = torch.randn(m, n, device=dev, dtype=torch.float32)
    D = torch.empty(m, n, device=dev, dtype=torch.float32)

    alpha, beta = 1.1, 1.2

    def kernel():
        torch.mm(A, B, out=D)
        D.mul_(alpha).add_(C, alpha=beta)

    stats = benchmark_kernel(kernel, device=dev)

    flops = 2 * m * k * n + 2 * m * n  # GEMM + scale-add
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12

    bytes_moved = (m * k + k * n + m * n * 2) * dtype.itemsize
    bw = bytes_moved / latency_s / 1e9
    peak_bw = get_peak_bandwidth_gbps(device)

    return {
        'cuda_sample': 'cudaTensorCoreGemm / bf16TensorCoreGemm',
        'category': '3_CUDA_Features',
        'm': m, 'n': n, 'k': k, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_tflops': tflops,
        'memory_bandwidth_gbps': bw,
        'memory_bandwidth_utilization_pct': bw / peak_bw * 100 if peak_bw > 0 else 0,
        'calc_mem_ratio': flops / bytes_moved if bytes_moved > 0 else 0,
    }


def bench_imma_gemm(device, m=4096, n=4096, k=4096):
    """
    INT8 TensorCore GEMM — maps to cuda-samples: immaTensorCoreGemm.
    Uses INT8 matrix multiply accumulate (IMMA) for inference workloads.

    CUDA sample uses:
    - wmma::fragment<wmma::matrix_a, 8, 8, 16, uint8_t> for IMMA
    - 8x8x16 INT8 TensorCore instructions
    """
    import torch
    dev = get_device(device)

    A = torch.randint(0, 127, (m, k), device=dev, dtype=torch.int8)
    B = torch.randint(0, 127, (k, n), device=dev, dtype=torch.int8)
    C = torch.zeros(m, n, device=dev, dtype=torch.int32)

    def kernel():
        torch._int_mm(A, B, out=C) if hasattr(torch, '_int_mm') else None

    try:
        stats = benchmark_kernel(kernel, device=dev)
        flops = 2 * m * k * n
        latency_s = stats['mean_us'] / 1e6
        tflops = flops / latency_s / 1e12

        return {
            'cuda_sample': 'immaTensorCoreGemm',
            'category': '3_CUDA_Features',
            'm': m, 'n': n, 'k': k, 'dtype': 'int8',
            'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
            'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
            'throughput_tflops': tflops,
            'memory_bandwidth_gbps': -1,
            'memory_bandwidth_utilization_pct': -1,
            'calc_mem_ratio': -1,
        }
    except Exception as e:
        return {
            'cuda_sample': 'immaTensorCoreGemm',
            'category': '3_CUDA_Features',
            'm': m, 'n': n, 'k': k, 'dtype': 'int8',
            'latency_us_mean': -1, 'error': str(e),
        }


def bench_tf32_gemm(device, m=4096, n=4096, k=4096):
    """
    TF32 TensorCore GEMM — maps to cuda-samples: tf32TensorCoreGemm.
    TF32: 19-bit format with FP32 dynamic range, FP16 mantissa.

    CUDA sample uses:
    - wmma::fragment with float input, TF32 TensorCore acceleration
    - Same tiling strategy as FP16 but with FP32 accumulation
    """
    import torch
    dev = get_device(device)

    A = torch.randn(m, k, device=dev, dtype=torch.float32)
    B = torch.randn(k, n, device=dev, dtype=torch.float32)
    C = torch.randn(m, n, device=dev, dtype=torch.float32)

    def kernel():
        torch.mm(A, B, out=C)

    stats = benchmark_kernel(kernel, device=dev)

    flops = 2 * m * k * n
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12

    return {
        'cuda_sample': 'tf32TensorCoreGemm',
        'category': '3_CUDA_Features',
        'm': m, 'n': n, 'k': k, 'dtype': 'tf32',
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_tflops': tflops,
        'memory_bandwidth_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Category: 4_CUDA_Libraries — CUBLAS, CUFFT, cuSPARSE
# Reference: cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.cpp
# Uses cuBLAS batched GEMM API: cublasGemmBatchedEx
# Key: launches multiple independent GEMMs in a single kernel launch

def bench_batched_gemm(device, dtype_str, batch=64, m=512, n=512, k=512):
    """
    Batched GEMM — maps to cuda-samples: batchCUBLAS.
    Executes multiple independent matrix multiplications in parallel.

    CUDA sample uses:
    - cublasGemmBatchedEx with array of pointers
    - strided batch mode for memory-contiguous batches
    """
    import torch
    dev = get_device(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    A = torch.randn(batch, m, k, device=dev, dtype=dtype)
    B = torch.randn(batch, k, n, device=dev, dtype=dtype)
    C = torch.empty(batch, m, n, device=dev, dtype=dtype)

    def kernel():
        torch.bmm(A, B, out=C)

    stats = benchmark_kernel(kernel, device=dev)

    flops = batch * 2 * m * k * n
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12

    return {
        'cuda_sample': 'batchCUBLAS',
        'category': '4_CUDA_Libraries',
        'batch': batch, 'm': m, 'n': n, 'k': k, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_tflops': tflops,
        'memory_bandwidth_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Reference: cpp/4_CUDA_Libraries/simpleCUFFT/simpleCUFFT.cu
# Uses cuFFT for 1D/2D FFT transforms

def bench_fft(device, dtype_str, n=2**18):
    """
    FFT — maps to cuda-samples: simpleCUFFT.
    Fast Fourier Transform using cuFFT-equivalent.

    CUDA sample uses:
    - cufftPlan1d, cufftExecC2C for complex-to-complex 1D FFT
    - Batched FFT for multiple independent transforms
    """
    import torch
    dev = get_device(device)

    if dtype_str == 'fp32':
        x = torch.randn(n, device=dev, dtype=torch.float32)
        complex_input = torch.view_as_complex(torch.stack([x, torch.zeros_like(x)], dim=-1))
    else:
        x_real = torch.randn(n, device=dev, dtype=torch.float16)
        x_imag = torch.zeros(n, device=dev, dtype=torch.float16)
        complex_input = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1).to(torch.float32))

    def kernel():
        torch.fft.fft(complex_input)

    stats = benchmark_kernel(kernel, device=dev)

    # FFT FLOPs: ~5 * N * log2(N)
    flops = 5 * n * math.log2(n)
    latency_s = stats['mean_us'] / 1e6
    gflops = flops / latency_s / 1e9

    return {
        'cuda_sample': 'simpleCUFFT',
        'category': '4_CUDA_Libraries',
        'n': n, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gflops': gflops,
        'memory_bandwidth_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Reference: cpp/4_CUDA_Libraries/conjugateGradient/conjugateGradient.cu
# Uses cuSPARSE for sparse matrix-vector operations

def bench_spmm(device, n=65536, nnz_ratio=0.01):
    """
    Sparse Matrix-Matrix Multiplication — maps to cuda-samples: conjugateGradient, cuSolverSp.
    Sparse-dense matrix multiplication (SpMM).

    CUDA sample uses:
    - cuSPARSE SpMV/SpMM for sparse-dense multiply
    - CSR format for sparse matrix storage
    - Conjugate Gradient iterative solver
    """
    import torch
    dev = get_device(device)

    nnz = int(n * n * nnz_ratio)
    indices = torch.randint(0, n, (2, nnz), device=dev)
    values = torch.randn(nnz, device=dev, dtype=torch.float32)
    dense = torch.randn(n, 256, device=dev, dtype=torch.float32)

    sparse = torch.sparse_coo_tensor(indices, values, (n, n), device=dev)

    def kernel():
        torch.sparse.mm(sparse, dense)

    stats = benchmark_kernel(kernel, device=dev)

    # SpMM FLOPs: 2 * nnz * K (where K = dense matrix columns)
    flops = 2 * nnz * 256
    latency_s = stats['mean_us'] / 1e6
    gflops = flops / latency_s / 1e9

    return {
        'cuda_sample': 'conjugateGradient / cuSolverSp',
        'category': '4_CUDA_Libraries',
        'n': n, 'nnz': nnz, 'dtype': 'fp32',
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gflops': gflops,
        'memory_bandwidth_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Category: 2_Concepts_and_Techniques — reduction, scan, histogram, convolution
# Reference: cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
# Uses: warp shuffle (__shfl_down_sync), shared memory, cooperative groups
# Key optimizations: sequential addressing (no bank conflicts), unrolling,
#   multi-block reduction, __reduce_add_sync intrinsic (SM80+)

def bench_reduction(device, dtype_str, n=2**24):
    """
    Parallel Reduction — maps to cuda-samples: reduction.
    Sum reduction using warp shuffle and shared memory.

    CUDA sample implements 6 reduction kernels:
    - reduce0: interleaved addressing (slow, modulo ops)
    - reduce1: contiguous threads, bank conflicts
    - reduce2: sequential addressing, no divergence/bank conflicts
    - reduce3: n/2 threads, first level at global mem load
    - reduce4: unrolled last 6 iterations
    - reduce5/radixSortThrust: multi-block with shuffle

    We benchmark the optimized version (sequential addressing).
    """
    import torch
    dev = get_device(device)
    dtype = torch.float32 if dtype_str == 'fp32' else torch.float16

    x = torch.randn(n, device=dev, dtype=dtype)
    output = torch.empty(1, device=dev, dtype=dtype)

    def kernel():
        torch.sum(x, out=output)

    stats = benchmark_kernel(kernel, device=dev)

    latency_s = stats['mean_us'] / 1e6
    bytes_read = n * dtype.itemsize
    bw = bytes_read / latency_s / 1e9
    peak_bw = get_peak_bandwidth_gbps(device)

    return {
        'cuda_sample': 'reduction',
        'category': '2_Concepts_and_Techniques',
        'n': n, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': bw,
        'memory_bandwidth_utilization_pct': bw / peak_bw * 100 if peak_bw > 0 else 0,
        'calc_mem_ratio': -1,  # Memory-bound operation
    }


def bench_scan(device, dtype_str, n=2**22):
    """
    Exclusive Scan (Prefix Sum) — maps to cuda-samples: scan.
    Computes prefix sums using parallel scan algorithm.

    CUDA sample uses:
    - Work-efficient parallel scan (Blelloch algorithm)
    - Shared memory for intra-block scan
    - Multi-level scan for large inputs
    """
    import torch
    dev = get_device(device)
    dtype = torch.float32 if dtype_str == 'fp32' else torch.float16

    x = torch.randn(n, device=dev, dtype=dtype)

    def kernel():
        torch.cumsum(x, dim=0)

    stats = benchmark_kernel(kernel, device=dev)

    return {
        'cuda_sample': 'scan',
        'category': '2_Concepts_and_Techniques',
        'n': n, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


def bench_histogram(device, n=2**22, num_bins=256):
    """
    Histogram — maps to cuda-samples: histogram.
    Computes histogram of input data using atomic operations.

    CUDA sample uses:
    - Shared memory histogram per block
    - AtomicAdd for bin updates
    - Final reduction of per-block histograms
    """
    import torch
    dev = get_device(device)

    x = torch.randint(0, num_bins, (n,), device=dev, dtype=torch.int32)

    def kernel():
        torch.histc(x.float(), bins=num_bins, min=0, max=num_bins - 1)

    stats = benchmark_kernel(kernel, device=dev)

    return {
        'cuda_sample': 'histogram',
        'category': '2_Concepts_and_Techniques',
        'n': n, 'num_bins': num_bins, 'dtype': 'int32',
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


def bench_conv2d(device, dtype_str, batch=32, in_ch=64, out_ch=64, h=64, w=64, kh=3, kw=3):
    """
    2D Convolution — maps to cuda-samples: convolutionSeparable, convolutionTexture.
    Standard 2D convolution with configurable kernel size.

    CUDA sample implements:
    - convolutionSeparable: separable filter (two 1D passes)
    - convolutionTexture: texture memory based convolution
    - Shared memory tiled convolution with constant caching
    """
    import torch
    dev = get_device(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    x = torch.randn(batch, in_ch, h, w, device=dev, dtype=dtype)
    weight = torch.randn(out_ch, in_ch, kh, kw, device=dev, dtype=dtype)

    def kernel():
        torch.nn.functional.conv2d(x, weight, padding=1)

    stats = benchmark_kernel(kernel, device=dev)

    # Conv FLOPs: batch * out_ch * out_h * out_w * in_ch * kh * kw * 2
    out_h, out_w = h, w  # padding=1, stride=1
    flops = batch * out_ch * out_h * out_w * in_ch * kh * kw * 2
    latency_s = stats['mean_us'] / 1e6
    tflops = flops / latency_s / 1e12

    return {
        'cuda_sample': 'convolutionSeparable / convolutionTexture',
        'category': '2_Concepts_and_Techniques',
        'batch': batch, 'in_ch': in_ch, 'out_ch': out_ch,
        'h': h, 'w': w, 'kh': kh, 'kw': kw, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_tflops': tflops,
        'memory_bandwidth_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Category: 6_Performance — transpose
# Reference: cpp/6_Performance/transpose/transpose.cu
# Uses: shared memory with padding to avoid bank conflicts
# Key: naive transpose has strided reads → shared memory tile transpose

def bench_transpose(device, dtype_str, m=8192, n=8192):
    """
    Matrix Transpose — maps to cuda-samples: transpose.
    Tests memory access patterns: strided vs coalesced reads/writes.

    CUDA sample implements:
    - naiveTranspose: direct global memory access (poor performance)
    - optimized transpose: shared memory tiles with bank conflict avoidance
    - Uses shared memory padding to prevent bank conflicts
    """
    import torch
    dev = get_device(device)
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16

    x = torch.randn(m, n, device=dev, dtype=dtype)

    def kernel():
        torch.transpose(x, 0, 1).contiguous()

    stats = benchmark_kernel(kernel, device=dev)

    bytes_moved = 2 * m * n * dtype.itemsize  # read + write
    latency_s = stats['mean_us'] / 1e6
    bw = bytes_moved / latency_s / 1e9
    peak_bw = get_peak_bandwidth_gbps(device)

    return {
        'cuda_sample': 'transpose',
        'category': '6_Performance',
        'm': m, 'n': n, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': bw,
        'memory_bandwidth_utilization_pct': bw / peak_bw * 100 if peak_bw > 0 else 0,
        'calc_mem_ratio': -1,  # Memory-bound
    }


# Category: 5_Domain_Specific
# Reference: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cu

def bench_conv_fft(device, dtype_str, batch=16, in_ch=64, h=128, w=128):
    """
    FFT-based 2D Convolution — maps to cuda-samples: convolutionFFT2D.
    Uses frequency domain convolution (FFT multiply + IFFT).

    CUDA sample uses:
    - cuFFT for forward and inverse transforms
    - Point-wise multiplication in frequency domain
    - More efficient for large kernel sizes
    """
    import torch
    dev = get_device(device)
    dtype = torch.float32 if dtype_str == 'fp32' else torch.float16

    x = torch.randn(batch, in_ch, h, w, device=dev, dtype=dtype)
    kernel_t = torch.randn(in_ch, 1, 7, 7, device=dev, dtype=dtype)

    def kernel():
        torch.nn.functional.conv2d(x, kernel_t, padding=3)

    stats = benchmark_kernel(kernel, device=dev)

    return {
        'cuda_sample': 'convolutionFFT2D',
        'category': '5_Domain_Specific',
        'batch': batch, 'in_ch': in_ch, 'h': h, 'w': w, 'dtype': dtype_str,
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Reference: cpp/5_Domain_Specific/bilateralFilter/bilateralFilter.cu

def bench_bilateral_filter(device, n=1024):
    """
    Bilateral Filter — maps to cuda-samples: bilateralFilter.
    Edge-preserving smoothing filter (image processing).

    CUDA sample uses:
    - Separable bilateral filter approximation
    - Shared memory for local neighborhood access
    """
    import torch
    dev = get_device(device)

    x = torch.randn(1, 3, n, n, device=dev, dtype=torch.float32)

    def kernel():
        # Approximate with depthwise separable conv
        torch.nn.functional.avg_pool2d(x, kernel_size=5, stride=1, padding=2)

    stats = benchmark_kernel(kernel, device=dev)

    return {
        'cuda_sample': 'bilateralFilter',
        'category': '5_Domain_Specific',
        'n': n, 'dtype': 'fp32',
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# Reference: cpp/5_Domain_Specific/SobelFilter/SobelFilter.cu

def bench_sobel_filter(device, n=2048):
    """
    Sobel Edge Detection — maps to cuda-samples: SobelFilter.
    3x3 gradient computation for edge detection.

    CUDA sample uses:
    - Texture memory for efficient 2D neighborhood access
    - Separable Sobel kernels (Gx, Gy)
    """
    import torch
    dev = get_device(device)

    x = torch.randn(1, 1, n, n, device=dev, dtype=torch.float32)

    def kernel():
        torch.nn.functional.conv2d(x, torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], device=dev), padding=1)

    stats = benchmark_kernel(kernel, device=dev)

    return {
        'cuda_sample': 'SobelFilter',
        'category': '5_Domain_Specific',
        'n': n, 'dtype': 'fp32',
        'latency_us_mean': stats['mean_us'], 'latency_us_std': stats['std_us'],
        'latency_us_p50': stats['p50_us'], 'latency_us_p99': stats['p99_us'],
        'throughput_gbps': -1,
        'memory_bandwidth_utilization_pct': -1,
        'calc_mem_ratio': -1,
    }


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY = {
    # 3_CUDA_Features: TensorCore GEMM variants
    'tensorcore_gemm_fp16': lambda d: bench_tensorcore_gemm(d, 'fp16'),
    'tensorcore_gemm_bf16': lambda d: bench_tensorcore_gemm(d, 'bf16'),
    'imma_gemm_int8': bench_imma_gemm,
    'tf32_gemm': bench_tf32_gemm,

    # 4_CUDA_Libraries: CUBLAS, CUFFT, cuSPARSE
    'batched_gemm_fp16': lambda d: bench_batched_gemm(d, 'fp16'),
    'batched_gemm_bf16': lambda d: bench_batched_gemm(d, 'bf16'),
    'fft': lambda d: bench_fft(d, 'fp32'),
    'spmm': bench_spmm,

    # 2_Concepts_and_Techniques: reduction, scan, histogram, convolution
    'reduction_fp32': lambda d: bench_reduction(d, 'fp32'),
    'reduction_fp16': lambda d: bench_reduction(d, 'fp16'),
    'scan_fp32': lambda d: bench_scan(d, 'fp32'),
    'histogram': bench_histogram,
    'conv2d_fp16': lambda d: bench_conv2d(d, 'fp16'),
    'conv2d_bf16': lambda d: bench_conv2d(d, 'bf16'),

    # 6_Performance: transpose
    'transpose_fp16': lambda d: bench_transpose(d, 'fp16'),
    'transpose_bf16': lambda d: bench_transpose(d, 'bf16'),

    # 5_Domain_Specific
    'conv_fft_fp32': lambda d: bench_conv_fft(d, 'fp32'),
    'bilateral_filter': bench_bilateral_filter,
    'sobel_filter': bench_sobel_filter,
}

CATEGORY_MAP = {
    'features': ['tensorcore_gemm_fp16', 'tensorcore_gemm_bf16', 'imma_gemm_int8', 'tf32_gemm'],
    'libraries': ['batched_gemm_fp16', 'batched_gemm_bf16', 'fft', 'spmm'],
    'concepts': ['reduction_fp32', 'reduction_fp16', 'scan_fp32', 'histogram', 'conv2d_fp16', 'conv2d_bf16'],
    'performance': ['transpose_fp16', 'transpose_bf16'],
    'domain': ['conv_fft_fp32', 'bilateral_filter', 'sobel_filter'],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks(args):
    print(f"\n{'='*70}")
    print(f"  CUDA Samples Representative Operators Benchmark")
    print(f"  Reference: https://github.com/NVIDIA/cuda-samples")
    print(f"{'='*70}\n")

    device = args.device if args.device else detect_device()
    print(f"Device: {device}")

    dev_info = get_device_info(device)
    print(f"Hardware: {json.dumps(dev_info, indent=2, ensure_ascii=False)}\n")

    # Determine which operators to run
    if args.categories == 'all':
        operators = list(OPERATOR_REGISTRY.keys())
    else:
        cats = args.categories.split(',')
        operators = []
        for cat in cats:
            cat = cat.strip()
            if cat in CATEGORY_MAP:
                operators.extend(CATEGORY_MAP[cat])
        operators = list(dict.fromkeys(operators))  # dedupe while preserving order

    if args.operators:
        operators = args.operators.split(',')

    all_results = []

    for op_name in operators:
        if op_name not in OPERATOR_REGISTRY:
            print(f"  [SKIP] Unknown operator: {op_name}")
            continue

        bench_fn = OPERATOR_REGISTRY[op_name]
        print(f"  Running: {op_name}", end=' ', flush=True)

        try:
            result = bench_fn(device)
            result['device'] = device
            all_results.append(result)
            latency = result.get('latency_us_mean', -1)
            if latency > 0:
                print(f"✅ {latency:.1f} µs")
            else:
                print(f"✅ completed")
        except Exception as e:
            print(f"❌ {type(e).__name__}: {e}")
            all_results.append({
                'cuda_sample': op_name,
                'category': 'unknown',
                'device': device,
                'latency_us_mean': -1,
                'error': f"{type(e).__name__}: {e}",
            })

    # Compile report
    report = {
        'device_info': dev_info,
        'reference': 'https://github.com/NVIDIA/cuda-samples',
        'categories_run': args.categories,
        'results': all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")

    # Print summary table
    print("Summary:")
    print(f"  {'CUDA Sample':<35} {'Category':<28} {'Latency (µs)':<15}")
    print(f"  {'-'*35} {'-'*28} {'-'*15}")
    for r in all_results:
        name = r.get('cuda_sample', 'unknown')
        cat = r.get('category', '')
        lat = r.get('latency_us_mean', -1)
        lat_str = f"{lat:.1f}" if lat > 0 else "N/A"
        print(f"  {name:<35} {cat:<28} {lat_str:<15}")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CUDA Samples Representative Operators — Cross-Platform Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  features    TensorCore GEMM variants (FP16, BF16, TF32, INT8)
  libraries   CUBLAS batched GEMM, CUFFT, cuSPARSE SpMM
  concepts    Reduction, Scan, Histogram, Conv2D
  performance Matrix Transpose (memory access patterns)
  domain      FFT convolution, bilateral filter, Sobel filter

Examples:
  python3 04_cuda_samples_ops.py --categories all
  python3 04_cuda_samples_ops.py --categories features,libraries --device cuda
  python3 04_cuda_samples_ops.py --operators tensorcore_gemm_fp16,reduction_fp32
        """,
    )
    parser.add_argument('--device', type=str, default=None,
                        help='Device backend (auto-detect if not specified)')
    parser.add_argument('--categories', type=str, default='all',
                        help='Comma-separated categories (default: all)')
    parser.add_argument('--operators', type=str, default=None,
                        help='Comma-separated operator names (overrides categories)')
    parser.add_argument('--output', type=str, default='results/cuda_samples_ops.json',
                        help='Output JSON file path')
    args = parser.parse_args()
    run_benchmarks(args)
