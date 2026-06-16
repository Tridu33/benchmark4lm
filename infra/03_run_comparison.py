#!/usr/bin/env python3
"""
Benchmark 3: AI Infrastructure — Comparison Report Generator

Runs operator and communication benchmarks on available hardware,
generates a structured comparison report with visual tables.

Usage:
    # Run on current device
    python3 03_run_comparison.py [--device cuda|ppu|ascend] [--num-gpus 8]

    # Run on multiple devices (requires multi-device machine)
    python3 03_run_comparison.py --devices cuda,ppu,ascend --num-gpus 8

    # Generate report from existing results
    python3 03_run_comparison.py --report-only

    # Export to markdown
    python3 03_run_comparison.py --report-only --export-markdown
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Reference hardware specs (for comparison context)
# ---------------------------------------------------------------------------

REFERENCE_HARDWARE = {
    'NVIDIA A100 (80GB)': {
        'peak_tflops_bf16': 312,
        'peak_tflops_fp16': 312,
        'peak_bandwidth_gbps': 2039,
        'interconnect': 'NVLink 3.0 (600 GB/s)',
        'interconnect_bw_gbps': 600,
        'interconnect_type': 'NVLink',
    },
    'NVIDIA H100 (80GB)': {
        'peak_tflops_bf16': 989,
        'peak_tflops_fp16': 1979,
        'peak_bandwidth_gbps': 3350,
        'interconnect': 'NVLink 4.0 (900 GB/s)',
        'interconnect_bw_gbps': 900,
        'interconnect_type': 'NVLink',
    },
    'NVIDIA H20': {
        'peak_tflops_bf16': 148,
        'peak_tflops_fp16': 296,
        'peak_bandwidth_gbps': 900,
        'interconnect': 'NVLink 4.0 (900 GB/s)',
        'interconnect_bw_gbps': 900,
        'interconnect_type': 'NVLink',
    },
    'Alibaba PPU (80GB)': {
        'peak_tflops_bf16': 390,
        'peak_tflops_fp16': 390,
        'peak_bandwidth_gbps': 1200,
        'interconnect': 'HCCS (~300 GB/s)',
        'interconnect_bw_gbps': 300,
        'interconnect_type': 'HCCS',
    },
    'Huawei Ascend 910B': {
        'peak_tflops_bf16': 313,
        'peak_tflops_fp16': 313,
        'peak_bandwidth_gbps': 1200,
        'interconnect': 'HCCS (392 GB/s)',
        'interconnect_bw_gbps': 392,
        'interconnect_type': 'HCCS',
    },
    'Cambricon MLU370-X8': {
        'peak_tflops_bf16': 256,
        'peak_tflops_fp16': 512,
        'peak_bandwidth_gbps': 819,
        'interconnect': 'MLU-Link (200 GB/s)',
        'interconnect_bw_gbps': 200,
        'interconnect_type': 'MLU-Link',
    },
    'Hygon DCU Z100': {
        'peak_tflops_bf16': 280,
        'peak_tflops_fp16': 280,
        'peak_bandwidth_gbps': 1600,
        'interconnect': 'xGMI (PCIe)',
        'interconnect_bw_gbps': 200,
        'interconnect_type': 'xGMI',
    },
    'Moore Threads MTT S4000': {
        'peak_tflops_bf16': 200,
        'peak_tflops_fp16': 200,
        'peak_bandwidth_gbps': 1000,
        'interconnect': 'MUSA Link (400 GB/s)',
        'interconnect_bw_gbps': 400,
        'interconnect_type': 'MUSA Link',
    },
}


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def run_script(script_path: str, extra_args: List[str] = None) -> bool:
    """Run a benchmark script and return success status."""
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def run_operator_benchmark(device: str, output: str) -> bool:
    """Run operator performance benchmark."""
    return run_script(SCRIPT_DIR / '01_operator_perf.py', [
        '--device', device,
        '--output', output,
    ])


def run_communication_benchmark(backend: str, num_gpus: int, output: str) -> bool:
    """Run communication benchmark."""
    env = os.environ.copy()
    cmd = [sys.executable, str(SCRIPT_DIR / '02_communication_perf.py'),
           '--backend', backend,
           '--num-gpus', str(num_gpus),
           '--output', output]

    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def load_results(file_path: str) -> Optional[dict]:
    """Load benchmark results from JSON file."""
    path = Path(file_path)
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_operator_table(results: dict, ref_name: str = None) -> str:
    """Generate a markdown table for operator performance."""
    lines = []
    lines.append("## 算子性能对比 (Operator Performance)")
    lines.append("")

    if ref_name and ref_name in REFERENCE_HARDWARE:
        ref = REFERENCE_HARDWARE[ref_name]
        lines.append(f"**参考硬件**: {ref_name}")
        lines.append(f"- 理论 BF16 算力: {ref['peak_tflops_bf16']} TFLOPS")
        lines.append(f"- 理论峰值带宽: {ref['peak_bandwidth_gbps']} GB/s")
        lines.append(f"- 互联: {ref['interconnect']}")
        lines.append("")

    # Operator results table
    lines.append("| 算子 | 精度 | Batch | Hidden | Seq | 延迟 (µs) | 吞吐量 (TFLOPS) | 带宽利用率 | Calc/Mem |")
    lines.append("|------|------|-------|--------|-----|-----------|----------------|-----------|----------|")

    for r in results.get('operator_results', []):
        if r['latency_us_mean'] < 0:
            lines.append(f"| {r['operator']} | {r['dtype']} | {r['batch_size']} | "
                         f"{r['hidden_size']} | {r['seq_len']} | ❌ 失败 | ❌ | ❌ | ❌ |")
        else:
            lines.append(f"| {r['operator']} | {r['dtype']} | {r['batch_size']} | "
                         f"{r['hidden_size']} | {r['seq_len']} | {r['latency_us_mean']:.1f} ± {r['latency_us_std']:.1f} | "
                         f"{r['throughput_tflops']:.2f} | {r['memory_bandwidth_utilization_pct']:.1f}% | "
                         f"{r['calc_mem_ratio']:.1f} |")

    lines.append("")
    return '\n'.join(lines)


def generate_quantization_table(results: dict) -> str:
    """Generate quantization support table."""
    lines = []
    lines.append("## 混合精度与量化支持 (Quantization Support)")
    lines.append("")
    lines.append("| 格式 | 支持 | 硬件加速 |")
    lines.append("|------|------|---------|")

    quant = results.get('quantization_support', {})
    for fmt, info in quant.items():
        supported = "✅" if info.get('supported') else "❌"
        hw_accel = "✅" if info.get('hw_accelerated') else "❌"
        lines.append(f"| {fmt} | {supported} | {hw_accel} |")

    lines.append("")
    return '\n'.join(lines)


def generate_ilp_table(results: dict) -> str:
    """Generate ILP assessment table."""
    lines = []
    lines.append("## 指令级并行度 (Instruction-Level Parallelism)")
    lines.append("")
    lines.append("| 矩阵规模 | 串行延迟 (µs) | 并行延迟 (µs) | 加速比 | SIMD 效率 |")
    lines.append("|----------|--------------|--------------|--------|----------|")

    ilp = results.get('ilp_assessment', {})
    for shape, info in ilp.items():
        par = f"{info['parallel_us']:.1f}" if info['parallel_us'] else "N/A"
        lines.append(f"| {shape} | {info['sequential_us']:.1f} | {par} | "
                     f"{info['speedup']:.2f}x | {info['simd_efficiency']:.1f}% |")

    lines.append("")
    return '\n'.join(lines)


def generate_communication_table(results: dict) -> str:
    """Generate communication benchmark table."""
    lines = []
    lines.append("## 通信与互联性能 (Communication & Interconnect)")
    lines.append("")

    # Peak bandwidth
    peak = results.get('peak_interconnect_bandwidth_gbps', 'N/A')
    lines.append(f"**峰值互联带宽**: {peak} GB/s")
    lines.append("")

    # Point-to-point
    p2p = results.get('point_to_point', {})
    if p2p:
        lines.append("### 点对点通信 (Point-to-Point)")
        lines.append("")
        lines.append("| 数据量 | 带宽 (GB/s) | 延迟 (µs) | P99 延迟 | 抖动 |")
        lines.append("|--------|------------|----------|---------|------|")
        for size, stats in p2p.items():
            jitter = stats.get('latency_us_std', 0) / max(stats.get('latency_us_mean', 1), 1e-9) * 100
            lines.append(f"| {size} | {stats['bandwidth_gbps']} | "
                         f"{stats['latency_us_mean']:.1f} | {stats['latency_us_p99']:.1f} | "
                         f"{jitter:.1f}% |")
        lines.append("")

    # Collective operations
    collective = results.get('collective', {})
    if collective:
        lines.append("### 集合通信 (Collective Communication)")
        lines.append("")

        for op, data in collective.items():
            lines.append(f"**{op}**")
            lines.append("")
            lines.append("| 数据量 | 带宽 (GB/s) | 延迟 (µs) | P99 延迟 | 抖动 |")
            lines.append("|--------|------------|----------|---------|------|")
            for size, stats in data.items():
                lines.append(f"| {size} | {stats['bandwidth_gbps']} | "
                             f"{stats['latency_us_mean']:.1f} | {stats['latency_us_p99']:.1f} | "
                             f"{stats['jitter_pct']:.1f}% |")
            lines.append("")

    return '\n'.join(lines)


def generate_robustness_table(results: dict) -> str:
    """Generate library robustness test results."""
    lines = []
    lines.append("## 通信库鲁棒性 (Library Robustness)")
    lines.append("")

    robust = results.get('robustness', {})
    if robust:
        total = robust.get('total_rounds', 0)
        success = robust.get('successful_rounds', 0)
        failed = robust.get('failed_rounds', 0)
        rate = success / total * 100 if total > 0 else 0

        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 总轮次 | {total} |")
        lines.append(f"| 成功 | {success} |")
        lines.append(f"| 失败 | {failed} |")
        lines.append(f"| 成功率 | {rate:.1f}% |")
        lines.append(f"| 死锁检测 | {'⚠️ 是' if robust.get('deadlock_detected') else '✅ 否'} |")

        errors = robust.get('errors', [])
        if errors:
            lines.append(f"| 错误类型 | {', '.join(errors)} |")

        lines.append("")
    return '\n'.join(lines)


def generate_scaling_table(results: dict) -> str:
    """Generate scaling efficiency table."""
    lines = []
    lines.append("## 集群线性扩展率 (Scaling Efficiency)")
    lines.append("")

    scaling = results.get('scaling', {})
    if scaling:
        world_size = results.get('world_size', 1)
        lines.append(f"**节点数**: {world_size}")
        lines.append("")

        for op, data in scaling.items():
            lines.append(f"**{op}**")
            lines.append("")
            lines.append("| 数据量 | 延迟 (µs) | P99 延迟 (µs) |")
            lines.append("|--------|----------|-------------|")
            for size, metrics in data.items():
                lines.append(f"| {size} | {metrics['latency_us_mean']:.1f} | "
                             f"{metrics['latency_us_p99']:.1f} |")
            lines.append("")

    # Add theoretical comparison
    lines.append("### 理论扩展效率参考")
    lines.append("")
    lines.append("| 平台 | 8卡扩展效率 | 千卡扩展效率 |")
    lines.append("|------|------------|------------|")
    lines.append("| NVIDIA (NVLink + InfiniBand) | 85%-95% | 80%-90% |")
    lines.append("| 华为 Ascend (HCCS + RoCE) | 70%-85% | 60%-75% |")
    lines.append("| 阿里 PPU (HCCS + RoCE) | 65%-80% | 55%-70% |")
    lines.append("| 海光 DCU (PCIe + RoCE) | 60%-75% | 40%-60% |")
    lines.append("| 寒武纪 MLU (MLU-Link + RoCE) | 55%-70% | 40%-55% |")
    lines.append("")

    return '\n'.join(lines)


def generate_summary_report(all_results: dict) -> str:
    """Generate a comprehensive comparison summary."""
    lines = []
    lines.append("# AI 基础设施对比评测报告")
    lines.append("")
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 评测概述")
    lines.append("")
    lines.append("本评测对比国产 AI 加速卡（阿里云 PPU、华为 Ascend、寒武纪 MLU、海光 DCU 等）")
    lines.append("与 NVIDIA GPU 在算子性能、通信互联、量化支持等方面的差异。")
    lines.append("")

    # Device info
    for device_name, results in all_results.items():
        dev_info = results.get('device_info', {})
        lines.append(f"### 设备: {device_name}")
        lines.append(f"- 设备类型: {dev_info.get('device_type', 'unknown')}")
        lines.append(f"- 设备名称: {dev_info.get('device_name', 'unknown')}")
        lines.append(f"- 显存总量: {dev_info.get('memory_total_gb', 0)} GB")
        if 'compute_capability' in dev_info:
            lines.append(f"- 计算能力: {dev_info['compute_capability']}")
        if 'sm_count' in dev_info:
            lines.append(f"- SM 数量: {dev_info['sm_count']}")
        lines.append("")

    return '\n'.join(lines)


def generate_full_report(all_results: dict, output_path: str):
    """Generate complete markdown report."""
    parts = []

    # Summary
    parts.append(generate_summary_report(all_results))

    # Per-device operator performance
    for device_name, results in all_results.items():
        if 'operator_results' in results:
            parts.append(f"\n---\n")
            parts.append(f"## {device_name} 评测结果\n")
            parts.append(generate_operator_table(results, device_name))
            parts.append(generate_quantization_table(results))
            parts.append(generate_ilp_table(results))

    # Communication results (shared)
    for device_name, results in all_results.items():
        if 'collective' in results:
            parts.append(f"\n---\n")
            parts.append(generate_communication_table(results))
            parts.append(generate_robustness_table(results))
            parts.append(generate_scaling_table(results))
            break  # Only show once

    # Hardware comparison table
    parts.append("\n---\n")
    parts.append("## 硬件规格对比参考")
    parts.append("")
    parts.append("| 平台 | BF16 TFLOPS | 带宽 (GB/s) | 互联 | 互联带宽 (GB/s) |")
    parts.append("|------|------------|------------|------|----------------|")
    for name, specs in REFERENCE_HARDWARE.items():
        parts.append(f"| {name} | {specs['peak_tflops_bf16']} | {specs['peak_bandwidth_gbps']} | "
                     f"{specs['interconnect_type']} | {specs['interconnect_bw_gbps']} |")
    parts.append("")

    report = '\n'.join(parts)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*70}")
    print(f"Report saved to: {output}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AI Infra — Comparison Report Generator')
    parser.add_argument('--device', type=str, default=None,
                        help='Primary device to benchmark')
    parser.add_argument('--devices', type=str, default=None,
                        help='Comma-separated devices to benchmark')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs for communication benchmark')
    parser.add_argument('--report-only', action='store_true',
                        help='Only generate report from existing results')
    parser.add_argument('--export-markdown', action='store_true',
                        help='Export report to markdown')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if not args.report_only:
        # Run benchmarks
        devices = args.devices.split(',') if args.devices else (args.device or 'auto').split(',')

        for device in devices:
            if device == 'auto':
                # Auto-detect
                try:
                    sys.path.insert(0, str(SCRIPT_DIR))
                    import importlib
                    op_mod = importlib.import_module('01_operator_perf')
                    device = op_mod.detect_device()
                except Exception:
                    device = 'cuda'

            print(f"\n{'#'*70}")
            print(f"# Running benchmarks on: {device}")
            print(f"{'#'*70}\n")

            # Operator benchmark
            op_output = str(output_dir / f'operator_perf_{device}.json')
            success = run_operator_benchmark(device, op_output)
            if success:
                all_results[device] = load_results(op_output) or {}

            # Communication benchmark
            backend_map = {
                'cuda': 'nccl',
                'ppu': 'hccl',
                'ascend': 'hccl',
                'cambricon': 'cncl',
                'dcu': 'nccl',
                'musa': 'mccl',
            }
            backend = backend_map.get(device, 'nccl')
            comm_output = str(output_dir / f'comm_perf_{device}.json')

            if args.num_gpus > 1:
                success = run_communication_benchmark(backend, args.num_gpus, comm_output)
                if success:
                    if device in all_results:
                        all_results[device].update(load_results(comm_output) or {})
                    else:
                        all_results[device] = load_results(comm_output) or {}
    else:
        # Load existing results
        print("Loading existing results for report generation...\n")

        for json_file in output_dir.glob('operator_perf_*.json'):
            device = json_file.stem.replace('operator_perf_', '')
            all_results[device] = load_results(str(json_file)) or {}

        for json_file in output_dir.glob('comm_perf_*.json'):
            device = json_file.stem.replace('comm_perf_', '')
            if device in all_results:
                all_results[device].update(load_results(str(json_file)) or {})
            else:
                all_results[device] = load_results(str(json_file)) or {}

    # Generate report
    if all_results:
        report_path = str(output_dir / 'comparison_report.md')
        generate_full_report(all_results, report_path)

        if args.export_markdown:
            print(f"\nMarkdown report exported to: {report_path}")
    else:
        print("No results found to generate report.")


if __name__ == '__main__':
    main()
