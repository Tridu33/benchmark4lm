"""
core_profiler.py — Shared profiling utilities for all benchmark4lm scripts.

Provides:
  - GPU/NVML monitoring (utilization, memory, temperature, power, PCIe BW)
  - CPU monitoring (utilization, memory, threads)
  - PyTorch Profiler integration
  - NVIDIA Nsight Systems wrapper
  - LangFuse tracing helper
  - Result aggregation & Markdown report generation
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── NVML ───────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ── psutil ─────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ── PyTorch (optional at import time) ──────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── LangFuse (optional) ────────────────────────────────────────
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


# ================================================================
#  Data Classes
# ================================================================
@dataclass
class GpuSnapshot:
    timestamp: float = 0.0
    gpu_util_pct: float = 0.0
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    mem_pct: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    power_limit_w: float = 0.0
    pcie_bw_rx_mbs: float = 0.0
    pcie_bw_tx_mbs: float = 0.0
    sm_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    fan_pct: float = 0.0


@dataclass
class CpuSnapshot:
    timestamp: float = 0.0
    cpu_pct: float = 0.0
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    mem_pct: float = 0.0
    num_threads: int = 0
    load_avg_1: float = 0.0
    load_avg_5: float = 0.0
    load_avg_15: float = 0.0


@dataclass
class ProfilingResult:
    """Aggregated profiling results."""
    # Inference metrics
    ttft_list: List[float] = field(default_factory=list)          # time-to-first-token
    tpot_list: List[float] = field(default_factory=list)          # time-per-output-token
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    request_latencies: List[float] = field(default_factory=list)
    throughput_tps: float = 0.0                                    # tokens/sec
    throughput_rps: float = 0.0                                    # requests/sec
    gen_fps: float = 0.0                                           # generated tokens/sec
    flop_per_sec: float = 0.0
    kv_cache_used_mb: float = 0.0
    kv_cache_max_mb: float = 0.0
    max_concurrent_requests: int = 0

    # Training metrics
    train_loss_list: List[float] = field(default_factory=list)
    eval_loss_list: List[float] = field(default_factory=list)
    grad_norm_list: List[float] = field(default_factory=list)
    learning_rate_list: List[float] = field(default_factory=list)
    steps_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    convergence_status: str = "unknown"

    # Hardware snapshots
    gpu_snapshots: List[GpuSnapshot] = field(default_factory=list)
    cpu_snapshots: List[CpuSnapshot] = field(default_factory=list)

    # Peak / average memory
    peak_gpu_mem_mb: float = 0.0
    avg_gpu_mem_mb: float = 0.0
    peak_cpu_mem_mb: float = 0.0
    avg_cpu_mem_mb: float = 0.0

    # Utilization stats
    avg_gpu_util: float = 0.0
    avg_cpu_util: float = 0.0

    # PyTorch profiler summary
    pt_profiler_summary: Dict[str, Any] = field(default_factory=dict)

    # Nsight report path
    nsys_report_path: str = ""

    # LangFuse trace URL
    langfuse_trace_url: str = ""

    # Environment
    hw_summary: Dict[str, str] = field(default_factory=dict)
    sw_summary: Dict[str, str] = field(default_factory=dict)

    # Bottleneck analysis
    bottleneck_analysis: str = ""
    optimization_suggestions: List[str] = field(default_factory=list)


# ================================================================
#  GPU Monitor Thread
# ================================================================
class GpuMonitor(threading.Thread):
    """Periodically samples NVML metrics for a given GPU index."""

    def __init__(self, gpu_id: int = 0, interval_sec: float = 0.5):
        super().__init__(daemon=True)
        self.gpu_id = gpu_id
        self.interval = interval_sec
        self.snapshots: List[GpuSnapshot] = []
        self._stop_event = threading.Event()

    def run(self):
        if not NVML_AVAILABLE:
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except pynvml.NVMLError:
            return

        while not self._stop_event.is_set():
            snap = GpuSnapshot(timestamp=time.time())
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snap.gpu_util_pct = util.gpu
                snap.mem_pct = util.memory

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                snap.mem_used_mb = mem.used / (1024 ** 2)
                snap.mem_total_mb = mem.total / (1024 ** 2)

                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                snap.temperature_c = temp

                try:
                    pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    pw_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    snap.power_w = pw
                    snap.power_limit_w = pw_limit
                except pynvml.NVMLError:
                    pass

                try:
                    bw = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    snap.pcie_bw_rx_mbs = bw / (1024 ** 2) if bw != pynvml.NVML_ERROR_NOT_SUPPORTED else 0
                    bw_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    snap.pcie_bw_tx_mbs = bw_tx / (1024 ** 2) if bw_tx != pynvml.NVML_ERROR_NOT_SUPPORTED else 0
                except pynvml.NVMLError:
                    pass

                try:
                    snap.sm_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    snap.mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except pynvml.NVMLError:
                    pass

                try:
                    snap.fan_pct = pynvml.nvmlDeviceGetFanSpeed(handle)
                except pynvml.NVMLError:
                    pass

            except pynvml.NVMLError:
                pass

            self.snapshots.append(snap)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    @property
    def peak_mem_mb(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(s.mem_used_mb for s in self.snapshots)

    @property
    def avg_mem_mb(self) -> float:
        if not self.snapshots:
            return 0.0
        return np.mean([s.mem_used_mb for s in self.snapshots])

    @property
    def avg_util(self) -> float:
        if not self.snapshots:
            return 0.0
        return np.mean([s.gpu_util_pct for s in self.snapshots])


# ================================================================
#  CPU Monitor Thread
# ================================================================
class CpuMonitor(threading.Thread):
    """Periodically samples CPU metrics via psutil."""

    def __init__(self, interval_sec: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.snapshots: List[CpuSnapshot] = []
        self._stop_event = threading.Event()

    def run(self):
        if not PSUTIL_AVAILABLE:
            return

        # First call always returns 0 — prime it
        psutil.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            snap = CpuSnapshot(timestamp=time.time())
            snap.cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            snap.mem_used_mb = vm.used / (1024 ** 2)
            snap.mem_total_mb = vm.total / (1024 ** 2)
            snap.mem_pct = vm.percent
            snap.num_threads = psutil.cpu_count(logical=True)
            try:
                la = os.getloadavg()
                snap.load_avg_1, snap.load_avg_5, snap.load_avg_15 = la
            except OSError:
                pass

            self.snapshots.append(snap)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    @property
    def peak_mem_mb(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(s.mem_used_mb for s in self.snapshots)

    @property
    def avg_mem_mb(self) -> float:
        if not self.snapshots:
            return 0.0
        return np.mean([s.mem_used_mb for s in self.snapshots])

    @property
    def avg_util(self) -> float:
        if not self.snapshots:
            return 0.0
        return np.mean([s.cpu_pct for s in self.snapshots])


# ================================================================
#  PyTorch Profiler Wrapper
# ================================================================
class TorchProfilerCtx:
    """Context manager wrapping torch.profiler with CSV / JSON export."""

    def __init__(self, output_dir: str = "./out", name: str = "profile"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.profiler: Optional[Any] = None
        self.summary: Dict[str, Any] = {}

    def __enter__(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        if self.profiler is not None:
            self.profiler.__exit__(*args)
            out = self.output_dir / self.name

            # Table summary
            table = self.profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=30
            )
            self.summary["table"] = str(table)

            # Export Chrome trace
            chrome_path = str(out) + ".chrome_trace.json"
            self.profiler.export_chrome_trace(chrome_path)
            self.summary["chrome_trace"] = chrome_path

            # Export CPU & CUDA time aggregates
            for act in ["cpu_time_total", "cuda_time_total"]:
                try:
                    events = self.profiler.key_averages().table(
                        sort_by=act, row_limit=20
                    )
                    self.summary[act] = str(events)
                except Exception:
                    pass

            # Operator-level breakdown
            try:
                ka = self.profiler.key_averages(group_by_input_shape=True)
                op_data = []
                for ev in ka:
                    op_data.append({
                        "name": ev.key,
                        "cpu_time_us": ev.cpu_time_total,
                        "cuda_time_us": ev.cuda_time_total,
                        "self_cpu_time_us": ev.self_cpu_time_total,
                        "self_cuda_time_us": ev.self_cuda_time_total,
                    })
                self.summary["operators"] = op_data
            except Exception:
                pass

    @property
    def is_active(self) -> bool:
        return self.profiler is not None

    def step(self):
        if self.profiler is not None:
            self.profiler.step()


# ================================================================
#  Nsight Systems Wrapper
# ================================================================
def run_with_nsys(
    cmd: List[str],
    output_dir: str = "./out",
    report_name: str = "nsys_profile",
    duration_ms: int = 0,
    delay_ms: int = 0,
) -> str:
    """
    Run *cmd* under ``nsys profile``. Returns the .nsys-rep path.
    Requires nsys on $PATH (NVIDIA Nsight Systems installed).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nsys_args = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt,cudnn,cublas",
        "--capture-range=none",
        "--force-overwrite=true",
        f"--output={out / report_name}",
    ]
    if duration_ms > 0:
        nsys_args += ["--duration", str(duration_ms)]
    if delay_ms > 0:
        nsys_args += ["--delay", str(delay_ms)]

    nsys_args += cmd

    print(f"[nsys] Running: {' '.join(nsys_args)}")
    proc = subprocess.run(nsys_args, timeout=None)
    rep_path = str(out / f"{report_name}.nsys-rep")
    if os.path.exists(rep_path):
        return rep_path
    # Some nsight versions append .nsys-rep automatically
    alt = str(out / report_name)
    if os.path.exists(alt):
        return alt
    return ""


# ================================================================
#  LangFuse Helper
# ================================================================
class LangFuseTracer:
    """Simple LangFuse tracing wrapper for LLM calls."""

    def __init__(self, project_name: str = "benchmark4lm"):
        self.project_name = project_name
        self.langfuse: Optional[Any] = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse = Langfuse(
                    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
                    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                print(f"[LangFuse] Connected to {self.langfuse.auth_check()}")
            except Exception as e:
                print(f"[LangFuse] Init failed: {e}")
                self.langfuse = None

    def trace_generation(
        self,
        name: str,
        input_text: str,
        output_text: str,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        if self.langfuse is None:
            return None
        try:
            trace = self.langfuse.trace(
                name=name,
                input={"prompt": input_text},
                output={"response": output_text},
                metadata=metadata or {},
            )
            trace.generation(
                name="llm_call",
                model=model,
                prompt=input_text,
                completion=output_text,
                usage={
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                    "totalTokens": prompt_tokens + completion_tokens,
                },
                latency=latency_ms / 1000.0 if latency_ms else None,
                cost=cost,
            )
            self.langfuse.flush()
            return trace.url
        except Exception as e:
            print(f"[LangFuse] trace error: {e}")
            return None

    def shutdown(self):
        if self.langfuse:
            self.langfuse.flush()


# ================================================================
#  Environment Detection
# ================================================================
def collect_hw_summary() -> Dict[str, str]:
    """Return a dict of hardware info strings."""
    info: Dict[str, str] = {}
    if NVML_AVAILABLE:
        try:
            count = pynvml.nvmlDeviceGetCount()
            info["gpu_count"] = str(count)
            for i in range(min(count, 4)):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                info[f"gpu_{i}_name"] = pynvml.nvmlDeviceGetName(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                info[f"gpu_{i}_mem_gb"] = f"{mem.total / (1024**3):.1f}"
        except Exception as e:
            info["gpu_error"] = str(e)

    if PSUTIL_AVAILABLE:
        info["cpu_logical"] = str(psutil.cpu_count(logical=True))
        info["cpu_physical"] = str(psutil.cpu_count(logical=False))
        vm = psutil.virtual_memory()
        info["system_mem_gb"] = f"{vm.total / (1024**3):.1f}"

    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_device_name"] = torch.cuda.get_device_name(0)

    return info


def collect_sw_summary() -> Dict[str, str]:
    """Return software / dependency versions."""
    info: Dict[str, str] = {}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import deepspeed
        info["deepspeed"] = deepspeed.__version__
    except ImportError:
        pass
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except ImportError:
        pass
    try:
        import sglang
        info["sglang"] = getattr(sglang, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    try:
        import accelerate
        info["accelerate"] = accelerate.__version__
    except ImportError:
        pass
    info["python"] = f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
    info["platform"] = __import__('platform').platform()
    return info


# ================================================================
#  Markdown Report Generator
# ================================================================
def generate_report(
    result: ProfilingResult,
    output_path: str = "./out/result.md",
    scenario_name: str = "benchmark4lm",
    model_type: str = "llm",
    task_type: str = "inference",
) -> str:
    """Generate a comprehensive Markdown report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# {scenario_name} — Performance Analysis Report\n")
    lines.append(f"**Model Type:** {model_type}  ")
    lines.append(f"**Task Type:** {task_type}  ")
    lines.append(f"**Generated:** {datetime.now().isoformat()}  ")
    lines.append("")

    # ── Environment ──────────────────────────────────────────
    lines.append("## 1. Hardware & Software Environment\n")
    lines.append("### Hardware\n")
    lines.append("| Component | Value |")
    lines.append("|-----------|-------|")
    for k, v in result.hw_summary.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### Software Dependencies\n")
    lines.append("| Package | Version |")
    lines.append("|---------|---------|")
    for k, v in result.sw_summary.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # ── Inference Metrics ────────────────────────────────────
    if task_type == "inference" and result.ttft_list:
        lines.append("## 2. Inference Performance Metrics\n")
        ttft_arr = np.array(result.ttft_list)
        tpot_arr = np.array(result.tpot_list)
        lat_arr = np.array(result.request_latencies)

        lines.append("### Time-To-First-Token (TTFT)\n")
        lines.append(f"- **Mean:** {ttft_arr.mean():.2f} ms")
        lines.append(f"- **P50:** {np.percentile(ttft_arr, 50):.2f} ms")
        lines.append(f"- **P95:** {np.percentile(ttft_arr, 95):.2f} ms")
        lines.append(f"- **P99:** {np.percentile(ttft_arr, 99):.2f} ms")
        lines.append(f"- **Min/Max:** {ttft_arr.min():.2f} / {ttft_arr.max():.2f} ms")
        lines.append("")

        if len(tpot_arr) > 0 and tpot_arr.sum() > 0:
            lines.append("### Time-Per-Output-Token (TPOT)\n")
            lines.append(f"- **Mean:** {tpot_arr.mean():.2f} ms")
            lines.append(f"- **P50:** {np.percentile(tpot_arr, 50):.2f} ms")
            lines.append(f"- **P95:** {np.percentile(tpot_arr, 95):.2f} ms")
            lines.append("")

        lines.append("### End-to-End Request Latency\n")
        lines.append(f"- **Mean:** {lat_arr.mean():.2f} ms")
        lines.append(f"- **P50:** {np.percentile(lat_arr, 50):.2f} ms")
        lines.append(f"- **P95:** {np.percentile(lat_arr, 95):.2f} ms")
        lines.append(f"- **P99:** {np.percentile(lat_arr, 99):.2f} ms")
        lines.append("")

        lines.append("### Throughput\n")
        lines.append(f"- **Tokens/sec:** {result.throughput_tps:.2f}")
        lines.append(f"- **Requests/sec:** {result.throughput_rps:.2f}")
        lines.append(f"- **Gen FPS (tokens/sec):** {result.gen_fps:.2f}")
        lines.append(f"- **FLOP/s:** {result.flop_per_sec:.2e}")
        lines.append("")

        lines.append("### KV Cache\n")
        lines.append(f"- **Used:** {result.kv_cache_used_mb:.1f} MB")
        lines.append(f"- **Max Capacity:** {result.kv_cache_max_mb:.1f} MB")
        lines.append("")

        lines.append(f"### Concurrency\n")
        lines.append(f"- **Max concurrent requests per GPU:** {result.max_concurrent_requests}")
        lines.append("")

    # ── Training Metrics ─────────────────────────────────────
    if task_type == "training" and result.train_loss_list:
        lines.append("## 2. Training Performance Metrics\n")
        loss_arr = np.array(result.train_loss_list)
        lines.append("### Training Loss\n")
        lines.append(f"- **Initial:** {loss_arr[0]:.4f}")
        lines.append(f"- **Final:** {loss_arr[-1]:.4f}")
        lines.append(f"- **Min:** {loss_arr.min():.4f}")
        lines.append(f"- **Convergence Status:** {result.convergence_status}")
        lines.append("")

        if result.eval_loss_list:
            eval_arr = np.array(result.eval_loss_list)
            lines.append("### Eval Loss\n")
            lines.append(f"- **Final:** {eval_arr[-1]:.4f}")
            lines.append(f"- **Min:** {eval_arr.min():.4f}")
            lines.append("")

        lines.append("### Training Speed\n")
        lines.append(f"- **Steps/sec:** {result.steps_per_sec:.2f}")
        lines.append(f"- **Samples/sec:** {result.samples_per_sec:.2f}")
        lines.append("")

        if result.grad_norm_list:
            gn_arr = np.array(result.grad_norm_list)
            lines.append("### Gradient Norm\n")
            lines.append(f"- **Mean:** {gn_arr.mean():.4f}")
            lines.append(f"- **Max:** {gn_arr.max():.4f}")
            lines.append("")

        if result.learning_rate_list:
            lines.append("### Learning Rate Schedule\n")
            lines.append(f"- **Initial:** {result.learning_rate_list[0]:.2e}")
            lines.append(f"- **Final:** {result.learning_rate_list[-1]:.2e}")
            lines.append("")

    # ── GPU Utilization ──────────────────────────────────────
    if result.gpu_snapshots:
        lines.append("## 3. GPU Resource Monitoring\n")
        gpu_utils = [s.gpu_util_pct for s in result.gpu_snapshots]
        gpu_mems = [s.mem_used_mb for s in result.gpu_snapshots]
        gpu_temps = [s.temperature_c for s in result.gpu_snapshots]
        gpu_powers = [s.power_w for s in result.gpu_snapshots if s.power_w > 0]
        pcie_rx = [s.pcie_bw_rx_mbs for s in result.gpu_snapshots if s.pcie_bw_rx_mbs > 0]
        pcie_tx = [s.pcie_bw_tx_mbs for s in result.gpu_snapshots if s.pcie_bw_tx_mbs > 0]

        lines.append("### GPU Utilization\n")
        lines.append(f"- **Average:** {np.mean(gpu_utils):.1f}%")
        lines.append(f"- **Peak:** {np.max(gpu_utils):.1f}%")
        lines.append(f"- **Min:** {np.min(gpu_utils):.1f}%")
        lines.append("")

        lines.append("### GPU Memory\n")
        lines.append(f"- **Peak:** {np.max(gpu_mems):.1f} MB ({np.max(gpu_mems)/1024:.1f} GB)")
        lines.append(f"- **Average:** {np.mean(gpu_mems):.1f} MB ({np.mean(gpu_mems)/1024:.1f} GB)")
        lines.append("")

        lines.append("### GPU Temperature & Power\n")
        if gpu_temps and max(gpu_temps) > 0:
            lines.append(f"- **Peak Temperature:** {np.max(gpu_temps):.1f} °C")
        if gpu_powers:
            lines.append(f"- **Average Power:** {np.mean(gpu_powers):.1f} W")
            lines.append(f"- **Peak Power:** {np.max(gpu_powers):.1f} W")
        lines.append("")

        if pcie_rx or pcie_tx:
            lines.append("### PCIe Bandwidth\n")
            if pcie_rx:
                lines.append(f"- **RX Avg:** {np.mean(pcie_rx):.2f} MB/s")
            if pcie_tx:
                lines.append(f"- **TX Avg:** {np.mean(pcie_tx):.2f} MB/s")
            lines.append("")

    # ── CPU Utilization ──────────────────────────────────────
    if result.cpu_snapshots:
        lines.append("## 4. CPU Resource Monitoring\n")
        cpu_utils = [s.cpu_pct for s in result.cpu_snapshots]
        cpu_mems = [s.mem_used_mb for s in result.cpu_snapshots]
        lines.append(f"- **CPU Utilization Avg:** {np.mean(cpu_utils):.1f}%")
        lines.append(f"- **CPU Memory Peak:** {np.max(cpu_mems):.1f} MB")
        lines.append(f"- **CPU Memory Avg:** {np.mean(cpu_mems):.1f} MB")
        lines.append("")

    # ── PyTorch Profiler ─────────────────────────────────────
    if result.pt_profiler_summary:
        lines.append("## 5. PyTorch Profiler — Operator Breakdown\n")
        if "table" in result.pt_profiler_summary:
            lines.append("```")
            lines.append(result.pt_profiler_summary["table"])
            lines.append("```")
            lines.append("")

        if "operators" in result.pt_profiler_summary:
            lines.append("### Top CUDA Operators by Execution Time\n")
            lines.append("| Operator | CUDA Time (μs) | Self CUDA (μs) | CPU Time (μs) |")
            lines.append("|----------|---------------|----------------|---------------|")
            ops = sorted(
                result.pt_profiler_summary["operators"],
                key=lambda x: x.get("cuda_time_us", 0),
                reverse=True,
            )[:20]
            for op in ops:
                lines.append(
                    f"| {op['name']} | {op.get('cuda_time_us', 0):.0f} "
                    f"| {op.get('self_cuda_time_us', 0):.0f} "
                    f"| {op.get('cpu_time_us', 0):.0f} |"
                )
            lines.append("")

    # ── Nsight Report ────────────────────────────────────────
    if result.nsys_report_path:
        lines.append("## 6. NVIDIA Nsight Systems Report\n")
        lines.append(f"- **Report Path:** `{result.nsys_report_path}`")
        lines.append("- View with: `nsight-sys <report>` or `nsys stats <report>`")
        lines.append("")

    # ── LangFuse ─────────────────────────────────────────────
    if result.langfuse_trace_url:
        lines.append("## 7. LangFuse Tracing\n")
        lines.append(f"- **Trace URL:** {result.langfuse_trace_url}")
        lines.append("")

    # ── Bottleneck Analysis ──────────────────────────────────
    lines.append("## 8. Bottleneck Analysis & Optimization Suggestions\n")

    if result.bottleneck_analysis:
        lines.append(f"{result.bottleneck_analysis}\n")

    if result.optimization_suggestions:
        lines.append("### Recommendations\n")
        for i, s in enumerate(result.optimization_suggestions, 1):
            lines.append(f"{i}. {s}")
        lines.append("")

    # Auto-generated bottleneck analysis from data
    auto_suggestions = _auto_analyze(result)
    if auto_suggestions:
        lines.append("### Auto-Detected Issues\n")
        for s in auto_suggestions:
            lines.append(f"- {s}")
        lines.append("")

    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(f"\n[Report] Written to {out}")
    return str(out)


def _auto_analyze(result: ProfilingResult) -> List[str]:
    """Auto-detect common bottlenecks from profiling data."""
    suggestions: List[str] = []

    # GPU under-utilization
    if result.avg_gpu_util > 0 and result.avg_gpu_util < 50:
        suggestions.append(
            f"GPU utilization is low (avg {result.avg_gpu_util:.1f}%). "
            "Consider increasing batch size, using continuous batching, "
            "or enabling paged attention to improve compute saturation."
        )

    # Memory bottleneck
    if result.peak_gpu_mem_mb > 0 and result.avg_gpu_mem_mb > 0:
        mem_ratio = result.peak_gpu_mem_mb / max(result.avg_gpu_mem_mb, 1)
        if mem_ratio > 3.0:
            suggestions.append(
                f"Large memory spike detected (peak/avg = {mem_ratio:.1f}x). "
                "Potential memory fragmentation — consider memory pooling "
                "or reducing activation checkpointing granularity."
            )

    # TTFT too high
    if result.ttft_list:
        mean_ttft = np.mean(result.ttft_list)
        if mean_ttft > 500:
            suggestions.append(
                f"High TTFT ({mean_ttft:.0f} ms). Consider prefill chunking, "
                "speculative decoding, or KV-cache reuse to reduce first-token latency."
            )

    # PCIe bandwidth bottleneck
    if result.gpu_snapshots:
        pcie_rx = [s.pcie_bw_rx_mbs for s in result.gpu_snapshots if s.pcie_bw_rx_mbs > 0]
        if pcie_rx and np.mean(pcie_rx) > 1000:
            suggestions.append(
                "High PCIe bandwidth detected — consider moving data loading "
                "to GPU (cuFile) or increasing pin-memory workers."
            )

    # Training divergence
    if result.train_loss_list and len(result.train_loss_list) > 10:
        recent = result.train_loss_list[-10:]
        if any(np.isnan(x) or np.isinf(x) for x in recent):
            suggestions.append(
                "Loss contains NaN/Inf values — check learning rate, "
                "gradient clipping, and mixed precision stability."
            )
        elif recent[-1] > recent[0] * 1.5:
            suggestions.append(
                "Training loss diverging toward the end — consider reducing "
                "learning rate or increasing warmup steps."
            )

    # Grad norm explosion
    if result.grad_norm_list:
        max_gn = max(result.grad_norm_list)
        if max_gn > 100:
            suggestions.append(
                f"Gradient norm spike detected (max={max_gn:.1f}). "
                "Enable gradient clipping (max_norm=1.0) to stabilize training."
            )

    if not suggestions:
        suggestions.append("No major bottlenecks detected from the profiling data.")

    return suggestions


# ================================================================
#  CSV logger for time-series data
# ================================================================
class CsvTimeSeries:
    """Append time-series rows to a CSV file."""

    def __init__(self, output_dir: str = "./out", filename: str = "timeseries.csv"):
        self.path = Path(output_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._header_written = False

    def log(self, row: Dict[str, Any]):
        keys = list(row.keys())
        if not self._header_written:
            self._writer.writerow(keys)
            self._header_written = True
        self._writer.writerow([row.get(k, "") for k in keys])
        self._file.flush()

    def close(self):
        self._file.close()
