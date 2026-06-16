# AI Infrastructure Benchmark — 国产卡 vs NVIDIA 对比评测

本目录包含针对 AI 加速基础设施的系统化评测脚本，用于对比国产 AI 加速卡
（阿里云 PPU、华为 Ascend、寒武纪 MLU、海光 DCU、摩尔线程 MUSA 等）
与 NVIDIA GPU 在算子性能、通信互联等方面的差异。

## 目录结构

```
infra/
├── 01_operator_perf.py        # 算子性能评估脚本 (通用 DL 算子)
├── 02_communication_perf.py   # 通信与互联评估脚本
├── 03_run_comparison.py       # 对比报告生成器
├── 04_cuda_samples_ops.py     # CUDA Samples 代表性算子对比评测
├── run_multinode.sh           # 多节点通信评测启动脚本
├── requirements.txt           # Python 依赖
├── results/                   # 评测结果输出目录（自动创建）
│   ├── operator_perf_*.json
│   ├── comm_perf_*.json
│   ├── cuda_samples_ops.json
│   └── comparison_report.md
└── README.md                  # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r infra/requirements.txt

# 国产卡额外依赖（按需安装）
# 华为 Ascend: pip install torch-npu
# 寒武纪 MLU:  pip install torch-mlu
# 阿里 PPU:    pip install torch-ppu
# 摩尔线程:    pip install torch-musa
```

### 2. 运行算子性能评测

```bash
# 自动检测设备并评测所有算子
python3 infra/01_operator_perf.py

# 指定设备
python3 infra/01_operator_perf.py --device ppu

# 指定算子和配置
python3 infra/01_operator_perf.py \
    --device cuda \
    --operators gemm,flash_attention,rms_norm \
    --batch-sizes 1,4,16,64 \
    --hidden-sizes 4096,8192 \
    --seq-len 4096 \
    --dtypes bf16,fp16

# 查看结果
cat results/operator_perf.json | python3 -m json.tool
```

### 3. 运行通信性能评测

```bash
# 单节点 8 卡评测
python3 infra/02_communication_perf.py --num-gpus 8

# 指定后端
python3 infra/02_communication_perf.py --num-gpus 8 --backend hccl

# 指定测试数据量
python3 infra/02_communication_perf.py \
    --num-gpus 8 \
    --sizes 1MB,16MB,64MB,256MB,1GB \
    --output results/comm_perf_8gpu.json
```

### 4. 多节点评测

```bash
# 创建 hostfile (hosts.txt)
cat > hosts.txt << EOF
node1 slots=8
node2 slots=8
EOF

# 运行多节点评测
bash infra/run_multinode.sh hosts.txt 16 nccl results/multinode_16gpu.json

# 或使用 torchrun 手动启动
MASTER_ADDR=node1 torchrun \
    --nnodes=2 \
    --nproc-per-node=8 \
    --master-addr=node1 \
    --master-port=29500 \
    infra/02_communication_perf.py \
    --backend nccl \
    --num-gpus 16
```

### 5. 运行 CUDA Samples 代表性算子评测

```bash
# 运行所有 cuda-samples 类别的算子
python3 infra/04_cuda_samples_ops.py --categories all

# 只运行 TensorCore GEMM 和库相关算子
python3 infra/04_cuda_samples_ops.py --categories features,libraries

# 指定设备
python3 infra/04_cuda_samples_ops.py --device ppu --categories all

# 运行单个算子
python3 infra/04_cuda_samples_ops.py --operators tensorcore_gemm_fp16,reduction_fp32

# 查看结果
cat results/cuda_samples_ops.json | python3 -m json.tool
```

### 6. 生成对比报告

```bash
# 运行全部评测并生成报告
python3 infra/03_run_comparison.py \
    --device cuda \
    --num-gpus 8

# 从已有结果生成报告
python3 infra/03_run_comparison.py --report-only

# 导出 Markdown 报告
python3 infra/03_run_comparison.py --report-only --export-markdown
cat results/comparison_report.md
```

## 评测维度

### 算子性能 (01_operator_perf.py)

| 维度 | 指标 | 说明 |
|------|------|------|
| 延迟 | Latency (µs) | 算子执行时间，越低越好 |
| 吞吐量 | TFLOPS | 计算吞吐量，越高越好 |
| 内存带宽利用率 | BW Util (%) | 实际带宽 / 理论峰值 |
| 计算/访存比 | Calc/Mem | FLOPs per Byte，判断密集型 |
| SIMD 效率 | SIMD Eff (%) | 指令级并行度 |
| 量化支持 | FP8/INT8/INT4 | 硬件加速支持度 |

**覆盖算子**:
- GEMM (通用矩阵乘法)
- FlashAttention (高效注意力)
- LayerNorm / RMSNorm (归一化)
- RoPE (旋转位置编码)
- SwiGLU (激活函数)
- Softmax

### 通信与互联 (02_communication_perf.py)

| 维度 | 指标 | 说明 |
|------|------|------|
| 点对点带宽 | GB/s | 卡间 P2P 通信速率 |
| 通信延迟 | µs (mean, p99) | 端到端延迟及抖动 |
| All-Reduce 带宽 | GB/s | 集合通信效率 |
| All-Gather 带宽 | GB/s | 多对多通信效率 |
| Broadcast 带宽 | GB/s | 一对多通信效率 |
| Reduce-Scatter 带宽 | GB/s | 分散规约效率 |
| 扩展效率 | Scaling % | 增加 GPU 后的线性增长比例 |
| 鲁棒性 | 成功率/死锁 | 极端并发下的稳定性 |

### CUDA Samples 代表性算子 (04_cuda_samples_ops.py)

基于 NVIDIA [cuda-samples](https://github.com/NVIDIA/cuda-samples) 仓库中的代表性算子实现，
每个算子对应一个具体的 cuda-sample，确保评测基准的权威性和可比性。

| cuda-sample 路径 | 算子名称 | 对应 benchmark | 关键优化技术 |
|-----------------|---------|---------------|-------------|
| `3_CUDA_Features/cudaTensorCoreGemm` | FP16 TensorCore GEMM | `tensorcore_gemm_fp16` | WMMA API, shared memory tiling, SKEW_HALF |
| `3_CUDA_Features/bf16TensorCoreGemm` | BF16 TensorCore GEMM | `tensorcore_gemm_bf16` | BF16 WMMA, 16x16x16 tile |
| `3_CUDA_Features/immaTensorCoreGemm` | INT8 TensorCore GEMM | `imma_gemm_int8` | IMMA, 8x8x16 INT8 TensorCore |
| `3_CUDA_Features/tf32TensorCoreGemm` | TF32 TensorCore GEMM | `tf32_gemm` | TF32 TensorCore, FP32 accumulation |
| `4_CUDA_Libraries/batchCUBLAS` | Batched GEMM | `batched_gemm_*` | cublasGemmBatchedEx, strided batch |
| `4_CUDA_Libraries/simpleCUFFT` | FFT | `fft` | cuFFT 1D/2D, batched transforms |
| `4_CUDA_Libraries/conjugateGradient` | SpMM | `spmm` | cuSPARSE SpMV/SpMM, CSR format |
| `2_Concepts_and_Techniques/reduction` | Parallel Reduction | `reduction_*` | warp shuffle, __reduce_add_sync (SM80+) |
| `2_Concepts_and_Techniques/scan` | Prefix Sum | `scan_*` | Blelloch scan, shared memory |
| `2_Concepts_and_Techniques/histogram` | Histogram | `histogram` | atomicAdd, per-block histograms |
| `2_Concepts_and_Techniques/convolutionSeparable` | 2D Convolution | `conv2d_*` | separable filter, shared memory tiles |
| `6_Performance/transpose` | Matrix Transpose | `transpose_*` | shared memory with bank conflict avoidance |
| `5_Domain_Specific/convolutionFFT2D` | FFT Convolution | `conv_fft_*` | frequency domain multiply |
| `5_Domain_Specific/bilateralFilter` | Bilateral Filter | `bilateral_filter` | edge-preserving smoothing |
| `5_Domain_Specific/SobelFilter` | Sobel Edge Detection | `sobel_filter` | texture memory, separable kernels |

**分类运行**:
```bash
python3 04_cuda_samples_ops.py --categories features    # TensorCore GEMM 变体
python3 04_cuda_samples_ops.py --categories libraries   # CUBLAS/CUFFT/cuSPARSE
python3 04_cuda_samples_ops.py --categories concepts    # reduction/scan/histogram/conv
python3 04_cuda_samples_ops.py --categories performance # transpose
python3 04_cuda_samples_ops.py --categories domain      # FFT conv / bilateral / Sobel
```

### 报告对比 (03_run_comparison.py)

自动汇总多设备的评测结果，生成包含以下内容的 Markdown 报告：
- 设备硬件规格对比
- 算子性能对比表
- 量化支持对比
- 通信带宽对比
- 集群扩展效率对比
- 与参考硬件（A100, H100, Ascend 910B 等）的理论对比

## 设备后端支持

| 设备 | 后端 | 通信库 | 检测方式 |
|------|------|--------|---------|
| NVIDIA GPU | `cuda` | NCCL | `torch.cuda.is_available()` |
| 阿里 PPU | `ppu` | HCCL | `torch_ppu.is_available()` |
| 华为 Ascend | `ascend` | HCCL | `torch.npu.is_available()` |
| 寒武纪 MLU | `cambricon` | CNCL | `torch_mlu.is_mlu_available()` |
| 海光 DCU | `dcu` | RCCL | `torch.backends.hip` |
| 摩尔线程 | `musa` | MCCL | `torch_musa.is_available()` |

## 参考硬件规格

| 平台 | BF16 TFLOPS | 内存带宽 (GB/s) | 互联方式 | 互联带宽 (GB/s) |
|------|------------|----------------|---------|----------------|
| NVIDIA A100 | 312 | 2,039 | NVLink 3.0 | 600 |
| NVIDIA H100 | 989 | 3,350 | NVLink 4.0 | 900 |
| NVIDIA H20 | 148 | 900 | NVLink 4.0 | 900 |
| 阿里 PPU | 390 | 1,200 | HCCS | ~300 |
| 华为 Ascend 910B | 313 | 1,200 | HCCS | 392 |
| 寒武纪 MLU370-X8 | 256 | 819 | MLU-Link | 200 |
| 海光 DCU Z100 | 280 | 1,600 | xGMI/PCIe | 200 |
| 摩尔线程 S4000 | 200 | 1,000 | MUSA Link | 400 |

## 输出示例

算子性能结果 (`results/operator_perf.json`):
```json
{
  "device_info": {
    "device_type": "cuda",
    "device_name": "NVIDIA A100-SXM4-80GB",
    "memory_total_gb": 85.9,
    "compute_capability": "8.0",
    "sm_count": 108
  },
  "operator_results": [
    {
      "operator": "GEMM",
      "dtype": "bf16",
      "batch_size": 1,
      "hidden_size": 4096,
      "seq_len": 2048,
      "latency_us_mean": 823.5,
      "throughput_tflops": 248.3,
      "memory_bandwidth_utilization_pct": 12.1,
      "calc_mem_ratio": 1365.3
    }
  ],
  "quantization_support": {
    "fp8_e4m3": {"supported": false, "hw_accelerated": false},
    "int8": {"supported": true, "hw_accelerated": true}
  }
}
```

## 注意事项

1. **预热运行**: 每个算子测试包含 5 次预热运行，确保 GPU 频率稳定
2. **多次采样**: 每个配置测量 20 次，取均值、标准差、P50、P99
3. **温度影响**: 长时间运行可能导致降频，建议在每次评测前重置 GPU 状态
4. **多卡通信**: 需要确保所有卡的 NVLink/互联拓扑一致
5. **国产卡兼容性**: 部分国产卡的后端 API 可能与 CUDA 有差异，需要适配

## 常见问题

**Q: 报错 "No distributed backend available"**
A: 安装对应的通信库，如 `pip install torch` (NCCL 已内置)，或安装 `torch-npu` (HCCL)。

**Q: FlashAttention 跑不通？**
A: 国产卡可能不支持 flash-attn 库，脚本会自动回退到 PyTorch 原生 SDPA。

**Q: 多节点通信连不上？**
A: 检查防火墙（MASTER_PORT 默认 29500）、IB/RoCE 网络连通性、NCCL_IB_DISABLE 环境变量。

**Q: 如何在阿里云 PPU 上运行？**
A: 确保已安装 `torch-ppu`，然后 `python3 01_operator_perf.py --device ppu`。
