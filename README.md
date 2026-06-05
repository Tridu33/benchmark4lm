<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-green?style=flat-square&logo=nvidia&logoColor=white" alt="NVIDIA"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome"/>
</p>

<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=3000&pause=1000&center=true&vCenter=true&width=600&lines=benchmark4lm;LLM+%E8%AE%AD%E6%8E%A8%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7%E7%AE%B1" alt="Typing SVG" />
</h1>

<p align="center">
  <b>一站式大模型训练 & 推理性能剖析工具箱</b><br/>
  <sub>GPU/CPU 全链路监控 · PyTorch Profiler 算子级分析 · Nsight Systems 系统级追踪 · LangFuse 全链路追踪</sub>
</p>

---

## 📖 项目简介

**benchmark4lm** 是一套面向大模型（LLM / VLM / Speech / Video）的 **训练与推理性能分析工具集**。它提供了从硬件层（GPU 利用率、显存、PCIe 带宽）到框架层（算子耗时、KV-Cache、吞吐量）再到应用层（LangFuse 追踪、RAG 质量评估）的 **全栈性能剖析能力**。

> 🎯 **一句话目标**：跑一个脚本，拿到一份包含瓶颈分析和优化建议的完整性能报告。

```
┌─────────────────────────────────────────────────────────────────────┐
│                      benchmark4lm 工作流                           │
│                                                                     │
│   ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌────────┐ │
│   │ run_*.sh │───▶│ profile_*.py │───▶│core_profiler│───▶│ out/   │ │
│   │ 启动任务  │    │  挂载监控     │    │  采集&聚合  │    │ 报告   │ │
│   └──────────┘    └──────────────┘    └────────────┘    └────────┘ │
│       │                  │                    │                │    │
│   单卡/分布式      PyTorch Profiler      GPU/CPU 采样     Markdown │
│   任意训练/推理    Nsight Systems        LangFuse 追踪    CSV     │
│   框架脚本         内存/线程监控          算子级分析      Trace    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ 核心特性

<table>
<tr>
<td width="50%" valign="top">

### 🔍 多维度性能采集

| 指标类别 | 采集项 |
|---------|--------|
| 🚀 **推理** | TTFT、TPOT、吞吐量、KV-Cache、最大并发数、生成 FPS、FLOP/s |
| 🏋️ **训练** | Loss 收敛曲线、梯度范数、学习率调度、steps/sec、samples/sec |
| 🖥️ **硬件** | GPU 利用率、显存峰值/均值、CPU 利用率、PCIe 带宽、温度/功耗 |
| 🔬 **深度** | PyTorch Profiler 算子分析、Nsight `.nsys-rep`、Memcpy 耗时、CUDA Kernel 分布 |
| 🔗 **追踪** | LangFuse 全链路追踪与成本、Phoenix 数据漂移、RAGAS 质量评估 |

</td>
<td width="50%" valign="top">

### 🧩 广泛框架适配

```
推理引擎                        训练框架
──────────                      ──────────
⚡ vLLM                         🤗 HF Transformers + Accelerate
⚡ SGLang                       🚀 DeepSpeed
⚡ TensorRT-LLM                 🏔️ Megatron-LM
⚡ Ollama                       🦙 LLaMA-Factory
⚡ Llamafile                    🪓 Axolotl
⚡ kTransformers                🌊 Colossal-AI
⚡ LMDeploy                     🔧 PEFT (LoRA/QLoRA/Prefix/Prompt)
```

### 🤖 全模态覆盖

```
📝 语言大模型 (LLM)       🖼️ 多模态视觉 (VLM)
🔊 语音大模型 (Speech)    🎬 视频生成模型 (Video)
📚 RAG 应用               🎨 SFT / PEFT 微调
```

</td>
</tr>
</table>

---

## 🏗️ 项目架构

```
benchmark4lm/
│
├── 🧱 core_profiler.py           ← 核心引擎：GPU/CPU监控 · Profiler · Nsight · LangFuse · 报告生成
├── 📦 requirements.txt           ← Python 依赖清单
│
├── 📂 infer/                     ← ── 推理性能分析 ──────────────────────
│   ├── 📝 llm/                   ← 语言大模型 (vLLM / SGLang / Ollama / Llamafile / kTransformers / TRT-LLM)
│   │   ├── profile_infer.py
│   │   ├── run_infer.sh
│   │   └── README.md
│   ├── 🖼️ imglm/                 ← 多模态视觉模型 (LLaVA / Qwen-VL / vLLM Vision)
│   │   ├── profile_infer.py
│   │   ├── run_infer.sh
│   │   └── README.md
│   ├── 🔊 soundlm/               ← 语音大模型 (Whisper / SpeechT5 / API)
│   │   ├── profile_infer.py
│   │   ├── run_infer.sh
│   │   └── README.md
│   ├── 🎬 videolm/               ← 视频生成模型 (CogVideo / Diffusers / AnimateDiff)
│   │   ├── profile_infer.py
│   │   ├── run_infer.sh
│   │   └── README.md
│   └── 📚 rag/                   ← RAG 应用监控 (LangFuse + Phoenix · LangChain / LlamaIndex)
│       ├── profile_rag.py
│       ├── run_rag.sh
│       └── README.md
│
├── 📂 train/                     ← ── 训练性能分析 ──────────────────────
│   ├── 📝 llm/
│   │   ├── sft/                  ← LLM 全量微调 (HF / DeepSpeed / Megatron / LLaMA-Factory / Axolotl / ColossalAI)
│   │   └── peft/                 ← LLM 高效微调 (LoRA / QLoRA / Prefix Tuning / Prompt Tuning)
│   ├── 🖼️ imglm/
│   │   ├── sft/                  ← VLM 全量微调
│   │   └── peft/                 ← VLM 高效微调
│   ├── 🔊 soundlm/
│   │   ├── sft/                  ← 语音模型全量微调
│   │   └── peft/                 ← 语音模型高效微调
│   └── 🎬 videolm/
│       ├── sft/                  ← 视频模型全量微调
│       └── peft/                 ← 视频模型高效微调
│
└── 📂 out/                       ← ── 输出目录 ──────────────────────────
    ├── result.md                 ← 📊 综合性能报告（含瓶颈分析与优化建议）
    ├── *.json                    ← Chrome Trace 时间线
    ├── *.nsys-rep                ← Nsight Systems 报告
    └── *.csv                     ← 时间序列监控数据
```

> 📌 每个子目录统一包含 `profile_*.py`（Python 性能分析脚本）、`run_*.sh`（Bash 启动脚本）、`README.md`（详细说明文档），共计 **41 个文件**（14 Python + 13 Shell + 13 README + 1 核心模块）。

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 克隆项目
git clone <repo-url> && cd benchmark4lm

# 安装 Python 依赖
pip install -r requirements.txt

# 确保 NVIDIA 驱动 & CUDA 可用
nvidia-smi
```

> ⚠️ 部分可选依赖（TensorRT-LLM、Ollama、Llamafile、Axolotl、kTransformers）需单独安装，详见 `requirements.txt` 注释。

### 2️⃣ 运行性能分析

每个分析脚本通过 `--task-script` 参数接收一个 **Bash 启动脚本**，该脚本负责启动实际的训练或推理任务（支持单卡 / 多卡分布式）。

```bash
# 📝 示例：分析 LLM 推理性能（vLLM）
cd infer/llm
python profile_infer.py --task-script ./run_infer.sh

# 🏋️ 示例：分析 LLM SFT 全量训练
cd train/llm/sft
python profile_train.py --task-script ./run_train.sh

# 🔗 示例：分析 LLM LoRA 微调
cd train/llm/peft
python profile_train.py --task-script ./run_train.sh

# 📚 示例：RAG 应用全链路追踪
cd infer/rag
python profile_rag.py --task-script ./run_rag.sh
```

### 3️⃣ 查看报告

所有分析结果统一输出到 `out/` 目录：

```bash
# 查看综合 Markdown 报告
cat out/result.md

# 用 Chrome 打开 Trace 时间线
chrome://tracing  # 加载 out/*.json

# 用 Nsight Systems GUI 打开系统报告
nsys-ui out/*.nsys-rep
```

---

## 📊 报告输出示例

性能报告 `out/result.md` 包含以下章节：

```
┌──────────────────────────────────────────────────────┐
│                    性能分析报告                       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. 🖥️  硬件环境    — GPU 型号、驱动、CUDA 版本      │
│  2. 📦  软件环境    — Python/PyTorch/框架版本        │
│  3. ⏱️  推理指标    — TTFT / TPOT / 吞吐量 / FPS    │
│  4. 📈  训练指标    — Loss 曲线 / steps/sec / 梯度   │
│  5. 💾  资源监控    — GPU 利用率 / 显存 / PCIe 带宽  │
│  6. 🔬  算子分析    — Top-K 耗时算子 / Kernel 分布   │
│  7. 🔗  链路追踪    — LangFuse Trace / 成本分析      │
│  8. 🐛  瓶颈分析    — 最耗时算子 / 显存碎片诊断      │
│  9. 💡  优化建议    — 针对性调优方案                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ 核心模块详解 (`core_profiler.py`)

`core_profiler.py` 是所有脚本的共享基础设施，提供以下能力：

| 模块 | 功能 | 底层技术 |
|------|------|---------|
| 🖥️ **GPU 监控** | 利用率、显存（峰值/均值）、温度、功耗、PCIe 带宽 | `pynvml` (NVML) |
| 💻 **CPU 监控** | 利用率、内存、线程数 | `psutil` |
| 🔥 **PyTorch Profiler** | 算子级耗时、内存分配、CUDA Kernel 分析 | `torch.profiler` |
| 🔎 **Nsight Systems** | GPU/CPU 工作负载分布、内核启动延迟、Memcpy 时序 | `nsys` CLI |
| 🔗 **LangFuse 追踪** | LLM 全链路 Trace、Token 用量、延迟、成本 | `langfuse` SDK |
| 📊 **报告生成** | 聚合所有指标，输出 Markdown 报告 + CSV + Trace JSON | 自研聚合引擎 |

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  NVML 采样   │     │  psutil 采样 │     │  Profiler   │
│  (GPU 指标)  │     │  (CPU 指标)  │     │  (算子级)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│              core_profiler 聚合引擎                  │
│                                                     │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Markdown  │  │   CSV    │  │  Chrome Trace    │ │
│  │  报告     │  │ 时序数据  │  │  JSON + Nsight  │ │
│  └───────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 自定义 Bash 启动脚本

`run_*.sh` 是用户适配自身任务的入口，以下是典型模板：

**推理示例 (`infer/llm/run_infer.sh`)**

```bash
#!/usr/bin/env bash
# 使用 vLLM 启动 LLM 推理服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

**训练示例 (`train/llm/sft/run_train.sh`)**

```bash
#!/usr/bin/env bash
# 使用 DeepSpeed ZeRO-3 分布式训练
deepspeed --num_gpus=8 train.py \
    --model_name_or_path meta-llama/Llama-3-8B \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --deepspeed ds_config.json
```

> 💡 **提示**：`profile_*.py` 会自动挂载监控器并采集数据，您只需关注 `run_*.sh` 中的任务启动命令。

---

## 📋 支持矩阵

### 推理引擎

| 引擎 | LLM | VLM | Speech | Video | 特色指标 |
|------|:---:|:---:|:------:|:-----:|---------|
| **vLLM** | ✅ | ✅ | — | — | PagedAttention、KV-Cache 命中率 |
| **SGLang** | ✅ | — | — | — | RadixAttention、调度延迟 |
| **TensorRT-LLM** | ✅ | — | — | — | Tensor Core 利用率 |
| **Ollama** | ✅ | — | — | — | 本地推理延迟、内存映射 |
| **Llamafile** | ✅ | — | — | — | CPU 推理、量化效率 |
| **kTransformers** | ✅ | — | — | — | CPU+GPU 混合推理 |
| **LMDeploy** | ✅ | ✅ | — | — | TurboMind 引擎 |

### 训练框架

| 框架 | SFT | PEFT | 分布式 | 特色能力 |
|------|:---:|:----:|:-----:|---------|
| **HF Transformers + Accelerate** | ✅ | ✅ | FSDP / DDP | 最广泛的生态支持 |
| **DeepSpeed** | ✅ | ✅ | ZeRO 1/2/3 | 超大模型分布式训练 |
| **Megatron-LM** | ✅ | — | TP + PP + DP | 千亿参数级训练 |
| **LLaMA-Factory** | ✅ | ✅ | 多卡 | 一站式微调平台 |
| **Axolotl** | ✅ | ✅ | 多卡 | 配置化训练 |
| **Colossal-AI** | ✅ | ✅ | Gemini / Hybrid | 自动并行策略 |
| **PEFT** | — | ✅ | — | LoRA / QLoRA / Prefix / Prompt Tuning |

---

## 📐 监控指标详解

### 推理场景

```
┌─────────────────────────────────────────────────────────────────┐
│                        推理请求生命周期                          │
│                                                                 │
│   Request ──▶ [排队] ──▶ [Prefill] ──▶ [Decode × N] ──▶ Done   │
│                 │          │              │                      │
│                 ▼          ▼              ▼                      │
│              Queue     TTFT          TPOT                        │
│              Time   (首 Token)    (每 Token)                     │
│                                                                 │
│   📊 吞吐量 = tokens/sec    📊 KV-Cache 使用率                  │
│   📊 FLOP/s = 计算密度      📊 最大并发请求数                    │
│   📊 生成 FPS (视频)        📊 显存峰值 / 均值                   │
└─────────────────────────────────────────────────────────────────┘
```

### 训练场景

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练 Step 生命周期                        │
│                                                                 │
│   [数据加载] ──▶ [前向传播] ──▶ [反向传播] ──▶ [优化器] ──▶ Step│
│       │              │              │             │              │
│       ▼              ▼              ▼             ▼              │
│   DataLoader     Activation     Gradient     LR Schedule        │
│   吞吐量         Memory         范数         更新步长            │
│                                                                 │
│   📊 steps/sec              📊 samples/sec                      │
│   📊 Loss 收敛曲线          📊 梯度范数趋势                      │
│   📊 GPU 利用率时序         📊 显存碎片分析                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 依赖一览

| 类别 | 核心依赖 | 版本要求 |
|------|---------|---------|
| 🐍 **基础** | numpy, pandas, rich, click, matplotlib, seaborn | 详见 requirements.txt |
| 🔥 **PyTorch** | torch, transformers, accelerate, datasets, peft, trl | torch ≥ 2.1 |
| 🚀 **分布式** | deepspeed, megatron-lm | deepspeed ≥ 0.12 |
| ⚡ **推理** | vllm, sglang | vllm ≥ 0.3 |
| 🔬 **分析** | py-spy, line-profiler, memory-profiler, nvtx, tensorboard, wandb | — |
| 🔗 **追踪** | langfuse, arize-phoenix, openinference | langfuse ≥ 2.30 |
| 📚 **RAG** | langchain, llama-index | langchain ≥ 0.2 |
| 🖥️ **系统** | psutil, pynvml, GPUtil, py3nvml | psutil ≥ 5.9 |

---

## 🤝 贡献指南

欢迎贡献！以下是推荐的开发流程：

```bash
# 1. Fork 并克隆
git clone https://github.com/<your-fork>/benchmark4lm.git

# 2. 创建特性分支
git checkout -b feature/my-awesome-profiling

# 3. 开发并测试
python profile_infer.py --task-script ./run_infer.sh

# 4. 提交 PR
```

### 贡献方向

- 🔌 适配更多推理引擎 / 训练框架
- 📊 新增可视化图表类型
- 🧪 增加自动化测试 & CI
- 📖 完善文档 & 使用示例

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。

---

<p align="center">
  <b>如果这个项目对你有帮助，请给一颗 ⭐ Star 支持！</b>
</p>
