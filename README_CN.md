# RAG-Contamination-CodeGen

**检索增强生成中的检索污染现象：蒸馏小模型代码生成场景研究**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

> 语言切换: [English](README.md) | [中文](README_CN.md)

---

## 目录

- [研究背景与动机](#研究背景与动机)
- [实验方向](#实验方向)
- [方法论](#方法论)
- [仓库结构](#仓库结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [复现实验](#复现实验)
- [核心结果](#核心结果)
- [引用](#引用)

---

## 研究背景与动机

### 问题陈述

随着大语言模型的开源生态成熟，轻量化模型（参数量 ≤ 7B）在消费级硬件上的本地部署已成为广泛趋势。为弥补小模型在领域知识方面的不足，检索增强生成（Retrieval-Augmented Generation, RAG）被普遍采用作为性能增强手段。

然而，现有RAG系统的设计与评估主要围绕大规模模型（数十亿至数千亿参数）展开，其在小型模型上的效能迁移**缺乏系统性验证**。本研究观察到：在代码生成任务中，直接为蒸馏后的小模型接入通用RAG管道，并未带来预期的性能增益，甚至在某些场景下出现了输出质量退化。

我们命名此现象为：

> **检索污染（Retrieval Contamination）** — 由语义不匹配或冲突的检索上下文引发的生成质量退化，尤其影响那些有限能力在"内部参数化知识"与"外部检索输入"之间做出判断的模型。

### 实践意义

- **消费级GPU部署** (RTX 3060/4060, 8GB VRAM)：蒸馏+量化的小模型是唯一可行的本地方案
- **RAG已成为默认配置**：工程师普遍为小模型添加向量数据库，但未验证RAG是否真正有效
- **缺乏表征研究**：现有RAG研究聚焦于大模型(7B+)，小模型RAG行为缺乏系统表征

### 学术空白定位

| 现有工作 | 领域 | 空白 |
|---|---|---|
| RAG优化 (Self-RAG, RAFT) | 大模型 (7B+) | 不适用于SLM |
| 小模型能力 (MiniRAG, Pandey 2026) | QA领域 | 非代码生成；不区分蒸馏/基座 |
| 代码RAG (CodeRAG-Bench, Repoformer) | GPT-4/DeepSeekCoder 6.7B+ | 不涉及知识蒸馏 |
| 蒸馏+RAG (DRAG, ACL 2025) | 正向：教会SLM使用RAG | 我们表征*失败模式* |

**我们的位置（空白）**: ★ 蒸馏小模型 × RAG退化 × 代码生成 × 失败表征

---

## 实验方向

### 核心研究问题

> **检索增强生成是否可靠地提升了蒸馏小模型的代码生成能力？如果不能，在什么条件下会失败？**

### 子问题

1. 检索多样性如何影响参数受限模型的生成质量？
2. 简单的提示词策略能否缓解RAG引起的退化？
3. 观察到的退化是否跨模型厂商一致？
4. 哪些检索失败机制是蒸馏小模型特有的？

### 实验设计（三组架构）

| 组别 | 目的 | 模型 | 模式 |
|---|---|---|---|
| **对照组** | 建立No-RAG基线 | Qwen2.5-3B-Distill, Phi-3-mini, Gemma-2-2B, Qwen2.5-7B | No-RAG |
| **交叉验证组** | 隔离RAG vs RAG+遗忘效应 | Qwen2.5-3B-Distill, Qwen2.5-3B-Instruct (基座) | RAG, RAG+Forgetting |
| **外部验证组** | 验证跨模型一致性 | Phi-3-mini-4k, Gemma-2-2B, Qwen2.5-7B | RAG |

### 五种消融配置

| 配置 | 时间衰减 | 反馈评分 | MMR多样性 |
|---|---|---|---|
| 完整系统 | ✓ | ✓ | ✓ |
| 无时间衰减 | ✗ | ✓ | ✓ |
| 无反馈评分 | ✓ | ✗ | ✓ |
| 无多样性过滤 | ✓ | ✓ | ✗ |
| 基线 (Vanilla ChromaDB) | ✗ | ✗ | ✗ |

### 五种提示策略

| 策略 | 描述 |
|---|---|
| A. 当前策略 (参考) | 标准RAG提示 |
| B. 防御性提示 | "优先使用你自己的知识。检索到的上下文仅作参考。" |
| C. 最小化提示 | 仅附加上下文，无额外指令 |
| D. 冲突感知 | "如果检索到的代码与你的知识冲突，请忽略它。" |
| E. 无RAG (对照) | 纯蒸馏模型，无检索 |

---

## 方法论

### 系统架构

```
用户查询
    │
    ▼
┌─────────────────────────┐
│  ChromaDB 向量数据库     │  ← 303,869代码片段，来自15个开源仓库
│  (ONNX MiniLM-L6-v2)    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  记忆权重评分            │  W = α·W_temporal + β·W_feedback + γ·W_redundancy
│  (艾宾浩斯遗忘曲线)       │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  MMR 多样性重排          │  MMR(doc) = λ·sim(query,doc) - (1-λ)·max_sim(doc,selected)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  生命周期状态分类        │  Active → Warm → Cold → Archived → Suppressed
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LLM 生成               │  Qwen2.5-3B-Distill (Q4_K_M GGUF, llama.cpp)
│  (配合提示策略)          │
└─────────────────────────┘
```

### 记忆生命周期评分

```
W_total = α·W_temporal + β·W_feedback + γ·W_redundancy

W_temporal   = exp(-λ · Δt_days)                    [艾宾浩斯衰减, λ=0.1]
W_feedback   = (success + prior·k) / (total + k)     [贝叶斯平滑, k=2]
W_redundancy = 1 - max_sim_to_retrieved_set          [冗余惩罚]

超参数: α=0.5, β=0.3, γ=0.2
```

生命周期状态阈值：

| 状态 | 权重范围 | 衰减系数 | 检索行为 |
|---|---|---|---|
| 活跃 (Active) | W > 0.8 | 1.0 | 完全优先 |
| 温热 (Warm) | 0.5 < W ≤ 0.8 | 0.8 | 正常 |
| 冷却 (Cold) | 0.2 < W ≤ 0.5 | 0.5 | 降低优先级 |
| 归档 (Archived) | W ≤ 0.2 | 0.2 | 存储但不检索 |
| 抑制 (Suppressed) | W ≤ 0.05 | 0.05 | 软删除 |

### MMR 多样性重排

```
MMR(doc) = λ_mmr · sim(query, doc) - (1-λ_mmr) · max_sim(doc, 已选集)
λ_mmr = 0.7
```

### 模型与量化

| 组件 | 规格 |
|---|---|
| 基座模型 | Qwen2.5-3B |
| 蒸馏数据 | 50K代码示例，多阶段训练 |
| 量化方案 | GGUF Q4_K_M (4-bit) |
| 推理引擎 | llama.cpp |
| 上下文窗口 | 2048 tokens |

### 向量数据库

- **总片段数**: 303,869，来自15个开源GitHub仓库
- **嵌入模型**: ONNX all-MiniLM-L6-v2 (384维)
- **存储引擎**: ChromaDB persistent client
- **集合结构**: summary (仓库级), chunks (文件级), details (函数级)
- **分块策略**: Python使用AST，其他语言使用正则，内容哈希去重

### 评估基准

- **HumanEval** (OpenAI, 164题): 函数补全任务，基于执行的评估
- **MBPP** (Google, 974题): 入门级Python编程，基于断言的测试
- **评估工具**: EvalPlus (base + plus测试套件，严格Pass@1)

---

## 仓库结构

```
rag-contamination-codegen/
├── README.md                        # 英文说明
├── README_CN.md                     # 中文说明（本文件）
├── requirements.txt                 # Python依赖
├── LICENSE                          # MIT许可证
│
├── src/
│   ├── rag/                         # RAG系统核心
│   │   ├── search_engine.py         # ChromaDB三层分级搜索 (LRU缓存, 并行查询)
│   │   ├── forgetting_retrieval.py  # 艾宾浩斯衰减 + MMR重排 + 生命周期状态
│   │   ├── build_code_vector_db.py  # 从GitHub仓库构建向量数据库
│   │   └── expand_vector_db.py      # 扩展向量数据库
│   │
│   ├── training/                    # 蒸馏训练
│   │   ├── train_distill.py         # 多阶段蒸馏训练 (50K样本, LoRA)
│   │   ├── prepare_distill_data.py  # 蒸馏数据集准备
│   │   └── build_coding_dataset.py  # 构建代码训练数据集
│   │
│   └── evaluation/                  # 基准测试与评估
│       ├── config.py                # 中心化配置 (模型、提示词、路径)
│       ├── run_benchmark.py         # 完整基准测试管线 (多模型、多模式)
│       ├── run_ablation.py          # 消融实验运行器 (5配置 × 34问题)
│       ├── e2e_validate.py          # 端到端验证 (EvalPlus集成)
│       └── eval_coding.py           # 代码任务评估 (Pass@1, 语法, 执行)
│
├── analysis/                        # 可视化与分析
│   ├── generate_figures.py          # 从实验数据生成论文图表
│   └── paper_figures_nature.py      # Nature/CNS投稿级图表
│
├── paper/                           # 论文资源
│   ├── paper.pdf                    # 编译后的论文PDF
│   ├── paper_draft.txt              # 完整文本草稿
│   ├── paper_narrative.md           # 叙事结构与写作指南
│   └── figures/                     # 生成的图表 (PNG + PDF)
│
└── results/                         # 冻结的实验数据
    ├── statistics.json              # 全部实验统计 (2026-05-25冻结)
    ├── ablation_qwen2.5-3b.csv      # 消融原始数据
    ├── summary_qwen2.5-3b.json      # 摘要统计
    └── experiment_report.md         # 详细实验报告
```

---

## 环境配置

```bash
# 克隆仓库
git clone https://github.com/<user>/rag-contamination-codegen.git
cd rag-contamination-codegen

# 安装依赖
pip install -r requirements.txt

# LLM推理引擎 (选其一):
# 方案A: llama-cpp-python + CUDA加速
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 方案B: CPU-only
pip install llama-cpp-python
```

### 硬件需求

- Python 3.10+
- 8GB+ 显存 (GPU推理) 或 16GB+ 内存 (CPU推理)
- ChromaDB向量数据库 (预构建或自行构建)

---

## 快速开始

### 1. 构建向量数据库（可选 - 提供预构建数据库）

```python
from src.rag.build_code_vector_db import build_vector_db

build_vector_db(
    output_dir="./vector_db",
    repos=["pallets/flask", "psf/requests", "sqlalchemy/sqlalchemy"],
    max_chunks_per_repo=5000
)
```

### 2. 运行单次RAG查询

```python
from src.rag.search_engine import SearchEngine

engine = SearchEngine(db_path="./vector_db")
results = engine.search("二叉树搜索验证", top_k=5)
print(engine.format_results(results))
```

### 3. 测试遗忘感知检索

```python
from src.rag.forgetting_retrieval import ForgettingRetriever

retriever = ForgettingRetriever(
    db_path="./vector_db",
    records_path="./forgetting_records.json",
    forgetting_threshold=0.7,
    forgetting_weight=0.3
)

result = retriever.retrieve("验证BST插入", top_k=3, use_forgetting=True)
print(f"上下文 ({result['filtered_count']}条被过滤):\n{result['context']}")
```

### 4. 端到端验证

```bash
python src/evaluation/e2e_validate.py
```

---

## 复现实验

### 实验1: 消融研究

```bash
python src/evaluation/run_ablation.py \
    --model Qwen2.5-3B-Distill \
    --problems 34 \
    --configs full,no-temporal,no-feedback,no-diversity,baseline
```

### 实验2: 完整基准测试

```bash
python src/evaluation/run_benchmark.py \
    --models Qwen2.5-3B-Distill,Qwen2.5-3B-Instruct,Phi-3-mini-4k,Gemma-2-2B \
    --modes no-rag,rag,rag-forgetting \
    --datasets humaneval,mbpp
```

### 实验3: 提示策略对比

```bash
python src/evaluation/run_benchmark.py \
    --models Qwen2.5-3B-Distill \
    --modes rag \
    --prompt-strategies A,B,C,D,E
```

### 生成论文图表

```bash
python analysis/generate_figures.py
```

---

## 核心结果

### 消融实验结果 (34道编程题)

| 配置 | Pass@1 | 平均延迟 | 平均多样性 |
|---|---|---|---|
| **无多样性过滤** | **0.794** | 1.81s | 0.725 |
| 完整系统 | 0.765 | 1.74s | 0.905 |
| 基线 ChromaDB | 0.765 | 1.58s | 0.533 |
| 无时间衰减 | 0.735 | 1.50s | 0.905 |
| 无反馈评分 | 0.706 | 1.50s | 0.905 |

*参考值: 原始 Qwen2.5-3B (未微调): 0.676 Pass@1*

### 五大核心发现

1. **RAG对蒸馏模型净效果为零**: 仅蒸馏带来+11.8%增益 (0.676 → 0.794)。在蒸馏基础上添加RAG，零额外收益。

2. **检索多样性与Pass@1呈非单调关系**: 最高多样性(0.905)**不**产生最佳性能。最优多样性区间: 0.70-0.75。呈现倒U型曲线。

3. **防御性提示完全缓解污染**: 从0.735 (标准RAG) → 0.765 (防御性提示)，恢复至无RAG基线水平。成本为零。

4. **中等难度任务最敏感**: "帮助"和"损害"案例都发生在中等难度 — 模型的"最近发展区"。简单题模型已掌握，难题模型无法利用检索上下文。

5. **RAG改变55.9%的答案**: 与No-RAG相比，仅约70%文本重叠 (Jaccard)，表明RAG从根本上改变了生成方式，而非仅做补充。

### 污染分析 (34题详细)

| 类别 | 数量 | 百分比 |
|---|---|---|
| 中立 | 32 | 94.1% |
| RAG有帮助 | 1 | 2.9% |
| RAG有损害 | 1 | 2.9% |
| **净效应** | **0** | — |

### 提示策略对比

| 策略 | Pass@1 | vs No-RAG |
|---|---|---|
| **B. 防御性** | **0.765** | =0.000 |
| E. 无RAG (对照) | 0.765 | (基线) |
| A. 当前策略 | 0.735 | -0.029 |
| C. 最小化 | 0.735 | -0.029 |
| D. 冲突感知 | 0.706 | -0.059 |

---

## 引用

```bibtex
@article{rag-contamination-codegen,
  title={{Retrieval Contamination in Long-Term RAG Systems for Code Generation}},
  author={},
  journal={},
  year={2026},
  note={预印本}
}
```

---

## 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

如有问题或合作意向，请在本仓库提交 Issue。
