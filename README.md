# RAG-Contamination-CodeGen

**Retrieval Contamination in Distilled Small Language Models for Code Generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

> Language: [English](README.md) | [中文](README_CN.md)

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Experiment Direction](#experiment-direction)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproduce Experiments](#reproduce-experiments)
- [Key Results](#key-results)
- [Citation](#citation)

---

## Background & Motivation

### The Problem

Small Language Models (SLMs, ≤7B parameters) deployed on consumer GPUs are increasingly paired with Retrieval-Augmented Generation (RAG) as a default enhancement strategy. The prevailing assumption: *RAG always helps by filling knowledge gaps.*

**We challenge this assumption.** When a distilled model (whose weights already encode domain-specific knowledge via knowledge distillation) is combined with a generic RAG pipeline, the retrieved context can introduce **conflicting code patterns** that overwrite the model's internally stored knowledge. We term this phenomenon:

> **Retrieval Contamination** — RAG-induced performance degradation caused by semantically mismatched or conflicting retrieved context, particularly affecting models with limited capacity to adjudicate between internal knowledge and external input.

### Why This Matters

- **Consumer GPU deployment** (RTX 3060/4060, 8GB VRAM): Distilled + quantized SLMs are the only practical option
- **RAG is ubiquitous**: Practitioners routinely add vector databases without verifying whether RAG helps
- **No prior characterization**: Existing RAG research focuses on large models (7B+); small-model RAG behavior is uncharted

### Research Gap

| Prior Work | Domain | Gap |
|---|---|---|
| RAG optimization (Self-RAG, RAFT) | Large models (7B+) | Doesn't transfer to SLMs |
| Small model capability (MiniRAG, Pandey 2026) | QA domain | Not code generation; doesn't distinguish distilled vs base |
| Code RAG (CodeRAG-Bench, Repoformer) | GPT-4/DeepSeekCoder 6.7B+ | Doesn't involve distillation |
| Distillation + RAG (DRAG, ACL 2025) | Positive: teaching SLMs | We characterize *failure modes* |

**Our Position (the gap)**: ★ Distilled SLM × RAG Degradation × Code Generation × Failure Characterization

---

## Experiment Direction

### Core Research Question

> **Does Retrieval-Augmented Generation reliably improve the code generation capability of distilled small language models, and if not, under what conditions does it fail?**

### Sub-Questions

1. How does retrieval diversity affect generation quality in parameter-constrained models?
2. Can simple prompting strategies mitigate RAG-induced degradation?
3. Is the observed degradation vendor-agnostic (consistent across model families)?
4. What retrieval failure mechanisms are specific to distilled small models?

### Experimental Design (Three-Group Architecture)

| Group | Purpose | Models | Modes |
|---|---|---|---|
| **Control** | Establish No-RAG baseline | Qwen2.5-3B-Distill, Phi-3-mini, Gemma-2-2B, Qwen2.5-7B | No-RAG |
| **Cross-Validation** | Isolate RAG vs RAG+Forgetting effect | Qwen2.5-3B-Distill, Qwen2.5-3B-Instruct (base) | RAG, RAG+Forgetting |
| **External Validation** | Verify cross-model consistency | Phi-3-mini-4k, Gemma-2-2B, Qwen2.5-7B | RAG |

### Five Ablation Configurations

| Configuration | Temporal Decay | Feedback Scoring | MMR Diversity |
|---|---|---|---|
| Full System | ✓ | ✓ | ✓ |
| No-Temporal | ✗ | ✓ | ✓ |
| No-Feedback | ✓ | ✗ | ✓ |
| No-Diversity | ✓ | ✓ | ✗ |
| Baseline (Vanilla ChromaDB) | ✗ | ✗ | ✗ |

### Five Prompt Strategies

| Strategy | Description |
|---|---|
| A. Current (Reference) | Standard RAG prompt |
| B. Defensive | "Prioritize your own knowledge. Retrieved context is advisory only." |
| C. Minimal | Context appended without instruction |
| D. Conflict-Aware | "If retrieved code conflicts with your knowledge, ignore it." |
| E. No RAG (Control) | Pure distilled model, no retrieval |

---

## Methodology

### System Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  ChromaDB Vector DB  │  ← 303,869 code chunks from 15 open-source repos
│  (ONNX MiniLM-L6-v2) │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Memory Weight Score │  W = α·W_temporal + β·W_feedback + γ·W_redundancy
│  (Ebbinghaus decay)  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  MMR Re-ranking      │  MMR(doc) = λ·sim(query,doc) - (1-λ)·max_sim(doc,selected)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Lifecycle State     │  Active → Warm → Cold → Archived → Suppressed
│  Classification      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  LLM Generation      │  Qwen2.5-3B-Distill (Q4_K_M GGUF via llama.cpp)
│  (with prompt strat) │
└─────────────────────┘
```

### Memory Lifecycle Scoring

```
W_total = α·W_temporal + β·W_feedback + γ·W_redundancy

W_temporal   = exp(-λ · Δt_days)                    [Ebbinghaus decay, λ=0.1]
W_feedback   = (success + prior·k) / (total + k)     [Bayesian smoothed, k=2]
W_redundancy = 1 - max_sim_to_retrieved_set          [Diversity penalty]

Hyperparameters: α=0.5, β=0.3, γ=0.2
```

Lifecycle States:

| State | Weight Range | Attenuation | Retrieval Behavior |
|---|---|---|---|
| Active | W > 0.8 | 1.0 | Full priority |
| Warm | 0.5 < W ≤ 0.8 | 0.8 | Normal |
| Cold | 0.2 < W ≤ 0.5 | 0.5 | Deprioritized |
| Archived | W ≤ 0.2 | 0.2 | Stored, not retrieved |
| Suppressed | W ≤ 0.05 | 0.05 | Soft-deleted |

### MMR Diversity Re-ranking

```
MMR(doc) = λ_mmr · sim(query, doc) - (1-λ_mmr) · max_sim(doc, selected_set)
λ_mmr = 0.7
```

### Model & Quantization

| Component | Specification |
|---|---|
| Base Model | Qwen2.5-3B |
| Distillation | 50K code examples, multi-stage |
| Quantization | GGUF Q4_K_M (4-bit) |
| Inference Engine | llama.cpp |
| Context Window | 2048 tokens |

### Vector Database

- **Total chunks**: 303,869 from 15 open-source GitHub repositories
- **Embeddings**: ONNX all-MiniLM-L6-v2 (384-dimensional)
- **Storage**: ChromaDB persistent client
- **Collections**: summary (repo-level), chunks (file-level), details (function-level)
- **Chunking**: AST-based for Python, regex-based for other languages
- **Deduplication**: Content-hash based

### Benchmarks

- **HumanEval** (OpenAI, 164 problems): Function completion with execution tests
- **MBPP** (Google, 974 problems): Entry-level Python with assert-based tests
- **Evaluation harness**: EvalPlus (base + plus test suites for rigorous Pass@1)

---

## Repository Structure

```
rag-contamination-codegen/
├── README.md                        # This file (English)
├── README_CN.md                     # Chinese version
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
│
├── src/
│   ├── rag/                         # RAG System Core
│   │   ├── search_engine.py         # ChromaDB hierarchical search (3 collections, LRU cache)
│   │   ├── forgetting_retrieval.py  # Ebbinghaus decay + MMR re-ranking + lifecycle states
│   │   ├── build_code_vector_db.py  # Vector DB construction from GitHub repos
│   │   └── expand_vector_db.py      # Expand DB with additional repositories
│   │
│   ├── training/                    # Distillation Training
│   │   ├── train_distill.py         # Multi-stage distillation (50K examples, LoRA)
│   │   ├── prepare_distill_data.py  # Dataset preparation for distillation
│   │   └── build_coding_dataset.py  # Build coding datasets from scratch
│   │
│   └── evaluation/                  # Benchmarking & Evaluation
│       ├── config.py                # Central configuration (models, prompts, paths)
│       ├── run_benchmark.py         # Full benchmark pipeline (multi-model, multi-mode)
│       ├── run_ablation.py          # Ablation study runner (5 configs × 34 problems)
│       ├── e2e_validate.py          # End-to-end validation with EvalPlus
│       └── eval_coding.py           # Coding task evaluation (Pass@1, syntax, execution)
│
├── analysis/                        # Visualization & Analysis
│   ├── generate_figures.py          # Generate paper figures from experiment data
│   └── paper_figures_nature.py      # Nature/CNS submission-grade figures
│
├── paper/                           # Paper Resources
│   ├── paper.pdf                    # Compiled paper PDF
│   ├── paper_draft.txt              # Full text draft
│   ├── paper_narrative.md           # Narrative structure & writing guidelines
│   └── figures/                     # Generated figures (PNG + PDF)
│       ├── fig1_pass_at_1_ablation.png
│       ├── fig2_diversity_vs_pass.png
│       ├── fig3_contamination_stats.png
│       └── fig4_latency_vs_performance.png
│
└── results/                         # Frozen Experiment Data
    ├── statistics.json              # All experiment statistics (frozen 2026-05-25)
    ├── ablation_qwen2.5-3b.csv      # Raw ablation data
    ├── summary_qwen2.5-3b.json      # Summary statistics
    └── experiment_report.md         # Detailed experiment report
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<user>/rag-contamination-codegen.git
cd rag-contamination-codegen

# Install dependencies
pip install -r requirements.txt

# For LLM inference (choose one):
# Option A: llama-cpp-python with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Option B: CPU-only
pip install llama-cpp-python
```

### Prerequisites

- Python 3.10+
- 8GB+ VRAM (for model inference) or 16GB+ RAM (CPU-only)
- ChromaDB vector database (built or downloaded)

---

## Quick Start

### 1. Build the Vector Database (Optional - pre-built DB available)

```python
from src.rag.build_code_vector_db import build_vector_db

# Clone repos and build index
build_vector_db(
    output_dir="./vector_db",
    repos=["pallets/flask", "psf/requests", "sqlalchemy/sqlalchemy"],
    max_chunks_per_repo=5000
)
```

### 2. Run a Single RAG Query

```python
from src.rag.search_engine import SearchEngine

engine = SearchEngine(db_path="./vector_db")
results = engine.search("binary search tree validation", top_k=5)
print(engine.format_results(results))
```

### 3. Test with Forgetting-Aware Retrieval

```python
from src.rag.forgetting_retrieval import ForgettingRetriever

retriever = ForgettingRetriever(
    db_path="./vector_db",
    records_path="./forgetting_records.json",
    forgetting_threshold=0.7,
    forgetting_weight=0.3
)

result = retriever.retrieve("validate BST insertion", top_k=3, use_forgetting=True)
print(f"Context ({result['filtered_count']} filtered):\n{result['context']}")
```

### 4. Run End-to-End Validation

```bash
python src/evaluation/e2e_validate.py
```

---

## Reproduce Experiments

### Experiment 1: Ablation Study

```bash
python src/evaluation/run_ablation.py \
    --model Qwen2.5-3B-Distill \
    --problems 34 \
    --configs full,no-temporal,no-feedback,no-diversity,baseline
```

### Experiment 2: Full Benchmark

```bash
python src/evaluation/run_benchmark.py \
    --models Qwen2.5-3B-Distill,Qwen2.5-3B-Instruct,Phi-3-mini-4k,Gemma-2-2B \
    --modes no-rag,rag,rag-forgetting \
    --datasets humaneval,mbpp
```

### Experiment 3: Prompt Strategy Comparison

```bash
python src/evaluation/run_benchmark.py \
    --models Qwen2.5-3B-Distill \
    --modes rag \
    --prompt-strategies A,B,C,D,E
```

### Generate Figures

```bash
python analysis/generate_figures.py
```

---

## Key Results

### Ablation Study (34 coding problems)

| Configuration | Pass@1 | Avg Latency | Avg Diversity |
|---|---|---|---|
| **No-Diversity** | **0.794** | 1.81s | 0.725 |
| Full System | 0.765 | 1.74s | 0.905 |
| Baseline ChromaDB | 0.765 | 1.58s | 0.533 |
| No-Temporal | 0.735 | 1.50s | 0.905 |
| No-Feedback | 0.706 | 1.50s | 0.905 |

*Reference: Original Qwen2.5-3B (un-finetuned): 0.676 Pass@1*

### Five Key Findings

1. **RAG provides zero net benefit on distilled models**: Distillation alone gives +11.8% gain (0.676 → 0.794). Adding RAG yields 0% additional improvement.

2. **Retrieval diversity is non-monotonic with Pass@1**: Maximum diversity (0.905) does **not** yield best performance. Optimal diversity range: 0.70-0.75.

3. **Defensive prompting fully mitigates contamination**: From 0.735 (standard RAG) → 0.765 (defensive prompt), recovering the No-RAG baseline.

4. **Medium-difficulty tasks are most contamination-sensitive**: Both the "helped" and "harmed" cases occur at medium difficulty — the model's zone of proximal development.

5. **RAG alters 55.9% of answers** compared to No-RAG, with only ~70% text overlap (Jaccard), indicating fundamental generation alteration, not supplementation.

### Contamination Analysis (34 problems)

| Category | Count | Percentage |
|---|---|---|
| Neutral | 32 | 94.1% |
| RAG Helped | 1 | 2.9% |
| RAG Harmed | 1 | 2.9% |
| **Net Effect** | **0** | — |

### Prompt Strategy Comparison

| Strategy | Pass@1 | vs No-RAG |
|---|---|---|
| **B. Defensive** | **0.765** | =0.000 |
| E. No RAG (Control) | 0.765 | (baseline) |
| A. Current | 0.735 | -0.029 |
| C. Minimal | 0.735 | -0.029 |
| D. Conflict-Aware | 0.706 | -0.059 |

---

## Citation

```bibtex
@article{rag-contamination-codegen,
  title={{Retrieval Contamination in Long-Term RAG Systems for Code Generation}},
  author={},
  journal={},
  year={2026},
  note={Preprint}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration: open an issue on this repository.
