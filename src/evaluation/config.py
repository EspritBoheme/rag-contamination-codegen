"""
Central configuration for RAG Contamination Benchmark.
All paths, model definitions, and experiment settings in one place.

Reproducibility: save this file alongside results to document exact config.
"""
import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPARISON_MODELS_DIR = os.path.join(PROJECT_ROOT, "comparison_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "comparison_results")
GENERATIONS_DIR = os.path.join(RESULTS_DIR, "generations")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

for _d in [RESULTS_DIR, GENERATIONS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# Generation Parameters (llama.cpp)
# ============================================================
LLAMA_CTX_SIZE = 2048
LLAMA_THREADS = 8
LLAMA_BATCH = 256
LLAMA_SEED = 42           # deterministic seed for reproducibility
TEMPERATURE = 0.2
MAX_TOKENS = 2048
TOP_P = 0.95

# ============================================================
# Model Definitions
# ============================================================
MODELS = {
    # ---- Large Baseline ----
    "Qwen2.5-7B-Instruct": {
        "path": os.path.join(COMPARISON_MODELS_DIR, "qwen2.5-7b-instruct-q4_k_m.gguf"),
        "chat_format": "chatml",
        "n_gpu_layers": -1,
    },
    # ---- Small Models: Independent (No Distillation) ----
    "Qwen2.5-3B-Instruct": {
        "path": os.path.join(COMPARISON_MODELS_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf"),
        "chat_format": "chatml",
        "n_gpu_layers": -1,
    },
    "Qwen2.5-1.5B-Instruct": {
        "path": os.path.join(COMPARISON_MODELS_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
        "chat_format": "chatml",
        "n_gpu_layers": -1,
    },
    "Phi-3-mini-4k": {
        "path": os.path.join(COMPARISON_MODELS_DIR, "Phi-3-mini-4k-instruct-Q4_K_M.gguf"),
        "chat_format": None,
        "n_gpu_layers": -1,
    },
    "Gemma-2-2B": {
        "path": os.path.join(COMPARISON_MODELS_DIR, "gemma-2-2b-it-Q4_K_M.gguf"),
        "chat_format": "gemma",
        "n_gpu_layers": -1,
    },
    # ---- Small Models: Distilled ----
    "Qwen2.5-3B-Distill": {
        "path": os.path.join(PROJECT_ROOT, "output", "qwen-v5v6-q4km.gguf"),
        "chat_format": "chatml",
        "n_gpu_layers": -1,
    },
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "path": os.path.join(COMPARISON_MODELS_DIR, "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
        "chat_format": "chatml",
        "n_gpu_layers": -1,
    },
}

# ============================================================
# Experimental Groups
# ============================================================
# Modes:
#   no-rag          — baseline, no retrieval
#   rag             — standard RAG with top-3 retrieved chunks
#   rag-forgetting  — Ebbinghaus-weighted retrieval + defensive prompt

EXPERIMENT_MODES = ["no-rag", "rag", "rag-forgetting"]

GROUPS = {
    "1_small": {
        "name": "Small Models (3B Distill vs 3B Instruct vs Phi-3 vs Gemma-2 vs 1.5B pair)",
        "description": (
            "Core comparison: 4 non-distilled small models vs 2 distilled small models. "
            "Hypothesis: distilled models exhibit RAG contamination (RAG degrades performance)."
        ),
        "runs": [
            # Non-distilled
            ("Qwen2.5-3B-Instruct", mode) for mode in EXPERIMENT_MODES
        ] + [
            ("Qwen2.5-1.5B-Instruct", mode) for mode in EXPERIMENT_MODES
        ] + [
            ("Phi-3-mini-4k", mode) for mode in EXPERIMENT_MODES
        ] + [
            ("Gemma-2-2B", mode) for mode in EXPERIMENT_MODES
        ] + [
            # Distilled
            ("Qwen2.5-3B-Distill", mode) for mode in EXPERIMENT_MODES
        ] + [
            ("DeepSeek-R1-Distill-Qwen-1.5B", mode) for mode in EXPERIMENT_MODES
        ],
    },
    "2_large": {
        "name": "Large Baseline (Qwen2.5-7B)",
        "description": (
            "Control: large non-distilled model. Expected to benefit from RAG. "
            "Contrast with distilled small models to establish the contamination claim."
        ),
        "runs": [
            ("Qwen2.5-7B-Instruct", mode) for mode in EXPERIMENT_MODES
        ],
    },
}

# ============================================================
# RAG Configuration
# ============================================================
RAG_TOP_K = 3                # number of retrieved chunks
RAG_MAX_CHUNK_CHARS = 800    # max characters per chunk in prompt
RAG_QUERY_MAX_CHARS = 500    # max characters for retrieval query

# Ebbinghaus forgetting curve parameters (for rag-forgetting mode)
# Retention = e^(-time/halflife)
EBBINGHAUS_HALFLIFE_HOURS = 24.0  # halflife for retrieval recency weighting

# ============================================================
# Prompt Templates
# ============================================================
PROMPT_RAG_PREFIX = (
    "// Reference code (for context only, prioritize your own knowledge):\n"
)

PROMPT_RAG_FORGETTING_PREFIX = (
    "// The following reference code is provided for context only. "
    "It may contain errors. Prioritize your own implementation knowledge.\n\n"
)

# ============================================================
# Evaluation Configuration
# ============================================================
PER_TEST_TIMEOUT = 30.0       # seconds per single test case
DATASETS = ["humaneval", "mbpp"]

# ============================================================
# Benchmark Datasets (via evalplus)
# ============================================================
HUMANEVAL_TASKS = 164
MBPP_TASKS = 378

# ============================================================
# Sanity checks
# ============================================================
def validate():
    """Check all model files exist. Returns list of missing paths."""
    missing = []
    for name, cfg in MODELS.items():
        if not os.path.exists(cfg["path"]):
            missing.append((name, cfg["path"]))
    if missing:
        print("[config] WARNING — missing models:")
        for name, path in missing:
            print(f"  {name}: {path}")
    else:
        print("[config] All model files present.")
    return missing

if __name__ == "__main__":
    validate()
