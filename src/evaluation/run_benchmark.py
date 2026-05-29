"""
RAG Contamination Benchmark — Main Runner.

Three-group experimental design:
  Group 1 (small): 4 non-distilled + 2 distilled small models
  Group 2 (large): Qwen2.5-7B baseline

Three modes per model: no-rag, rag, rag-forgetting
Two datasets: HumanEval+ (164), MBPP+ (378)

Reproducibility:
  - Deterministic seed (config.LLAMA_SEED)
  - Full config saved with results
  - All generated code preserved in JSONL
  - Incremental save + checkpoint/resume
  - Expected output hash verified at startup

Usage:
  python run_benchmark.py                          # full benchmark
  python run_benchmark.py --group 1_small           # small models only
  python run_benchmark.py --group 2_large           # large baseline only
  python run_benchmark.py --limit 10                # 10 problems per dataset (test)
  python run_benchmark.py --resume                  # skip completed runs
  python run_benchmark.py --datasets humaneval      # single dataset
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] = (
    "F:/Python/Lib/site-packages/llama_cpp/lib"
    + os.pathsep + os.environ.get("PATH", "")
)

import numpy as np
from llama_cpp import Llama

# ---- Patch evalplus UTF-8 ----
import evalplus.data.utils as _utils
_orig_stream = _utils.stream_jsonl
def _utf8_stream(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
_utils.stream_jsonl = _utf8_stream

from config import (
    MODELS, GROUPS, EXPERIMENT_MODES, DATASETS,
    LLAMA_CTX_SIZE, LLAMA_THREADS, LLAMA_BATCH, LLAMA_SEED,
    TEMPERATURE, MAX_TOKENS, TOP_P,
    RAG_TOP_K, RAG_MAX_CHUNK_CHARS, RAG_QUERY_MAX_CHARS,
    PROMPT_RAG_PREFIX, PROMPT_RAG_FORGETTING_PREFIX,
    RESULTS_DIR, GENERATIONS_DIR, VECTOR_DB_DIR,
    HUMANEVAL_TASKS, MBPP_TASKS, PER_TEST_TIMEOUT,
)
from standalone_eval import evaluate_standalone as evaluate

# ============================================================
# Logging
# ============================================================
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# RAG
# ============================================================
_rag_collection = None
_rag_ef = None


def get_rag():
    global _rag_collection, _rag_ef
    if _rag_collection is not None:
        return _rag_collection, _rag_ef

    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    import sqlite3

    _rag_ef = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    for coll_name in ["chunks", "details"]:
        try:
            _rag_collection = client.get_collection(coll_name, embedding_function=_rag_ef)
            _rag_collection.query(query_texts=["test"], n_results=1)
            log(f"RAG: {coll_name} — {_rag_collection.count()} entries")
            return _rag_collection, _rag_ef
        except Exception:
            pass

    # Fallback: sqlite direct read
    db_path = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")
    if not os.path.exists(db_path):
        log("RAG: no vector DB found, RAG disabled")
        _rag_collection = None
        return None, _rag_ef

    conn = sqlite3.connect(db_path)
    docs = []
    for coll in ["chunks", "details"]:
        try:
            rows = conn.execute(
                "SELECT string_value FROM embedding_metadata "
                "WHERE key='chroma:document' "
                "AND id IN (SELECT id FROM embeddings WHERE collection_id IN "
                "(SELECT id FROM collections WHERE name=?))", (coll,)
            ).fetchall()
            for (doc,) in rows:
                if doc:
                    docs.append(doc)
        except Exception:
            pass
    conn.close()

    class SqliteColl:
        def __init__(self, docs, ef):
            self._docs = docs
            self._ef = ef
        def count(self):
            return len(self._docs)
        def query(self, query_texts, n_results=3):
            qv = np.array(self._ef(query_texts))
            results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            for i, doc in enumerate(self._docs[:500]):
                try:
                    dv = np.array(self._ef([doc[:1000]]))
                    sim = float(np.dot(qv, dv.T).flatten()[0])
                    results["documents"][0].append(doc)
                    results["metadatas"][0].append({"idx": i})
                    results["distances"][0].append(1.0 - sim)
                except Exception:
                    continue
            if results["documents"][0]:
                idx = sorted(range(len(results["distances"][0])),
                           key=lambda i: results["distances"][0][i])[:n_results]
                for k in results:
                    results[k][0] = [results[k][0][i] for i in idx]
            return results

    _rag_collection = SqliteColl(docs, _rag_ef)
    log(f"RAG: sqlite fallback — {_rag_collection.count()} docs")
    return _rag_collection, _rag_ef


def retrieve_context(query: str, top_k: int = RAG_TOP_K) -> str:
    try:
        coll, ef = get_rag()
        if coll is None:
            return ""
        results = coll.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        lines = [f"// Retrieved {i+1}:\n{d[:RAG_MAX_CHUNK_CHARS]}"
                 for i, d in enumerate(docs[:top_k])]
        return "\n\n".join(lines)
    except Exception:
        return ""


# ============================================================
# Prompt building
# ============================================================
def build_prompt(item: dict, mode: str) -> str:
    base_prompt = item["prompt"]
    if mode == "no-rag":
        return base_prompt

    query = (item.get("prompt", "") + " " + item.get("entry_point", ""))[:RAG_QUERY_MAX_CHARS]
    ctx = retrieve_context(query, top_k=RAG_TOP_K)

    if not ctx:
        return base_prompt

    if mode == "rag-forgetting":
        return (f"{PROMPT_RAG_FORGETTING_PREFIX}"
                f"{ctx}\n\n"
                f"// Complete the function below:\n{base_prompt}")
    else:
        return (f"{PROMPT_RAG_PREFIX}"
                f"{ctx}\n\n"
                f"// Complete the function below:\n{base_prompt}")


# ============================================================
# Code extraction
# ============================================================
def clean_trailing_garbage(code: str) -> str:
    """Remove trailing print calls, test examples, and broken lines."""
    lines = code.split("\n")
    last_good = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith(("return ", "def ", "class ", "pass", "raise ",
                                "yield ")):
            last_good = i
            break
        if stripped in ("", "}", "]", ")"):
            continue
        if stripped.startswith(("#", "//", "print(", ">>>", "...",
                                "if __name__")):
            last_good = i - 1
            continue
        if stripped and not stripped.startswith(" "):
            last_good = i
            break
    return "\n".join(lines[:last_good + 1]).strip()


def extract_code(completion: str, prompt: str) -> str:
    """Extract function body from model completion."""
    text = completion.strip()

    # Strip markdown code fences
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Strip prompt regurgitation (keep function defs)
    prompt_lines = prompt.strip().split("\n")
    text_lines = text.split("\n")
    matched = 0
    for pl, tl in zip(prompt_lines, text_lines):
        if pl.strip() == tl.strip():
            matched += 1
        else:
            break

    if matched >= 3:
        kept = []
        for line in text_lines:
            s = line.strip()
            if s.startswith(("def ", "class ", "import ", "from ", "@",
                           "if __name__")):
                kept.append(line)
            elif not any(s == pl.strip() for pl in prompt_lines):
                kept.append(line)
        if kept:
            text = "\n".join(kept).strip()

    # Recover missing function signature
    if not re.search(r'^\s*(def |class |async def )', text, re.MULTILINE):
        m_def = re.search(
            r'((?:async\s+)?def\s+\w+\s*\([^)]*\)(?:\s*->\s*\w+(?:\[\w+(?:,\s*\w+)*\])?)?\s*:)',
            completion)
        if m_def:
            text = m_def.group(1) + "\n" + text

    return clean_trailing_garbage(text)


# ============================================================
# Model management
# ============================================================
def load_model(model_name: str) -> Optional[Llama]:
    cfg = MODELS[model_name]
    path = cfg["path"]
    if not os.path.exists(path):
        log(f"ERROR: model not found: {path}")
        return None
    log(f"Loading {model_name} ({os.path.basename(path)})...")
    t0 = time.time()
    llm = Llama(
        model_path=path,
        n_gpu_layers=cfg.get("n_gpu_layers", -1),
        n_ctx=LLAMA_CTX_SIZE,
        n_threads=LLAMA_THREADS,
        n_batch=LLAMA_BATCH,
        use_mmap=False,
        verbose=False,
        seed=LLAMA_SEED,
        chat_format=cfg.get("chat_format", None),
    )
    log(f"  Loaded in {time.time() - t0:.1f}s (seed={LLAMA_SEED})")
    return llm


def unload_model(llm):
    """Free GPU memory by deleting the Llama instance."""
    if llm is not None:
        del llm
        gc.collect()


# ============================================================
# Generation + Evaluation (single run)
# ============================================================
def generate_solutions(
    llm: Llama,
    problems: list,
    mode: str,
    model_name: str,
    dataset: str,
    limit: Optional[int],
) -> Tuple[List[dict], float, int, int]:
    """Generate code for each problem. Returns (samples, total_time, total_tokens, errors)."""
    n = len(problems[:limit]) if limit else len(problems)
    problems_subset = problems[:n] if limit else problems
    all_problems = problems

    mode_label = {"no-rag": "NoRAG", "rag": "RAG", "rag-forgetting": "RAGFF"}[mode]

    samples = []
    total_tokens = 0
    total_time = 0
    gen_errors = 0

    for i, item in enumerate(problems_subset):
        task_id = item["task_id"]
        prompt_text = build_prompt(item, mode)

        t0 = time.time()
        try:
            output = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
            )
            completion = output["choices"][0]["message"]["content"]
            tokens = output["usage"]["completion_tokens"]
        except Exception as e:
            log(f"  [{i+1}/{n}] GEN ERROR: {e}")
            traceback.print_exc()
            gen_errors += 1
            samples.append({
                "task_id": task_id,
                "solution": item.get("prompt", "") + "    pass\n",
            })
            continue

        elapsed = time.time() - t0
        total_tokens += tokens
        total_time += elapsed

        code = extract_code(completion, prompt_text)
        samples.append({"task_id": task_id, "solution": code})

        if (i + 1) % 30 == 0:
            tps = total_tokens / total_time if total_time > 0 else 0
            eta = (total_time / (i + 1)) * (n - i - 1)
            log(f"  [{i+1}/{n}] {mode_label} tps={tps:.0f} eta={eta/60:.1f}m")

    # Fill remaining problems with pass placeholder (evalplus requirement)
    if limit and len(samples) < len(all_problems):
        generated_ids = {s["task_id"] for s in samples}
        for item in all_problems:
            if item["task_id"] not in generated_ids:
                samples.append({
                    "task_id": item["task_id"],
                    "solution": item.get("prompt", "") + "    pass\n",
                })

    log(f"  {mode_label} gen done: {total_time:.0f}s, {total_tokens}tok, "
        f"{gen_errors} errors")
    return samples, total_time, total_tokens, gen_errors


def run_single(
    llm: Llama,
    problems: list,
    dataset: str,
    mode: str,
    model_name: str,
    limit: Optional[int],
) -> dict:
    """Generate and evaluate one (model, mode, dataset) combination."""
    mode_label = {"no-rag": "NoRAG", "rag": "RAG", "rag-forgetting": "RAGFF"}[mode]
    full_count = len(problems)

    # Generate
    samples, gen_time, total_tokens, gen_errors = generate_solutions(
        llm, problems, mode, model_name, dataset, limit)

    # Save JSONL
    safe_name = model_name.replace(" ", "_").replace(".", "-")
    jsonl_path = os.path.join(GENERATIONS_DIR,
                              f"{safe_name}_{dataset}_{mode_label}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Evaluate
    n_generated = min(limit, full_count) if limit else full_count
    eval_results = evaluate(
        dataset=dataset, samples=jsonl_path,
        parallel=1, i_just_wanna_run=True, base_only=False,
    )

    # Parse
    passed_base = 0
    passed_plus = 0
    total_tasks = 0
    for task_id, task_results in eval_results.get("eval", {}).items():
        total_tasks += 1
        first = task_results[0] if task_results else None
        if first:
            base_ok = first.get("base_status", "") == "pass"
            plus_ok = first.get("plus_status", "") == "pass"
            if base_ok:
                passed_base += 1
            if base_ok and plus_ok:
                passed_plus += 1

    pass_at_1 = round(passed_base / total_tasks * 100, 1) if total_tasks else 0
    pass_at_1_plus = round(passed_plus / total_tasks * 100, 1) if total_tasks else 0

    log(f"  RESULT: Pass@1={pass_at_1}%  Pass@1+={pass_at_1_plus}%")

    return {
        "model": model_name,
        "mode": mode,
        "dataset": dataset,
        "n_generated": n_generated,
        "n_total": total_tasks,
        "passed_base": passed_base,
        "passed_plus": passed_plus,
        "pass_at_1_pct": pass_at_1,
        "pass_at_1_plus_pct": pass_at_1_plus,
        "gen_time_s": round(gen_time, 1),
        "gen_tokens": total_tokens,
        "avg_tok_per_s": round(total_tokens / gen_time, 1) if gen_time > 0 else 0,
        "gen_errors": gen_errors,
        "jsonl_path": jsonl_path,
        "eval_result_path": jsonl_path.replace(".jsonl", "_eval_results.json"),
    }


# ============================================================
# Result persistence
# ============================================================
_RESULTS_META = {
    "config": {
        "seed": LLAMA_SEED,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "ctx_size": LLAMA_CTX_SIZE,
    },
    "models": {k: {
        "path": os.path.basename(v["path"]),
        "chat_format": v["chat_format"],
    } for k, v in MODELS.items()},
    "started_at": None,
    "runs": {},
}


def make_run_key(model_name, mode, dataset):
    return f"{model_name}|{mode}|{dataset}"


def save_state(results: dict, meta: dict):
    """Incremental save of all results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {**meta, "results": results, "saved_at": ts}
    out_file = os.path.join(RESULTS_DIR, f"benchmark_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out_file


def load_previous_state() -> Tuple[dict, set]:
    """Find most recent result file and return (results_dict, completed_keys)."""
    pattern = os.path.join(RESULTS_DIR, "benchmark_*.json")
    files = sorted(__import__("glob").glob(pattern))
    if not files:
        return {}, set()
    latest = files[-1]
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    completed = set(results.keys())
    log(f"Resume: {len(completed)} completed runs from {os.path.basename(latest)}")
    return results, completed


# ============================================================
# Reporting
# ============================================================
def print_table(results: dict, group_keys: list):
    """Print paper-ready results table."""
    for ds_name in ["humaneval", "mbpp"]:
        print(f"\n{'─'*70}")
        print(f"  {ds_name.upper()}")
        print(f"  {'Model':<32} {'Mode':<14} {'Pass@1':>7} {'Pass@1+':>7}")
        print(f"  {'─'*32} {'─'*14} {'─'*7} {'─'*7}")
        for gkey in group_keys:
            for mname, mode in GROUPS[gkey]["runs"]:
                key = make_run_key(mname, mode, ds_name)
                if key in results:
                    r = results[key]
                    mode_label = {"no-rag": "No-RAG", "rag": "RAG",
                                  "rag-forgetting": "RAG+FF"}[mode]
                    print(f"  {mname:<32} {mode_label:<14} "
                          f"{r['pass_at_1_pct']:>6.1f}% {r['pass_at_1_plus_pct']:>6.1f}%")


def print_contamination_analysis(results: dict):
    """Per-model contamination gap and forgetting recovery."""
    for model_name in MODELS:
        print(f"\n{'─'*70}")
        print(f"  {model_name}")
        print(f"  {'Dataset':<12} {'Metric':<12} {'NoRAG':>7} {'RAG':>7} "
              f"{'RAG+FF':>7} {'Gap':>8} {'Recovery':>8}")
        print(f"  {'─'*12} {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")
        for ds_name in ["humaneval", "mbpp"]:
            for metric, field in [("Pass@1", "pass_at_1_pct"),
                                   ("Pass@1+", "pass_at_1_plus_pct")]:
                no_rag = results.get(make_run_key(model_name, "no-rag", ds_name), {})
                rag = results.get(make_run_key(model_name, "rag", ds_name), {})
                ff = results.get(make_run_key(model_name, "rag-forgetting", ds_name), {})
                nr = no_rag.get(field, None)
                r = rag.get(field, None)
                f = ff.get(field, None)
                if nr is not None and r is not None:
                    gap = nr - r
                    rec = f - r if f is not None else None
                    nr_s = f"{nr:.1f}%" if isinstance(nr, (int, float)) else "N/A"
                    r_s = f"{r:.1f}%" if isinstance(r, (int, float)) else "N/A"
                    f_s = f"{f:.1f}%" if isinstance(f, (int, float)) else "N/A"
                    gap_s = f"{gap:+.1f}%" if isinstance(gap, (int, float)) else "N/A"
                    rec_s = f"{rec:+.1f}%" if isinstance(rec, (int, float)) else "N/A"
                    print(f"  {ds_name:<12} {metric:<12} {nr_s:>7} {r_s:>7} "
                          f"{f_s:>7} {gap_s:>8} {rec_s:>8}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAG Contamination Benchmark")
    parser.add_argument("--group", nargs="*", default=None,
                       help="Groups to run (1_small, 2_large)")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit problems per dataset (0 = full)")
    parser.add_argument("--datasets", nargs="*", default=None,
                       choices=["humaneval", "mbpp"],
                       help="Datasets to evaluate (default: both)")
    parser.add_argument("--resume", action="store_true",
                       help="Skip already-completed runs")
    parser.add_argument("--models", nargs="*", default=None,
                       help="Specific models to run (default: all in group)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    datasets = args.datasets if args.datasets else DATASETS
    group_keys = args.group if args.group else list(GROUPS.keys())

    # Normalize group keys
    normalized = []
    for g in group_keys:
        for k in GROUPS:
            if k.startswith(g) or g == k:
                normalized.append(k)
    group_keys = sorted(set(normalized)) if normalized else list(GROUPS.keys())

    # ---- Load problems ----
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    he_data = get_human_eval_plus()
    mbpp_data = get_mbpp_plus()
    he_problems = sorted(he_data.values(), key=lambda x: x["task_id"])
    mbpp_problems = sorted(mbpp_data.values(), key=lambda x: x["task_id"])

    print("=" * 70)
    print("RAG CONTAMINATION BENCHMARK")
    print("=" * 70)
    print(f"HumanEval+: {len(he_problems)} problems")
    print(f"MBPP+:      {len(mbpp_problems)} problems")
    print(f"Groups:     {len(group_keys)}")
    print(f"Models:     {len(MODELS)}")
    print(f"Seed:       {LLAMA_SEED}")
    print(f"Temperature:{TEMPERATURE}")
    if limit:
        print(f"LIMIT:      {limit} problems (test mode)")
    print()

    # ---- Resume ----
    all_results, completed_keys = {}, set()
    if args.resume:
        all_results, completed_keys = load_previous_state()

    _RESULTS_META["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- Run experiments ----
    loaded_models = {}
    current_model = None

    for gkey in group_keys:
        ginfo = GROUPS[gkey]
        print(f"\n{'='*70}")
        print(f"GROUP: {ginfo['name']}")
        print(f"{'='*70}")

        for mname, mode in ginfo["runs"]:
            # Filter by --models if specified
            if args.models and mname not in args.models:
                continue

            model_cfg = MODELS[mname]

            # Model switching: unload old model
            if current_model is not None and current_model != mname:
                log(f"Unloading {current_model} (GPU memory)")
                unload_model(loaded_models.pop(current_model, None))
                current_model = None

            # Load model if needed
            if mname not in loaded_models:
                llm = load_model(mname)
                if llm is None:
                    continue
                loaded_models[mname] = llm
                current_model = mname
            else:
                llm = loaded_models[mname]

            for ds_name in datasets:
                # Skip completed runs in resume mode
                run_key = make_run_key(mname, mode, ds_name)
                if args.resume and run_key in completed_keys:
                    log(f"SKIP {run_key} (already completed)")
                    continue

                problems = he_problems if ds_name == "humaneval" else mbpp_problems
                log(f"RUN: {mname} | {mode} | {ds_name} "
                    f"({'full' if limit is None else f'{limit}/{len(problems)}'})")

                summary = run_single(llm, problems, ds_name, mode, mname, limit)
                all_results[run_key] = summary

                # Incremental save
                save_state(all_results, _RESULTS_META)

    # ---- Final save & report ----
    final_path = save_state(all_results, _RESULTS_META)
    log(f"Results saved: {final_path}")

    print_table(all_results, group_keys)
    print_contamination_analysis(all_results)
    log("Done.")


if __name__ == "__main__":
    main()
