"""Quick end-to-end validation: 3B NoRAG HumanEval+ with 5 problems."""
import json, os, re, sys, time

os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] = (
    "F:/Python/Lib/site-packages/llama_cpp/lib"
    + os.pathsep + os.environ.get("PATH", ""))

from llama_cpp import Llama

GEN_DIR = "D:/模型微调加蒸馏/comparison_results/generations"
os.makedirs(GEN_DIR, exist_ok=True)

print("=== E2E Validation: 3B NoRAG HumanEval+ (5 problems) ===")
print("Loading model...", flush=True)
llm = Llama(
    model_path="D:/模型微调加蒸馏/comparison_models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_gpu_layers=16, n_ctx=2048, n_threads=8, n_batch=256,
    use_mmap=False, verbose=False, chat_format="chatml",
)
print("Loaded.", flush=True)

from evalplus.data import get_human_eval_plus, get_mbpp_plus
problems = sorted(get_human_eval_plus().values(), key=lambda x: x["task_id"])[:5]

samples = []
for item in problems:
    prompt = item["prompt"]
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024, temperature=0.2,
    )
    completion = output["choices"][0]["message"]["content"]

    # extract_code (same as run_official_bench.py)
    text = completion.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Strip prompt prefix
    import re as _re
    def _norm(s): return _re.sub(r'\s+', ' ', s).strip()
    pn = _norm(prompt)
    tn = _norm(text)
    common = 0
    for a, b in zip(pn, tn):
        if a == b: common += 1
        else: break
    if common >= 60:
        seen = 0; split_at = 0; in_space = False
        for j, ch in enumerate(text):
            cur_space = ch.isspace()
            if cur_space and in_space:
                split_at = j + 1
                continue
            if seen >= common: break
            seen += 1; in_space = cur_space; split_at = j + 1
        text = text[split_at:].strip()

    samples.append({"task_id": item["task_id"], "solution": text})
    print(f"  {item['task_id']}: {len(text)} chars", flush=True)

# Fill remaining with placeholders (evalplus requires all 164)
all_problems = sorted(get_human_eval_plus().values(), key=lambda x: x["task_id"])
gen_ids = {s["task_id"] for s in samples}
for item in all_problems:
    if item["task_id"] not in gen_ids:
        samples.append({"task_id": item["task_id"], "solution": item.get("prompt", "") + "    pass\n"})

jsonl_path = os.path.join(GEN_DIR, "E2E_TEST_humaneval_NoRAG.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"JSONL: {jsonl_path} ({len(samples)} samples)", flush=True)

# Run evalplus evaluation
print("Running evalplus evaluate()...", flush=True)
from evalplus.evaluate import evaluate
evaluate(dataset="humaneval", samples=jsonl_path, parallel=1, i_just_wanna_run=True, base_only=False)

result_path = jsonl_path.replace(".jsonl", "_eval_results.json")
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    passed_base = 0
    passed_plus = 0
    total = 0
    for task_id, results in eval_data.get("eval", {}).items():
        total += 1
        first = results[0] if results else None
        if first:
            if first.get("base_status") == "pass":
                passed_base += 1
            if first.get("plus_status") == "pass":
                passed_plus += 1

    print(f"\n=== RESULTS ===")
    print(f"base Pass@1: {passed_base}/{total} ({passed_base/total*100:.1f}%)")
    print(f"plus Pass@1: {passed_plus}/{total} ({passed_plus/total*100:.1f}%)")
    print(f"Valid eval entries: {total}")

    # Show which 5 samples actually passed
    for tid, results in eval_data.get("eval", {}).items():
        first = results[0] if results else None
        if first and first.get("base_status") == "pass":
            print(f"  PASS base: {tid}")
else:
    print("ERROR: No eval results found!")
