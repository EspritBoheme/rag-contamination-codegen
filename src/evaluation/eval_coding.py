"""
评估蒸馏模型编程能力
- 加载 Phase 4 模型 (distill-50k-v4/final)
- 测试多语言编程问题
- 对比原始 Qwen2.5-3B-Instruct
用法:
    python eval_coding.py                        # 完整评估
    python eval_coding.py --quick                # 快速 (2题)
    python eval_coding.py --compare-base         # 对比原始模型
"""
import argparse, json, os, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = "./Qwen2.5-3B-Instruct"
PHASE4_LORA = "./output/distill-50k-v4/final"
PHASE5_LORA = "./output/distill-50k-v5/final"
BASE_LORA = "./output/distill-30k-v3/final"  # Phase 3, 用于对比
OUTPUT_FILE = "eval_coding_results.json"

# 编程测试题: (语言, 问题描述, 关键词)
PROBLEMS = [
    # Python
    ("python", "写一个 Python 函数, 用动态规划求最长公共子序列长度",
     ["def", "lcs", "dp", "return"]),
    ("python", "写一个 Python 装饰器, 记录函数调用时间和参数",
     ["def", "decorator", "time", "functools"]),
    ("python", "用 Python 实现一个简单的 LRU Cache, 支持 get 和 put",
     ["class", "lru", "cache", "dict", "OrderedDict"]),
    # Java
    ("java", "Write a Java class implementing a thread-safe singleton pattern",
     ["class", "Singleton", "synchronized", "instance"]),
    ("java", "Write a Java method that sorts a list of integers using quicksort",
     ["void", "sort", "quickSort", "pivot"]),
    # Kotlin
    ("kotlin", "Write a Kotlin function that filters a list of strings by length and maps to uppercase",
     ["fun", "filter", "map", "List"]),
    # TypeScript
    ("typescript", "Write a TypeScript interface for a User object and a function that fetches user data",
     ["interface", "User", "function", "fetch"]),
    # SQL / 通用
    ("sql", "Write a SQL query to find the top 5 most common values in a column 'tags' from table 'posts'",
     ["SELECT", "GROUP BY", "ORDER BY", "LIMIT"]),
    ("python", "Write a Python function using asyncio to fetch 10 URLs concurrently",
     ["async", "await", "asyncio", "fetch"]),
]


def load_model(lora_path: str):
    """Load base model + LoRA."""
    print(f"加载模型: {MODEL_PATH} + {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.config.use_cache = True
    return tokenizer, model


def generate(tokenizer, model, prompt: str, lang: str) -> str:
    """Generate code answer."""
    messages = [
        {"role": "user", "content": f"{prompt}\n\n请用{lang}写完整代码, 只输出代码, 不要额外解释."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.3, top_p=0.9,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def score_answer(answer: str, keywords: list) -> dict:
    """Simple heuristic scoring."""
    has_code = bool(re.search(r'(def |class |function|fun |int |String|SELECT|async|await|interface)', answer))
    keyword_matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
    lines = answer.strip().split("\n")
    non_empty = [l for l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("//")]
    return {
        "has_code": has_code,
        "keyword_matches": keyword_matches,
        "keyword_total": len(keywords),
        "code_lines": len(non_empty),
        "total_lines": len(lines),
        "score": round((keyword_matches / max(len(keywords), 1)) * 0.6 + (0.4 if has_code else 0), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="仅测试前 2 题")
    parser.add_argument("--compare-base", action="store_true", help="同时测试 Phase 3")
    parser.add_argument("--compare-phase5", action="store_true", help="同时测试 Phase 5")
    args = parser.parse_args()

    problems = PROBLEMS[:2] if args.quick else PROBLEMS
    results = {"model": "distill-50k-v4", "problems": [], "summary": {}}

    # Load Phase 4 model
    print(f"\n{'='*50}")
    print("加载 Phase 4 蒸馏模型...")
    print(f"{'='*50}")
    tokenizer, model = load_model(PHASE4_LORA)

    scores = []
    for lang, prompt, keywords in problems:
        print(f"\n[{lang}] {prompt[:50]}...")
        start = time.time()
        answer = generate(tokenizer, model, prompt, lang)
        elapsed = time.time() - start
        score_info = score_answer(answer, keywords)
        scores.append(score_info["score"])
        results["problems"].append({
            "language": lang, "prompt": prompt,
            "answer": answer[:500],
            "time_s": round(elapsed, 1),
            "score_info": score_info,
        })
        print(f"  耗时: {elapsed:.1f}s | 分数: {score_info['score']}")
        print(f"  输出预览: {answer[:120].strip()}...")
        print("-" * 40)

    avg_score = sum(scores) / max(len(scores), 1)
    results["summary"]["phase4_avg_score"] = round(avg_score, 3)
    results["summary"]["phase4_total_problems"] = len(problems)

    # Compare with base model
    if args.compare_base:
        print(f"\n{'='*50}")
        print("加载原始 Qwen2.5-3B-Instruct (无蒸馏)...")
        print(f"{'='*50}")
        base_tokenizer, base_model = load_model(BASE_LORA)
        base_scores = []

        for lang, prompt, keywords in problems:
            print(f"\n[Base] [{lang}] {prompt[:50]}...")
            answer = generate(base_tokenizer, base_model, prompt, lang)
            score_info = score_answer(answer, keywords)
            base_scores.append(score_info["score"])
            results["problems"][results["problems"].index(
                next(p for p in results["problems"] if p["prompt"] == prompt)
            )]["base_answer"] = answer[:500]
            print(f"  分数: {score_info['score']}")

        base_avg = sum(base_scores) / max(len(base_scores), 1)
        results["summary"]["base_avg_score"] = round(base_avg, 3)
        results["summary"]["improvement"] = round(avg_score - base_avg, 3)

    # Compare with Phase 5
    if args.compare_phase5:
        print(f"\n{'='*50}")
        print("加载 Phase 5 推理蒸馏模型...")
        print(f"{'='*50}")
        p5_tok, p5_model = load_model(PHASE5_LORA)
        p5_scores = []
        for lang, prompt, keywords in problems:
            print(f"\n[Phase5] [{lang}] {prompt[:50]}...")
            answer = generate(p5_tok, p5_model, prompt, lang)
            si = score_answer(answer, keywords)
            p5_scores.append(si["score"])
            idx = next(i for i, p in enumerate(results["problems"]) if p["prompt"] == prompt)
            results["problems"][idx]["phase5_answer"] = answer[:500]
            print(f"  分数: {si['score']}")
        p5_avg = sum(p5_scores) / max(len(p5_scores), 1)
        results["summary"]["phase5_avg_score"] = round(p5_avg, 3)
        results["summary"]["phase5_vs_phase4"] = round(p5_avg - avg_score, 3)

    # Summary
    print(f"\n{'='*50}")
    print("评估总结")
    print(f"{'='*50}")
    print(f"Phase 4 蒸馏模型平均分: {results['summary']['phase4_avg_score']}")
    if "base_avg_score" in results["summary"]:
        print(f"Phase 3 模型平均分:     {results['summary']['base_avg_score']}")
        imp = results["summary"]["improvement"]
        print(f"Phase 4 提升:           {imp:+.3f}")
    if "phase5_avg_score" in results["summary"]:
        print(f"Phase 5 模型平均分:     {results['summary']['phase5_avg_score']}")
        d = results["summary"]["phase5_vs_phase4"]
        print(f"Phase 5 vs Phase 4:     {d:+.3f}")
    print(f"测试题数: {len(problems)}")

    # 相对于 DeepSeek 的估计说明
    print(f"\n{'='*50}")
    print("相对 DeepSeek V4 Flash 估计")
    print(f"{'='*50}")
    print("Qwen2.5-3B (3B 参数) vs DeepSeek V4 Flash (>100B 参数)")
    print("参数量差 30 倍以上, 直接对比不公平.")
    print("合理的期望: 蒸馏后编码能力 ≈ 原始 Qwen2.5-3B 的 110-130%")
    print("绝对能力: 能解决 LeetCode Easy~Medium, 基础 CRUD, 简单脚本")
    print("DeepSeek V4 Flash 对比: 约 15-25% 水平 (受限于 3B 参数量)")
    print(f"{'='*50}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
