"""
消融实验 & 跨模型泛化评测脚本
用于 NLP 期刊 SCI 论文实验

用法:
  # 消融实验 (Qwen2.5-3B 四组配置)
  python run_ablation.py --ablation all
  python run_ablation.py --ablation full,no-rag,no-forgetting,baseline

  # 跨模型泛化实验
  python run_ablation.py --model Phi-3-mini-4k-instruct --ablation full,baseline
  python run_ablation.py --model TinyLlama-1.1B --ablation full,baseline

  # 指定 benchmark
  python run_ablation.py --ablation all --benchmark_path ./bench_coding_v2.py

  # 只跑快速测试 (5题)
  python run_ablation.py --ablation all --quick
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch

# ============================================================
# 配置
# ============================================================
MODEL_PATHS = {
    "qwen2.5-3b": "./output/p2-q4_k_m.gguf",       # 蒸馏模型 (P2)
    "qwen2.5-3b-base": "./output/qwen-v5-q4km.gguf",  # 原始基座 GGUF
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

RESULTS_DIR = "./ablation_results"
CSV_FILE = os.path.join(RESULTS_DIR, "ablation_results.csv")

# ============================================================
# Benchmark 问题 (占位符 — 替换为你的完整 benchmark)
# 格式: (id, difficulty, lang, prompt, keywords, desc, test_cases, ban_imports)
# ============================================================
PROBLEMS = [
    # === 原有 15 题 (bench_coding_v2.py) ===
    (1, "easy", "python",
     "写一个 Python 函数实现二分查找",
     ["def", "binary", "mid", "return", "low", "high"],
     "binary search",
     [("arr=[1,3,5,7,9]; print(binary_search(arr,5))", "2"),
      ("arr=[1,3,5,7,9]; print(binary_search(arr,4))", "-1")],
     []),
    (2, "easy", "python",
     "写一个 Python 装饰器, 记录函数执行时间",
     ["def", "decorator", "time", "functools", "wraps"],
     "decorator pattern", None, []),
    (3, "medium", "python",
     "用 Python 实现 LRU Cache, 支持 get/put 操作, O(1) 复杂度",
     ["class", "lru", "cache", "OrderedDict", "dict"],
     "LRU cache",
     [("c=LRUCache(2); c.put(1,1); c.put(2,2); print(c.get(1))", "1"),
      ("c=LRUCache(2); c.put(1,1); c.put(2,2); c.put(3,3); print(c.get(1))", "-1")],
     []),
    (4, "medium", "python",
     "用 Python 实现一个简单的 Markdown 解析器, 支持标题、粗体、链接",
     ["def", "parse", "re", "markdown", "#", "**"],
     "markdown parser", None, []),
    (5, "medium", "python",
     "用 asyncio 实现并发 URL 批量下载器, 限制最大并发数为 10",
     ["async", "await", "asyncio", "semaphore", "fetch"],
     "async downloader", None, []),
    (6, "hard", "python",
     "实现一个 Python 版的 mini Redis, 支持 GET/SET/DEL/EXPIRE 命令, 使用 asyncio 处理连接",
     ["class", "async", "dict", "expire", "command"],
     "mini redis", None,
     ["aioredis", "redis", "redislite"]),
    (7, "easy", "java",
     "Write a Java method to reverse a linked list iteratively",
     ["Node", "next", "prev", "while", "return"],
     "linked list reverse", None, []),
    (8, "medium", "java",
     "Implement a thread-safe producer-consumer pattern in Java using BlockingQueue",
     ["BlockingQueue", "put", "take", "Thread", "synchronized"],
     "producer consumer", None, []),
    (9, "easy", "kotlin",
     "Write a Kotlin function using higher-order functions to filter, transform, and aggregate a list",
     ["fun", "filter", "map", "reduce", "List"],
     "collection operations", None, []),
    (10, "medium", "typescript",
     "Implement a TypeScript generic EventEmitter class with on/emit/off methods and type-safe events",
     ["class", "EventEmitter", "on", "emit", "Map", "generic"],
     "event emitter", None,
     ["rxjs", "eventemitter3", "mitt"]),
    (11, "medium", "sql",
     "Write a SQL query to find the top 5 departments with highest average salary, excluding departments with fewer than 5 employees",
     ["SELECT", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "AVG"],
     "complex SQL query", None, []),
    (12, "medium", "go",
     "Write a Go function that implements a concurrent web scraper using goroutines and channels",
     ["func", "go", "chan", "goroutine", "WaitGroup"],
     "concurrent scraper", None, []),
    (13, "hard", "rust",
     "Implement a thread-safe HashMap in Rust with read-write lock, supporting get/insert/remove",
     ["fn", "HashMap", "Mutex", "RwLock", "Arc"],
     "concurrent hashmap", None, []),
    (14, "hard", "python",
     "实现 A* 寻路算法, 支持网格地图, 返回最短路径",
     ["def", "astar", "heap", "heuristic", "path", "neighbor"],
     "A* pathfinding",
     [("grid=[[0,0,0],[1,1,0],[0,0,0]]; print(astar(grid,(0,0),(2,2)))", "4")],
     []),
    (15, "expert", "python",
     "用 Python 实现一个简易版正则表达式引擎, 支持 . * + ? 和字符类",
     ["def", "match", "parse", "regex", "state", "NFA", "DFA"],
     "regex engine",
     [("print(match('a*b', 'aaab'))", "True"),
      ("print(match('a.b', 'axb'))", "True")],
     ["re"]),
    # === 新增 19 题 (expand_benchmark.py) ===
    (16, "easy", "python",
     "用 Python 实现 count_vowels(s), 统计字符串中元音字母个数(不区分大小写)",
     ["def", "vowel", "count", "lower", "return"],
     "count vowels",
     [("print(count_vowels('Hello World'))", "3"),
      ("print(count_vowels('AEIOU'))", "5"),
      ("print(count_vowels('xyz'))", "0")],
     []),
    (17, "easy", "python",
     "用 Python 实现 is_anagram(s, t), 判断 t 是否是 s 的字母异位词",
     ["def", "anagram", "sorted", "count", "return"],
     "check anagram",
     [("print(is_anagram('anagram', 'nagaram'))", "True"),
      ("print(is_anagram('rat', 'car'))", "False"),
      ("print(is_anagram('listen', 'silent'))", "True")],
     []),
    (18, "medium", "python",
     "用 Python 实现 length_of_longest_substring(s), 返回最长无重复字符子串长度",
     ["def", "substring", "set", "sliding", "max"],
     "longest substring no repeat",
     [("print(length_of_longest_substring('abcabcbb'))", "3"),
      ("print(length_of_longest_substring('bbbbb'))", "1"),
      ("print(length_of_longest_substring('pwwkew'))", "3")],
     []),
    (19, "easy", "python",
     "用 Python 实现 max_subarray(nums), 返回最大连续子数组和 (Kadane算法)",
     ["def", "max", "subarray", "current", "return"],
     "max subarray sum",
     [("print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))", "6"),
      ("print(max_subarray([1]))", "1"),
      ("print(max_subarray([-1,-2,-3]))", "-1")],
     []),
    (20, "easy", "python",
     "用 Python 实现 move_zeroes(nums), 将数组中所有 0 移到末尾, 保持非零元素相对顺序, 原地操作",
     ["def", "zero", "swap", "pointer", "return"],
     "move zeroes",
     [("nums=[0,1,0,3,12]; move_zeroes(nums); print(nums)", "[1, 3, 12, 0, 0]"),
      ("nums=[0]; move_zeroes(nums); print(nums)", "[0]")],
     []),
    (21, "medium", "python",
     "用 Python 实现 product_except_self(nums), 返回除自身外所有元素的乘积, 不用除法, O(n)时间",
     ["def", "product", "prefix", "suffix", "return"],
     "product except self",
     [("print(product_except_self([1,2,3,4]))", "[24, 12, 8, 6]"),
      ("print(product_except_self([-1,1,0,-3,3]))", "[0, 0, 9, 0, 0]")],
     []),
    (22, "easy", "python",
     "用 Python 实现 reverse_list(head), 反转单链表, 定义 ListNode 类",
     ["def", "class", "ListNode", "next", "prev"],
     "reverse linked list",
     [("head=ListNode(1,ListNode(2,ListNode(3))); r=reverse_list(head); print(r.val,r.next.val,r.next.next.val)", "3 2 1")],
     []),
    (23, "easy", "python",
     "用 Python 实现 has_cycle(head), 判断链表是否有环 (Floyd 快慢指针)",
     ["def", "cycle", "slow", "fast", "next"],
     "detect cycle",
     [("head=ListNode(1); head.next=ListNode(2); head.next.next=head; print(has_cycle(head))", "True"),
      ("head=ListNode(1,ListNode(2)); print(has_cycle(head))", "False")],
     []),
    (24, "easy", "python",
     "用 Python 实现 is_same_tree(p, q), 判断两棵二叉树是否完全相同, 定义 TreeNode 类",
     ["def", "class", "TreeNode", "val", "left", "right"],
     "same tree",
     [("p=TreeNode(1,TreeNode(2),TreeNode(3)); q=TreeNode(1,TreeNode(2),TreeNode(3)); print(is_same_tree(p,q))", "True"),
      ("p=TreeNode(1,TreeNode(2)); q=TreeNode(1,None,TreeNode(2)); print(is_same_tree(p,q))", "False")],
     []),
    (25, "medium", "python",
     "用 Python 实现 is_valid_bst(root), 验证二叉搜索树是否有效",
     ["def", "bst", "valid", "min", "max", "TreeNode"],
     "validate BST",
     [("root=TreeNode(2,TreeNode(1),TreeNode(3)); print(is_valid_bst(root))", "True"),
      ("root=TreeNode(5,TreeNode(1),TreeNode(4,TreeNode(3),TreeNode(6))); print(is_valid_bst(root))", "False")],
     []),
    (26, "medium", "python",
     "用 Python 实现 coin_change(coins, amount), 返回凑成金额的最少硬币数, 不能凑返回-1",
     ["def", "coin", "dp", "min", "return"],
     "coin change",
     [("print(coin_change([1,5,11], 15))", "3"),
      ("print(coin_change([2], 3))", "-1"),
      ("print(coin_change([1], 0))", "0")],
     []),
    (27, "medium", "python",
     "用 Python 实现 length_of_lis(nums), 返回最长严格递增子序列长度",
     ["def", "lis", "dp", "binary", "return"],
     "longest increasing subseq",
     [("print(length_of_lis([10,9,2,5,3,7,101,18]))", "4"),
      ("print(length_of_lis([0,1,0,3,2,3]))", "4"),
      ("print(length_of_lis([7,7,7,7,7]))", "1")],
     []),
    (28, "medium", "python",
     "用 Python 实现 num_islands(grid), 计算二维网格中岛屿数量 (1=陆地, 0=水)",
     ["def", "island", "dfs", "grid", "return"],
     "count islands",
     [("grid=[['1','1','1'],['0','1','0'],['1','0','0']]; print(num_islands(grid))", "2"),
      ("grid=[['1','0'],['0','1']]; print(num_islands(grid))", "2")],
     []),
    (29, "medium", "python",
     "用 Python 实现 MinStack 类, push/pop/top/get_min 全部 O(1)",
     ["class", "MinStack", "push", "pop", "min"],
     "min stack",
     [("s=MinStack(); s.push(1); s.push(0); print(s.get_min()); s.pop(); print(s.top())", "0\n1")],
     []),
    (30, "hard", "python",
     "用 Python 实现 trap(height), 计算柱状图能接的雨水量, 双指针 O(n)",
     ["def", "trap", "water", "left", "right", "max"],
     "trapping rain water",
     [("print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))", "6"),
      ("print(trap([4,2,0,3,2,5]))", "9")],
     []),
    (31, "hard", "python",
     "用 Python 实现 merge_k_lists(lists), 合并 k 个有序链表, 用最小堆 O(n log k)",
     ["def", "merge", "heap", "ListNode", "return"],
     "merge k sorted lists",
     [("lists=[ListNode(1,ListNode(4,ListNode(5))),ListNode(1,ListNode(3,ListNode(4))),ListNode(2,ListNode(6))]; r=merge_k_lists(lists); print(r.val,r.next.val,r.next.next.val)", "1 1 2")],
     []),
    (32, "hard", "python",
     "用 Python 实现 max_sliding_window(nums, k), 返回滑动窗口最大值, 用双端队列 O(n)",
     ["def", "sliding", "window", "deque", "max"],
     "max sliding window",
     [("print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))", "[3, 3, 5, 5, 6, 7]"),
      ("print(max_sliding_window([1], 1))", "[1]")],
     []),
    (33, "easy", "python",
     "用 Python 实现 is_prime(n), 判断是否为素数, O(sqrt(n))",
     ["def", "prime", "sqrt", "return", "False"],
     "check prime",
     [("print(is_prime(7))", "True"),
      ("print(is_prime(4))", "False"),
      ("print(is_prime(1))", "False")],
     []),
    (34, "medium", "python",
     "用 Python 实现 my_atoi(s), 将字符串转整数, 处理空格/符号/溢出",
     ["def", "atoi", "strip", "int", "overflow"],
     "string to integer",
     [("print(my_atoi('42'))", "42"),
      ("print(my_atoi('   -42'))", "-42"),
      ("print(my_atoi('4193 with words'))", "4193")],
     []),
]


# ============================================================
# 数据类: 单题结果
# ============================================================
@dataclass
class ProblemResult:
    problem_id: str
    difficulty: str
    lang: str
    desc: str
    # 指标
    pass_at_1: float = 0.0       # 0 or 1
    latency_s: float = 0.0       # 推理耗时
    tokens_generated: int = 0
    peak_vram_mb: float = 0.0    # 峰值显存
    # 遗忘相关
    retrieval_count: int = 0     # 检索到的条目数
    forgotten_count: int = 0     # 被遗忘机制过滤的条目数
    old_snippet_ratio: float = 0.0  # 旧代码片段占比
    # 答案
    answer_preview: str = ""


# ============================================================
# 检索模块 (接入 forgetting_retrieval.py)
# ============================================================
class RetrievalInterface:
    """检索模块接口。"""

    def __init__(self, use_forgetting: bool = True, forgetting_threshold: float = 0.7):
        self.use_forgetting = use_forgetting
        self.forgetting_threshold = forgetting_threshold
        self._init_db()

    def _init_db(self):
        """初始化检索模块。"""
        try:
            from forgetting_retrieval import ForgettingRetriever
            self.retriever = ForgettingRetriever(
                db_path="./vector_db",
                records_path="./forgetting_records.json",
                forgetting_threshold=self.forgetting_threshold,
                forgetting_weight=0.3,
            )
            print(f"  [检索] 遗忘感知检索已加载 (forgetting={'ON' if self.use_forgetting else 'OFF'})")
        except Exception as e:
            print(f"  [检索] 加载失败: {e}, 使用空检索")
            self.retriever = None

    def retrieve(self, query: str, top_k: int = 3) -> dict:
        """检索相关代码片段。"""
        if self.retriever is None:
            return {
                "context": "",
                "total_retrieved": 0,
                "filtered_count": 0,
                "old_snippet_ratio": 0.0,
            }

        result = self.retriever.retrieve(
            query, top_k=top_k,
            use_forgetting=self.use_forgetting,
        )
        return result


# ============================================================
# 评分函数 (从 bench_coding_v2.py 精简)
# ============================================================
def extract_code_block(text: str) -> str:
    """提取代码块。"""
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    # 尝试提取 def 开头的代码
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            in_code = True
        if in_code:
            code_lines.append(line)
    return '\n'.join(code_lines) if code_lines else text


def check_execution_python(code: str, test_cases: list) -> float:
    """执行 Python 代码验证。返回 pass rate。"""
    import subprocess
    import tempfile

    if not test_cases:
        return 1.0

    code = extract_code_block(code)
    passed = 0
    for test_input, expected in test_cases:
        full_code = code + "\n" + test_input
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(full_code)
                tmp_path = f.name
            r = subprocess.run([sys.executable, tmp_path],
                               capture_output=True, text=True, timeout=5)
            os.unlink(tmp_path)
            output = r.stdout.strip()
            if expected.lower() in output.lower():
                passed += 1
        except Exception:
            pass
    return passed / max(len(test_cases), 1)


def score_answer(answer: str, keywords: list, lang: str, test_cases: list) -> dict:
    """简化评分。返回各项指标。"""
    # 关键词匹配
    kw_matched = sum(1 for kw in keywords if kw.lower() in answer.lower())
    kw_score = kw_matched / max(len(keywords), 1)

    # 代码结构
    has_code = bool(re.search(r'(def |class |import |return )', answer))
    has_code_block = '```' in answer

    # 语法检查 (Python)
    syntax_ok = 1.0
    if lang == "python" and has_code:
        try:
            import ast
            ast.parse(extract_code_block(answer))
        except SyntaxError:
            syntax_ok = 0.0

    # 执行测试
    exec_pass = check_execution_python(answer, test_cases) if test_cases else 1.0

    # 综合 pass@1: 有代码 + 语法对 + 测试通过
    pass_at_1 = 1.0 if (has_code and syntax_ok >= 0.5 and exec_pass >= 0.5) else 0.0

    return {
        "pass_at_1": pass_at_1,
        "kw_score": kw_score,
        "has_code": has_code,
        "syntax_ok": syntax_ok,
        "exec_pass": exec_pass,
    }


# ============================================================
# 模型加载
# ============================================================
def load_model_hf(model_path: str, device: str = "auto"):
    """加载 HuggingFace 模型。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"  加载 HF 模型: {model_path}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试 4-bit 量化加载 (节省显存)
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print(f"    4-bit 量化加载")
    except Exception as e:
        print(f"    4-bit 加载失败 ({e}), 尝试 fp16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    elapsed = time.time() - t0
    print(f"    耗时: {elapsed:.1f}s")
    return model, tokenizer


def load_model_gguf(model_path: str):
    """加载 GGUF 模型 (llama-cpp-python)。"""
    # 添加 llama_cpp 动态库路径
    os.environ["PATH"] = (
        "F:/Python/Lib/site-packages/llama_cpp/lib"
        + os.pathsep + os.environ.get("PATH", "")
    )
    from llama_cpp import Llama

    print(f"  加载 GGUF 模型: {model_path}")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=8192,
        n_threads=8,
        n_batch=512,
        use_mmap=True,
        verbose=False,
        chat_format="chatml",
    )
    elapsed = time.time() - t0
    print(f"    耗时: {elapsed:.1f}s")
    return llm


# ============================================================
# 推理函数
# ============================================================
def get_peak_vram_mb() -> float:
    """获取当前峰值显存 (MB)。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_vram_stats():
    """重置显存统计。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def generate_hf(model, tokenizer, messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> tuple:
    """HF 模型推理。返回 (text, tokens)。"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return result, len(new_tokens)


def generate_gguf(llm, messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> tuple:
    """GGUF 模型推理。返回 (text, tokens)。"""
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = output["choices"][0]["message"]["content"]
    tokens = output["usage"]["completion_tokens"]
    return text, tokens


# ============================================================
# 单题评测
# ============================================================
def evaluate_problem(
    model, tokenizer_or_llm, problem: tuple,
    retrieval: RetrievalInterface,
    is_gguf: bool = False,
) -> ProblemResult:
    """评测单道题。"""
    idx, difficulty, lang, prompt, keywords, desc, test_cases, ban_imports = problem

    # 检索
    retrieval_result = retrieval.retrieve(prompt, top_k=3)
    context = retrieval_result["context"]

    # 构造 prompt
    if context:
        user_msg = f"参考以下代码片段:\n\n{context}\n\n---\n\n任务: {prompt}\n\n请直接给出完整代码实现, 不要额外解释。"
    else:
        user_msg = f"{prompt}\n\n请直接给出完整代码实现, 不要额外解释。"

    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write complete, runnable code."},
        {"role": "user", "content": user_msg},
    ]

    # 推理
    reset_vram_stats()
    t0 = time.time()
    if is_gguf:
        answer, tokens = generate_gguf(tokenizer_or_llm, messages)
    else:
        answer, tokens = generate_hf(model, tokenizer_or_llm, messages)
    latency = time.time() - t0
    peak_vram = get_peak_vram_mb()

    # 评分
    score = score_answer(answer, keywords, lang, test_cases)

    return ProblemResult(
        problem_id=str(idx),
        difficulty=difficulty,
        lang=lang,
        desc=desc,
        pass_at_1=score["pass_at_1"],
        latency_s=round(latency, 2),
        tokens_generated=tokens,
        peak_vram_mb=round(peak_vram, 1),
        retrieval_count=retrieval_result["total_retrieved"],
        forgotten_count=retrieval_result["filtered_count"],
        old_snippet_ratio=round(retrieval_result["old_snippet_ratio"], 3),
        answer_preview=answer[:200].replace('\n', ' '),
    )


# ============================================================
# 实验配置
# ============================================================
ABLATION_CONFIGS = {
    "full": {
        "name": "Full Model",
        "description": "蒸馏 + RAG + 遗忘检索 + 用户习惯",
        "use_rag": True,
        "use_forgetting": True,
        "model_key": "qwen2.5-3b",  # 蒸馏模型
    },
    "no-forgetting": {
        "name": "No-Forgetting",
        "description": "蒸馏 + 普通RAG (无遗忘机制)",
        "use_rag": True,
        "use_forgetting": False,
        "model_key": "qwen2.5-3b",
    },
    "no-rag": {
        "name": "No-RAG",
        "description": "只有蒸馏模型 (无检索)",
        "use_rag": False,
        "use_forgetting": False,
        "model_key": "qwen2.5-3b",
    },
    "baseline": {
        "name": "Baseline",
        "description": "原始 Qwen2.5-3B (未微调)",
        "use_rag": False,
        "use_forgetting": False,
        "model_key": "qwen2.5-3b-base",
    },
}


# ============================================================
# 运行单组实验
# ============================================================
def run_experiment(
    config_name: str,
    config: dict,
    problems: list,
    model=None, tokenizer_or_llm=None, is_gguf: bool = False,
) -> list:
    """运行一组消融实验。"""
    print(f"\n{'='*60}")
    print(f"实验: {config['name']}")
    print(f"  {config['description']}")
    print(f"  模型: {config.get('model_key', 'N/A')}")
    print(f"  RAG: {'ON' if config['use_rag'] else 'OFF'}")
    print(f"  遗忘: {'ON' if config['use_forgetting'] else 'OFF'}")
    print(f"{'='*60}")

    # 初始化检索模块
    retrieval = RetrievalInterface(
        use_forgetting=config["use_forgetting"],
        forgetting_threshold=0.7,
    )

    results = []
    total_pass = 0
    total_latency = 0.0
    total_vram = 0.0

    for i, problem in enumerate(problems):
        idx = problem[0]
        diff = problem[1]
        desc = problem[5]
        print(f"  [{i+1}/{len(problems)}] #{idx} [{diff}] {desc}", end="")

        result = evaluate_problem(
            model, tokenizer_or_llm, problem,
            retrieval, is_gguf=is_gguf,
        )
        results.append(result)

        total_pass += result.pass_at_1
        total_latency += result.latency_s
        total_vram = max(total_vram, result.peak_vram_mb)

        status = "PASS" if result.pass_at_1 > 0 else "FAIL"
        print(f"  {status} | {result.latency_s:.1f}s | VRAM:{result.peak_vram_mb:.0f}MB")

    # 汇总
    n = len(results)
    avg_pass = total_pass / max(n, 1)
    avg_latency = total_latency / max(n, 1)

    print(f"\n  --- {config['name']} 汇总 ---")
    print(f"  Pass@1: {avg_pass:.3f} ({int(total_pass)}/{n})")
    print(f"  平均延迟: {avg_latency:.2f}s")
    print(f"  峰值显存: {total_vram:.0f}MB")

    return results


# ============================================================
# 保存结果
# ============================================================
def save_results(all_results: list, csv_path: str):
    """保存结果到 CSV。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "experiment", "model", "description",
        "problem_id", "difficulty", "lang", "desc",
        "pass_at_1", "latency_s", "tokens_generated",
        "peak_vram_mb", "retrieval_count", "forgotten_count",
        "old_snippet_ratio", "answer_preview",
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n结果已保存: {csv_path}")


def save_summary(all_results: list, summary_path: str):
    """保存汇总统计。"""
    from collections import defaultdict

    # 按实验分组
    by_exp = defaultdict(list)
    for r in all_results:
        by_exp[r["experiment"]].append(r)

    summary = []
    for exp, results in by_exp.items():
        n = len(results)
        total_pass = sum(r["pass_at_1"] for r in results)
        avg_latency = sum(r["latency_s"] for r in results) / max(n, 1)
        max_vram = max(r["peak_vram_mb"] for r in results)
        avg_forgotten = sum(r["forgotten_count"] for r in results) / max(n, 1)
        avg_old_ratio = sum(r["old_snippet_ratio"] for r in results) / max(n, 1)

        summary.append({
            "experiment": exp,
            "model": results[0]["model"],
            "description": results[0]["description"],
            "num_problems": n,
            "pass_at_1": round(total_pass / n, 3),
            "pass_count": int(total_pass),
            "avg_latency_s": round(avg_latency, 2),
            "peak_vram_mb": round(max_vram, 1),
            "avg_forgotten_count": round(avg_forgotten, 1),
            "avg_old_snippet_ratio": round(avg_old_ratio, 3),
        })

    # 打印汇总表
    print(f"\n{'='*80}")
    print("实验汇总")
    print(f"{'='*80}")
    print(f"{'实验':<20} {'模型':<25} {'Pass@1':>8} {'延迟(s)':>8} {'显存(MB)':>10} {'遗忘过滤':>8}")
    print(f"{'-'*20} {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for s in summary:
        print(f"{s['experiment']:<20} {s['model']:<25} {s['pass_at_1']:>8.3f} {s['avg_latency_s']:>8.2f} {s['peak_vram_mb']:>10.0f} {s['avg_forgotten_count']:>8.1f}")

    # 保存 JSON
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="消融实验 & 跨模型泛化评测")
    parser.add_argument("--model", type=str, default="qwen2.5-3b",
                        help="模型名或路径 (qwen2.5-3b, Phi-3-mini-4k-instruct, TinyLlama-1.1B)")
    parser.add_argument("--ablation", type=str, default="all",
                        help="消融配置 (full,no-forgetting,no-rag,baseline 或 all)")
    parser.add_argument("--benchmark_path", type=str, default=None,
                        help="benchmark 问题文件路径 (可选, 默认用内置 PROBLEMS)")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试 (只跑前 5 题)")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR,
                        help="输出目录")
    parser.add_argument("--forgetting_threshold", type=float, default=0.7,
                        help="遗忘分数阈值")
    args = parser.parse_args()

    # 加载 benchmark
    problems = PROBLEMS
    if args.benchmark_path:
        # TODO: 从文件加载 benchmark
        print(f"从 {args.benchmark_path} 加载 benchmark...")
        # 示例: exec(open(args.benchmark_path).read()) 然后取 PROBLEMS
        pass

    if args.quick:
        problems = problems[:5]
        print(f"快速模式: 只跑前 {len(problems)} 题")

    # 解析消融配置
    if args.ablation == "all":
        ablation_names = list(ABLATION_CONFIGS.keys())
    else:
        ablation_names = [x.strip() for x in args.ablation.split(",")]

    # 确定模型
    model_key = args.model
    model_path = MODEL_PATHS.get(model_key, model_key)

    # 判断是否 GGUF
    is_gguf = model_path.endswith(".gguf")

    # 加载模型
    print(f"\n{'='*60}")
    print(f"加载模型: {model_key} ({model_path})")
    print(f"{'='*60}")

    model = None
    tokenizer_or_llm = None

    if is_gguf:
        tokenizer_or_llm = load_model_gguf(model_path)
    else:
        model, tokenizer_or_llm = load_model_hf(model_path)

    # 运行实验
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for config_name in ablation_names:
        if config_name not in ABLATION_CONFIGS:
            print(f"未知配置: {config_name}, 跳过")
            continue

        config = ABLATION_CONFIGS[config_name].copy()

        # 跨模型实验时, 强制使用指定模型
        if model_key != "qwen2.5-3b":
            config["model_key"] = model_key
            if config_name == "baseline":
                # 跨模型的 baseline 用同一个模型但无 RAG
                config["name"] = f"Baseline ({model_key})"
                config["description"] = f"{model_key} 无检索"

        results = run_experiment(
            config_name, config, problems,
            model, tokenizer_or_llm, is_gguf,
        )

        for r in results:
            row = asdict(r)
            row["experiment"] = config_name
            row["model"] = config.get("model_key", model_key)
            row["description"] = config["description"]
            all_results.append(row)

    # 保存
    csv_path = os.path.join(args.output_dir, f"ablation_{model_key.replace('/', '_')}.csv")
    summary_path = os.path.join(args.output_dir, f"summary_{model_key.replace('/', '_')}.json")

    save_results(all_results, csv_path)
    save_summary(all_results, summary_path)

    # 清理
    del model, tokenizer_or_llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n完成!")


if __name__ == "__main__":
    main()
