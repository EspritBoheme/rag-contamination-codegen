"""
Convert LMSYS-Chat-GPT-5 dataset to Qwen3.5-4B distillation format.
Output: JSONL with source_text + target_text, ready for SFT/QLoRA training.
"""
from datasets import load_from_disk
from transformers import AutoTokenizer
import json
import os

# ===== CONFIG =====
MODEL_PATH = "./Qwen3.5-4B"
INPUT_PATH = "data/lmsys_gpt5_coding_clean"
OUTPUT_PATH = "data/gpt5_distill_ready.jsonl"
# ==================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading dataset...")
ds = load_from_disk(INPUT_PATH)
print(f"Total samples: {len(ds)}")

# Track stats
total_tokens = 0
too_long = 0
skipped = 0
MAX_TOKENS = 4096  # Qwen3.5-4B max context

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for idx, sample in enumerate(ds):
        # Convert OpenAI message format to Qwen chat format (only user side)
        # Original: [system, user] -> we use only the user part + teacher response
        messages = sample["content"]
        user_content = messages[-1]["content"]  # last message is user

        # Some user messages have Alpaca format wrapping, strip it
        # Keep it clean - use user msg directly
        source_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True  # adds <|im_start|>assistant\n
        )

        target_text = sample["teacher_response"] + "<|im_end|>"

        # Check token length
        source_tokens = len(tokenizer.encode(source_text))
        target_tokens = len(tokenizer.encode(target_text))
        total = source_tokens + target_tokens

        if total > MAX_TOKENS:
            too_long += 1
            continue

        total_tokens += total

        record = {
            "id": sample["id"],
            "source_text": source_text,
            "target_text": target_text,
            "category": sample["category"],
            "source_len": source_tokens,
            "target_len": target_tokens,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nSaved: {OUTPUT_PATH}")
print(f"  Samples written: {idx + 1 - skipped - too_long}")
print(f"  Skipped (too long >{MAX_TOKENS} tokens): {too_long}")
print(f"  Avg total tokens: {total_tokens // max(1, idx + 1 - too_long)}")

# Quick stats on a few samples
print("\n=== Sample output ===")
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    line = json.loads(f.readline())
    print("SOURCE:", line["source_text"][:200])
    print("TARGET:", line["target_text"][:200])
    print(f"Source tokens: {line['source_len']}, Target tokens: {line['target_len']}")
