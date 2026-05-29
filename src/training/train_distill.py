"""
Qwen2.5-3B 蒸馏微调 (389k merged data, ~18h on RTX 4060 8GB)
用法: python train_distill_18h.py
"""
import os
os.environ["USE_TF"] = "0"

import torch, json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ===== 配置 =====
MODEL_PATH = "./Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./output/qwen25-distill-389k"
DATA_PATH = "./data/merged_distillation.jsonl"
MERGE_SAVE_STEPS = 5000       # save checkpoint every N steps
MAX_LENGTH = 2048
EPOCHS = 1
# ================

def convert_to_chat(example):
    """Convert any schema to Qwen chat format messages."""
    keys = set(example.keys())

    # Already processed: source_text + target_text (GPT-5/Stratos)
    if "source_text" in example:
        return [
            {"role": "user", "content": example["source_text"]},
            {"role": "assistant", "content": example["target_text"]},
        ]

    # UltraChat / No Robots: messages list already OpenAI format
    if "messages" in example:
        msgs = example["messages"]
        # Keep as-is, already {role, content}
        return msgs

    # ShareGPT: conversations [{from, value}]
    if "conversations" in example:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        msgs = []
        for turn in example["conversations"]:
            role = role_map.get(turn.get("from", ""), "user")
            msgs.append({"role": role, "content": turn.get("value", "")})
        return msgs

    # Alpaca: instruction + optional input -> output
    if "instruction" in example and "output" in example:
        inst = example["instruction"]
        inp = example.get("input", "") or ""
        content = inst + ("\n" + inp if inp else "")
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": example["output"]},
        ]

    # Dolly: instruction + optional context -> response
    if all(k in example for k in ("instruction", "response")):
        ctx = example.get("context", "") or ""
        inst = example["instruction"]
        content = inst + ("\n" + ctx if ctx else "")
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": example["response"]},
        ]

    # Fallback: try to use first string field
    print(f"  WARN unknown schema: {list(example.keys())}")
    return [{"role": "user", "content": str(list(example.values())[0])},
            {"role": "assistant", "content": str(list(example.values())[-1])}]


def tokenize_fn(examples):
    """Convert examples to Qwen chat template, mask assistant loss."""
    all_input_ids = []
    all_labels = []
    all_attn = []

    tokenizer = tokenize_fn._tokenizer

    for i in range(len(examples[list(examples.keys())[0]])):
        example = {k: examples[k][i] for k in examples}
        msgs = convert_to_chat(example)

        # Apply Qwen2.5 chat template
        # Format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )

        encoded = tokenizer(
            full_text, truncation=True, max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        # Find assistant <|im_start|> tokens to mask loss
        # Labels: -100 for user/system, input_ids for assistant
        label = [-100] * len(input_ids)

        # Find assistant turns in chat template output
        template = "<|im_start|>assistant"
        template_ids = tokenizer.encode(template, add_special_tokens=False)

        # Scan for assistant token boundaries
        start = 0
        while start < len(input_ids):
            # Find next <|im_start|>assistant
            pos = None
            for j in range(start, len(input_ids) - len(template_ids) + 1):
                if input_ids[j:j+len(template_ids)] == template_ids:
                    pos = j
                    break
            if pos is None:
                break
            # Find end of this assistant block (next <|im_start|> or <|im_end|>)
            end = pos + len(template_ids)
            label[pos:end] = [-100] * (end - pos)  # mask the "assistant" tag itself
            # Find where assistant content ends (next <|im_start|>)
            sep_token_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
            next_sep = len(input_ids)
            for j in range(end, len(input_ids)):
                if input_ids[j:j+len(sep_token_ids)] == sep_token_ids:
                    next_sep = j
                    break
            label[end:next_sep] = input_ids[end:next_sep]  # train on assistant content
            start = next_sep

        all_input_ids.append(input_ids)
        all_labels.append(label)
        all_attn.append(encoded["attention_mask"])

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attn,
    }


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LENGTH

    # 2. 4-bit quant
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 3. LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    ))
    model.print_trainable_parameters()
    model.config.use_cache = False
    print(f"VRAM after LoRA: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 4. Load + tokenize data
    print("\n[LOAD] Loading dataset...")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"  Total: {len(ds)} samples")

    # Attach tokenizer to tokenize_fn
    tokenize_fn._tokenizer = tokenizer

    print("[TOKENIZE] Converting all schemas to Qwen chat format...")
    # Process in batches to show progress
    ds = ds.map(
        tokenize_fn, batched=True, batch_size=500,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    # Filter out empty/invalid samples
    ds = ds.filter(lambda x: len(x["input_ids"]) > 10 and len(x["input_ids"]) <= MAX_LENGTH)
    print(f"  Valid samples: {len(ds)}")

    # 5. Training args
    accum_steps = 8
    total_steps = len(ds) // accum_steps
    est_sec = total_steps * 1.0  # ~1s per step
    est_hours = est_sec / 3600
    print(f"\n  Steps: {total_steps} | Est: {est_hours:.1f}h")
    print(f"  Save every {MERGE_SAVE_STEPS} steps -> ~{MERGE_SAVE_STEPS * accum_steps} samples/ckpt")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accum_steps,
        learning_rate=2e-4,
        warmup_steps=50,
        fp16=True,
        logging_steps=10,
        save_steps=MERGE_SAVE_STEPS,
        save_total_limit=3,           # keep only 3 recent checkpoints
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        optim="adamw_8bit",            # saves VRAM
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=default_data_collator,
    )

    print(f"\n[START] Training for ~{est_hours:.1f}h...")
    print(f"  Samples: {len(ds)} | Accum: {accum_steps} | Effective batch: {accum_steps}")
    print(f"  Max length: {MAX_LENGTH} | LR: 2e-4 cosine | Save every {MERGE_SAVE_STEPS} steps")
    trainer.train()

    # Save final
    print("\n[SAVE] Saving final model...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Done! Model saved to {OUTPUT_DIR}/final")
