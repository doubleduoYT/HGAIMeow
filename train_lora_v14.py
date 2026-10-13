#!/usr/bin/env python3
"""CPU-friendly LoRA continuation for a PRETRAINED bilingual language model.

An experiment, not a claim that CPU LoRA will improve open-domain fluency.
Keep base weights frozen; never train the 0.5B model from scratch on Actions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def dataset(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_chat(tokenizer, record, max_length):
    messages = record["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("Cannot train on a record without a final assistant response")
    # Standard Qwen chat template, with all preceding turns as prefix and loss
    # applied ONLY to assistant tokens, never to system/user examples.
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    start = max(0, len(full_ids) - max_length)
    ids = full_ids[start:]
    assistant_start = max(0, len(prefix_ids) - start)
    labels = [-100] * min(assistant_start, len(ids)) + ids[assistant_start:]
    if sum(label != -100 for label in labels) < 5:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def stack_batch(records, pad):
    seq_len = max(len(r["input_ids"]) for r in records)
    ids = torch.full((len(records), seq_len), pad, dtype=torch.long)
    mask = torch.zeros((len(records), seq_len), dtype=torch.long)
    labels = torch.full((len(records), seq_len), -100, dtype=torch.long)
    for index, record in enumerate(records):
        n = len(record["input_ids"])
        ids[index, :n] = torch.tensor(record["input_ids"])
        mask[index, :n] = 1
        labels[index, :n] = torch.tensor(record["labels"])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@torch.no_grad()
def evaluate(model, tokens, pad, limit=12):
    model.eval()
    vals = []
    for record in tokens[:limit]:
        batch = stack_batch([record], pad)
        outputs = model(**batch)
        vals.append(float(outputs.loss.detach()))
    model.train()
    return sum(vals) / max(1, len(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="v14_data/train_v14.jsonl")
    p.add_argument("--valid", default="v14_data/heldout_v14.jsonl")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--output", default="hgaim_v14_lora")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.00012)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--eval-every", type=int, default=12)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261013)
    args = p.parse_args()
    if not (1 <= args.steps <= 3000 and 64 <= args.max_length <= 512 and args.threads >= 1):
        p.error("Invalid resource limits")
    if args.lr <= 0:
        p.error("Learning rate must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_train, raw_val = dataset(args.train), dataset(args.valid)
    val = [r for r in raw_val if r.get("kind") != "independent_generation"]
    train = [x for row in raw_train if (x := encode_chat(tokenizer, row, args.max_length))]
    valid = [x for row in val if (x := encode_chat(tokenizer, row, args.max_length))]
    if not train or not valid:
        raise SystemExit("No usable train/validation token examples")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.train()
    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=args.lr, weight_decay=0.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    base_loss = evaluate(model, valid, tokenizer.pad_token_id)
    best_val = base_loss
    best_step = 0
    history = []
    print(json.dumps({
        "base_model": args.base_model, "train_samples": len(train),
        "holdout_samples": len(valid), "baseline_loss": base_loss,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        sample = rng.choice(train)
        batch = stack_batch([sample], tokenizer.pad_token_id)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite LoRA training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % args.eval_every or step == 0:
            if step != args.steps:
                continue
        val_loss = evaluate(model, valid, tokenizer.pad_token_id)
        improved = val_loss < best_val - .002
        if improved:
            best_val, best_step = val_loss, step
            model.save_pretrained(output)
        item = {"step": step, "train_loss": float(loss.detach()), "holdout_loss": val_loss,
                "improved": improved}
        history.append(item)
        print(json.dumps(item), flush=True)
    # If it never improved the heldout criterion, save final weights clearly as
    # experimental, not as a promoted candidate.
    if best_step == 0:
        model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    report = {
        "base_model": args.base_model, "steps_completed": args.steps,
        "selected_step": best_step, "baseline_holdout_loss": base_loss,
        "selected_holdout_loss": best_val if best_step else None,
        "last_holdout_loss": history[-1]["holdout_loss"], "improved_holdout": bool(best_step),
        "max_length": args.max_length, "train_samples": len(train),
        "validation_samples": len(valid), "validation_method": "teacher-forced assistant-only loss",
        "training": history,
        "warning": "Heldout loss is not conversational quality. Review actual raw generation before deploying. LoRA files are not directly a llama.cpp GGUF.",
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="training"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
