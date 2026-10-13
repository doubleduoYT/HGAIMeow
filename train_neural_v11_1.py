#!/usr/bin/env python3
"""HGAIM v11.1: checkpoint selection by held-out dialogue loss.

The held-out dialogues are never used for gradients. A separate generative test
must still be run after training; token loss does not establish conversation quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import hgai_core as h
import train_v10_quality as quality
from train_neural_v11 import dialogue_example, dialogue_examples


def read_dialogues(paths):
    return [item for path in paths for item in dialogue_examples(path)]


def make_tensors(tok, examples, device):
    length = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), length), tok.pad, dtype=torch.long)
    y = torch.full((len(examples), length), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)


@torch.no_grad()
def dialogue_loss(model, tok, cfg, records, device, batch_size=4):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(records), batch_size):
        examples = [dialogue_example(tok, cfg, item) for item in records[start:start + batch_size]]
        x, y = make_tensors(tok, examples, device)
        _, loss = model(x, y)
        tokens = int((y[:, 1:] != -100).sum().item())
        if tokens:
            total_loss += float(loss) * tokens
            total_tokens += tokens
    return total_loss / max(1, total_tokens)


def training_batch(tok, cfg, training, english, facts, dialogues, rng, device,
                   dialogue_ratio, english_ratio, fact_ratio):
    examples = []
    for _ in range(cfg["batch_size"]):
        roll = rng.random()
        if roll < dialogue_ratio:
            examples.append(dialogue_example(tok, cfg, rng.choice(dialogues)))
            continue
        roll -= dialogue_ratio
        if english and roll < english_ratio:
            source = english
        elif facts and roll < english_ratio + fact_ratio:
            source = facts
        else:
            source = training
        q, answer = rng.choice(source)
        examples.append(h.example(tok, cfg, q, answer, ctx=False))
    return make_tensors(tok, examples, device)


def ensure_holdout(train_dialogues, validation):
    train_questions = {h.normalize(question) for question, _, _ in train_dialogues}
    valid_questions = {h.normalize(question) for question, _, _ in validation}
    overlaps = train_questions & valid_questions
    if overlaps:
        raise ValueError(f"training/validation dialogue questions overlap: {sorted(overlaps)[:5]}")
    if not train_dialogues or not validation:
        raise ValueError("Training and holdout dialogue files must both contain records")


def main():
    p = argparse.ArgumentParser(description="Train HGAIM v11.1 using held-out dialogue selection")
    p.add_argument("--train-file", default="train.txt")
    p.add_argument("--dialogues-file", action="append", default=[])
    p.add_argument("--validation-dialogues", default="validate_dialogues_v11_1.jsonl")
    p.add_argument("--model-file", default="hgai_model_v11_1.pth")
    p.add_argument("--init-model-file", default="hgai_model_v10_preview.pth")
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--dialogue-ratio", type=float, default=0.55)
    p.add_argument("--english-ratio", type=float, default=0.20)
    p.add_argument("--fact-ratio", type=float, default=0.07)
    p.add_argument("--general-loss-tolerance", type=float, default=0.60)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    if not args.dialogues_file:
        args.dialogues_file = ["train_dialogues_v11.jsonl", "train_dialogues_v11_1_extra.jsonl"]
    if args.steps < 1 or args.eval_every < 1 or args.lr <= 0:
        p.error("Steps, eval-every and learning rate must be positive")
    if min(args.dialogue_ratio, args.english_ratio, args.fact_ratio) < 0 or sum(
        (args.dialogue_ratio, args.english_ratio, args.fact_ratio)
    ) > 1:
        p.error("Training sampling ratios must be nonnegative and sum to at most 1")
    torch.set_num_threads(args.threads if args.threads > 0 else 2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    train_dialogues = read_dialogues(args.dialogues_file)
    valid_dialogues = dialogue_examples(args.validation_dialogues)
    ensure_holdout(train_dialogues, valid_dialogues)
    source = Path(args.init_model_file)
    ck = h.checkpoint_load(source)
    if not ck or ck.get("version") != "hgai-v10" or ck.get("preset") != "main":
        raise SystemExit(f"Missing compatible main checkpoint: {source}")
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    cfg = dict(ck["config"])
    tok = h.Tokenizer(ck["vocab"])
    model = h.HGAIModel(len(tok.vocab), cfg).to(device)
    model.load_state_dict(ck["model"])
    model.train()

    all_pairs = quality.load_training_pairs(args.train_file)
    train_pairs, general_validation = h.split_pairs(all_pairs)
    english = [row for row in train_pairs if h.is_english_question(row[0])]
    focus = []
    focus_path = Path(args.train_file).with_name("train_v10_extra.txt")
    if focus_path.exists():
        focus_questions = {h.normalize(q) for q, _ in h.parse_pairs(focus_path.read_text(encoding="utf-8"))}
        focus = [row for row in train_pairs if h.normalize(row[0]) in focus_questions]
    if not focus:
        focus = train_pairs[: min(len(train_pairs), 1000)]
    general_validation = general_validation[:]
    if not general_validation:
        raise ValueError("No general validation examples available")

    baseline_dialogue = dialogue_loss(model, tok, cfg, valid_dialogues, device)
    baseline_general = quality.eval_raw_loss(
        model, tok, cfg, general_validation, device, args.seed + 2718, batches=8
    )
    best_score = baseline_dialogue
    best_dialogue = baseline_dialogue
    best_general = baseline_general
    best_step = 0
    best_model = quality.cpu_state_dict(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.04)
    eval_history = []
    print(json.dumps({
        "train_dialogue_turns": len(train_dialogues),
        "holdout_dialogue_turns": len(valid_dialogues),
        "train_english_pairs": len(english),
        "params": h.count_params(model),
        "initial_dialogue_loss": baseline_dialogue,
        "initial_general_loss": baseline_general,
        "selection": "dialogue_loss + 0.4 * general_regression, bounded by general tolerance",
    }, ensure_ascii=False))

    for step in range(1, args.steps + 1):
        model.train()
        x, y = training_batch(
            tok, cfg, train_pairs, english, focus, train_dialogues, rng, device,
            args.dialogue_ratio, args.english_ratio, args.fact_ratio,
        )
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        warmup = min(70, max(5, args.steps // 10))
        lr_scale = step / warmup if step <= warmup else max(
            0.2, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        val_dialogue = dialogue_loss(model, tok, cfg, valid_dialogues, device)
        val_general = quality.eval_raw_loss(
            model, tok, cfg, general_validation, device, args.seed + 2718, batches=8
        )
        general_regression = max(0.0, val_general - baseline_general)
        score = val_dialogue + 0.4 * general_regression
        eligible = general_regression <= args.general_loss_tolerance
        selected = bool(eligible and score < best_score - 0.002)
        if selected:
            best_model = quality.cpu_state_dict(model)
            best_score = score
            best_dialogue = val_dialogue
            best_general = val_general
            best_step = step
        report = {
            "step": step, "train_loss": float(loss), "dialogue_loss": val_dialogue,
            "general_loss": val_general, "general_regression": general_regression,
            "selection_score": score, "eligible": eligible, "selected": selected,
        }
        eval_history.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    final_state = quality.cpu_state_dict(model)
    checkpoint = {
        "version": "hgai-v10", "preset": "main", "config": cfg, "vocab": tok.vocab,
        "model": best_model, "resume_model": final_state,
        "step": int(ck.get("step", 0)) + args.steps,
        "best_step": best_step, "best_val": best_general, "last_val": eval_history[-1]["general_loss"],
        "params": h.count_params(model), "trainer": "v11.1-dialogue-selected",
        "semantic_ready": False, "artifact_ready": False, "raw_neural_gate": False,
        "selection_score": best_score, "selection_dialogue_loss": best_dialogue,
        "baseline_dialogue_loss": baseline_dialogue,
        "baseline_general_loss": baseline_general,
        "selected_improved_over_baseline": best_step > 0,
        "source_checkpoint": str(source),
    }
    torch.save(checkpoint, args.model_file)
    manifest = {
        "version": "neural-first-v11.1", "selected_step": best_step,
        "selected_improved_over_baseline": best_step > 0,
        "dialogue_train_count": len(train_dialogues),
        "dialogue_holdout_count": len(valid_dialogues),
        "baseline_dialogue_loss": baseline_dialogue,
        "chosen_dialogue_loss": best_dialogue,
        "baseline_general_loss": baseline_general,
        "chosen_general_loss": best_general,
        "selection_history": eval_history,
        "train_files_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [*args.dialogues_file, args.validation_dialogues]
        },
        "selected_model_note": "A lower held-out dialogue loss is not proof of helpful free-form conversation. Human review is mandatory.",
    }
    manifest_path = Path(args.model_file).with_suffix(".training.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("selection_history", "train_files_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
