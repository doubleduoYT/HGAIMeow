#!/usr/bin/env python3
"""Neural-first bilingual multi-turn trainer, compatible with v10 checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import hgai_core as h
import train_v10_quality as quality


def dialogue_examples(path):
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        turns = json.loads(line)["turns"]
        history = []
        if len(turns) < 2:
            raise ValueError(f"{path}:{line_no}: expected at least two dialogue turns")
        for turn in turns:
            question = str(turn["user"]).strip()
            answer = str(turn["assistant"]).strip()
            if not question or not answer:
                raise ValueError(f"{path}:{line_no}: empty dialogue field")
            records.append((question, answer, tuple(history[-2:])))
            history.append((question, answer))
    return records


def dialogue_example(tok, cfg, record):
    question, answer, history = record
    prompt = f"<|system|> {h.system_text_for_question(question)}<|end|>\n"
    prompt += "".join(
        f"<|user|> {q}\n<|assistant|> {a}<|end|>\n"
        for q, a in history
    )
    prompt += f"<|user|> {question}\n<|assistant|> "
    input_ids = tok.encode(prompt)
    targets = tok.encode(answer + "<|end|>")
    ids = input_ids + targets
    labels = [-100] * len(input_ids) + targets
    return ids[-cfg["block_size"]:], labels[-cfg["block_size"]:]


def install_neural_sampler(dialogues, dialogue_ratio, english_ratio, focus_ratio):
    base_sampler = quality.make_batch
    cache = {"pool_id": None, "english": []}

    def sampler(tok, cfg, pairs, device, rng, ctx_prob=.22,
                focus_pairs=None, focus_prob=0.0, focus_ctx_prob=.08):
        if focus_pairs is None:
            return base_sampler(tok, cfg, pairs, device, rng,
                                ctx_prob=0.0, focus_pairs=None, focus_prob=0.0)
        if cache["pool_id"] != id(pairs):
            cache["english"] = [item for item in pairs if h.is_english_question(item[0])]
            cache["pool_id"] = id(pairs)
        examples = []
        for _ in range(cfg["batch_size"]):
            if dialogues and rng.random() < dialogue_ratio:
                examples.append(dialogue_example(
                    tok, cfg, dialogues[rng.randrange(len(dialogues))]
                ))
                continue
            roll = rng.random()
            if cache["english"] and roll < english_ratio:
                pool = cache["english"]
            elif focus_pairs and roll < english_ratio + focus_ratio:
                pool = focus_pairs
            else:
                pool = pairs
            q, a = pool[rng.randrange(len(pool))]
            examples.append(h.example(tok, cfg, q, a, ctx=False))
        max_length = max(len(ids) for ids, _ in examples)
        x = torch.full((len(examples), max_length), tok.pad, dtype=torch.long)
        y = torch.full((len(examples), max_length), -100, dtype=torch.long)
        for i, (ids, labels) in enumerate(examples):
            x[i, :len(ids)] = torch.tensor(ids)
            y[i, :len(labels)] = torch.tensor(labels)
        return x.to(device), y.to(device)

    quality.make_batch = sampler


def main():
    parser = argparse.ArgumentParser(description="HGAIM v11 neural-first training")
    parser.add_argument("--train-file", default="train.txt")
    parser.add_argument("--dialogues-file", default="train_dialogues_v11.jsonl")
    parser.add_argument("--model-file", default="hgai_model_v11.pth")
    parser.add_argument("--init-model-file", default="hgai_model_v10_preview.pth")
    parser.add_argument("--preset", choices=sorted(h.PRESETS), default="main")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dialogue-ratio", type=float, default=0.35)
    parser.add_argument("--english-ratio", type=float, default=0.25)
    parser.add_argument("--focus-ratio", type=float, default=0.12)
    args = parser.parse_args()
    for name in ("dialogue_ratio", "english_ratio", "focus_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"{name} must be in [0,1]")
    if args.english_ratio + args.focus_ratio > 1:
        parser.error("english_ratio + focus_ratio cannot exceed 1")
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.resume:
        args.init_model_file = ""
    elif not Path(args.init_model_file).is_file():
        parser.error(f"pretrained checkpoint not found: {args.init_model_file}")
    dialogues = dialogue_examples(args.dialogues_file)
    digest = hashlib.sha256(Path(args.dialogues_file).read_bytes()).hexdigest()
    print(f"v11: dialogue turns={len(dialogues)}, sha256={digest}")
    print("No retrieval context during neural fine-tuning")
    install_neural_sampler(dialogues, args.dialogue_ratio, args.english_ratio, args.focus_ratio)
    quality.train(args)
    metadata = {
        "version": "neural-first-v11",
        "dialogue_turns": len(dialogues),
        "dialogue_sha256": digest,
        "checkpoint": args.model_file,
        "steps_requested": args.steps,
        "ratios": {
            "dialogues": args.dialogue_ratio,
            "english": args.english_ratio,
            "facts": args.focus_ratio
        },
        "checkpoint_format": "hgai-v10-compatible",
        "note": "Independent raw-neural evaluation is still required."
    }
    Path(args.model_file).with_suffix(".training.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
