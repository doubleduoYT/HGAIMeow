#!/usr/bin/env python3
"""Build small, *diverse* v14 chat SFT data from the approved HGAIM v13 sources.

Unlike v13 this does not multiply each source phrase into 3 identical targets.
All source groups are partitioned already. Preserve unmodified data artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from nyang_style_v14 import nyangify

SYS_KO = ("너는 HGAIM이야. 친절하고 정확한 한국어로 대화하는 고양이 AI다냥. "
          "한국어 설명의 각 문장은 자연스럽게 냥체로 마무리해. "
          "게임 질문은 실제로 확인된 사실만 답하고, 모르는 설정을 지어내지 마. "
          "검색은 사용하지 않는다냥.")
SYS_EN = ("You are HGAIM, a helpful bilingual cat-themed AI. Answer in fluent English, "
          "use a light 'meow' in conversational responses, and be accurate. "
          "Separate uncertain fan claims from confirmed game facts. No internet search.")
PREFIXES = ("In Casualties: Unknown,", "C:U에 나오는")
IGNORE = ("<sprite", "<color", "\x00")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_fact(record):
    q, a = record["user"].strip(), record["assistant"].strip()
    lang = record["language"]
    if not q or not a or len(q) > 220 or any(flag in a for flag in IGNORE):
        return None
    # Preserve knowledge content, drop repetitive training-template boilerplate.
    if a.startswith("In Casualties: Unknown,") and " is described as follows: " in a:
        label_and_definition = a[len("In Casualties: Unknown, "):]
        name, meaning = label_and_definition.split(" is described as follows: ", 1)
        a = f"In C:U, {name} is described as: {meaning}"
    elif a.startswith("C:U에 나오는") and "의 게임 내 설명은 이래: " in a:
        label_and_definition = a[len("C:U에 나오는 "):]
        name, meaning = label_and_definition.split("의 게임 내 설명은 이래: ", 1)
        a = f"C:U의 {name} 설명에 따르면, {meaning}"
    if len(a) > 410:
        return None
    if len(a) < 12:
        return None
    a = nyangify(a, lang)
    return {"source_group": record["group"], "kind": "game", "language": lang,
            "messages": [{"role": "system", "content": SYS_KO if lang == "ko" else SYS_EN},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def convert_dialogue(record, source_id):
    turns = record.get("turns", [])
    if not 2 <= len(turns) <= 4:
        return None
    lang = "ko" if re.search("[가-힣]", turns[0]["user"]) else "en"
    messages = [{"role": "system", "content": SYS_KO if lang == "ko" else SYS_EN}]
    for turn in turns:
        q = turn["user"].strip()
        a = turn["assistant"].strip()
        if not q or not a:
            return None
        messages.extend([{"role": "user", "content": q},
                         {"role": "assistant", "content": nyangify(a, lang)}])
    return {"source_group": source_id, "kind": "dialogue", "language": lang,
            "messages": messages}


def prepare(args):
    rng = random.Random(args.seed)
    train_games = load_jsonl(args.game_train)
    heldout_games = load_jsonl(args.game_heldout)
    if {r["group"] for r in train_games} & {r["group"] for r in heldout_games}:
        raise ValueError("Game fact leakage: group present in both partitions")
    # One paraphrase per language and factual key. Repeated copies do not add knowledge.
    dedup = set()
    game_samples = []
    for record in train_games:
        key = (record["group"], record["language"])
        if key in dedup:
            continue
        dedup.add(key)
        converted = convert_fact(record)
        if converted:
            game_samples.append(converted)
    by_lang = {lang: [r for r in game_samples if r["language"] == lang] for lang in ("ko", "en")}
    # Cap broad generated facts; preserve ALL curated named-game facts, by kind.
    curated = [r for r in game_samples if r["source_group"].startswith("game:")]
    curated_ids = {r["source_group"] + ":" + r["language"] for r in curated}
    sampled = curated[:]
    for lang in ("ko", "en"):
        others = [r for r in by_lang[lang] if r["source_group"] + ":" + lang not in curated_ids]
        rng.shuffle(others)
        sampled.extend(others[:args.max_localized_per_lang])
    dialogues = []
    for path in args.dialogue_files:
        records = load_jsonl(path)
        for index, record in enumerate(records):
            converted = convert_dialogue(record, f"dialogue:{Path(path).name}:{index}")
            if converted:
                dialogues.append(converted)
    rng.shuffle(sampled)
    rng.shuffle(dialogues)
    if not dialogues or not sampled:
        raise ValueError("No usable training conversations/facts")

    val_games = []
    holdout_seen = set()
    for record in heldout_games:
        key = (record["group"], record["language"])
        if key in holdout_seen:
            continue
        holdout_seen.add(key)
        converted = convert_fact(record)
        if converted:
            val_games.append(converted)
    rng.shuffle(val_games)
    validation = val_games[:args.validation_count]
    # Fixed independent dialogue prompts, never used as training data.
    fixed = [
        ("ko", "C:U에 나오는 SAW 연구에 대해 설명해줘.", "그 프로젝트의 목적이 뭐였어?"),
        ("ko", "Gunsaw와 Casualties: Unknown은 어떤 관계야?", "차이점은 뭘까?"),
        ("ko", "마인크래프트의 레드스톤으로 뭘 만들 수 있어?", "초보자가 할 만한 건?"),
        ("ko", "오늘 코딩을 하다가 오류가 났어.", "원인을 어떻게 좁히지?"),
        ("en", "What is ADOFAI about?", "How is that different from Rhythm Doctor?"),
        ("en", "Can you explain what Changed is?", "Who is Puro?"),
        ("en", "My program crashed on Android.", "What should I check first?"),
        ("en", "Tell me about Minecraft redstone.", "Suggest a first experiment."),
    ]
    for index, (lang, q1, q2) in enumerate(fixed):
        validation.append({
            "kind": "independent_generation", "language": lang, "source_group": f"heldout-independent:{index}",
            "messages": [{"role": "system", "content": SYS_KO if lang == "ko" else SYS_EN},
                         {"role": "user", "content": q1},
                         {"role": "user", "content": q2}]})
    # Ensure the exact independent question text never appears in training text.
    train_questions = {m["content"].casefold().strip() for record in sampled+dialogues for m in record["messages"] if m["role"] == "user"}
    for record in validation:
        for m in record["messages"]:
            if m["role"] == "user" and m["content"].casefold().strip() in train_questions:
                raise ValueError(f"Exact validation question reused: {m['content']}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = dialogues + sampled
    rng.shuffle(train)
    for filename, records in [("train_v14.jsonl", train), ("heldout_v14.jsonl", validation)]:
        (output / filename).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    report = {
        "version": "v14-pretrained-lora",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "training_records": len(train), "dialogue_records": len(dialogues),
        "game_records": len(sampled), "heldout_records": len(validation),
        "split_rule": "game facts split atomically by v13 source group; no repeated answer paraphrases",
        "training_languages": dict(Counter(x["language"] for x in train)),
        "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in [args.game_train, args.game_heldout, *args.dialogue_files]},
        "warning": "Localization descriptions are authored game data, not verified runtime effects. Data rights/attribution review required."
    }
    (output / "report_v14.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-train", default="v13_game_data/game_qa_v13_train.jsonl")
    parser.add_argument("--game-heldout", default="v13_game_data/game_qa_v13_holdout.jsonl")
    parser.add_argument("--dialogue-files", nargs="+", default=[
        "train_dialogues_v11.jsonl", "train_dialogues_v11_1_extra.jsonl", "train_dialogues_v12.jsonl"])
    parser.add_argument("--max-localized-per-lang", type=int, default=220)
    parser.add_argument("--validation-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", default="v14_data")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
