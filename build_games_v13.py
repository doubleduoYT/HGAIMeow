#!/usr/bin/env python3
"""Build bilingual game Q/A from independently sourced localizations and reviewed facts.

Offline and deterministic. Original localization JSON files are NOT copied into
the generated dataset. Source-linked paraphrases retain their provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MARKUP = re.compile(r"<[^<>]{0,200}>")
SPACE = re.compile(r"\s+")
CATEGORIES = ("main", "buildings", "moodles")


def clean(text, max_len=360):
    value = str(text).replace("\\n", " ").replace("\n", " ")
    value = MARKUP.sub(" ", value).replace("█", "")
    value = SPACE.sub(" ", value).strip().strip(" -:;")
    if not value or value in {"???", "-", "<>"}:
        return ""
    if len(value) > max_len:
        head = value[:max_len]
        separators = [head.rfind(p) for p in (". ", "! ", "? ", "다. ", "요. ")]
        stop = max(separators)
        value = head[:stop + 1] if stop >= 100 else head.rstrip(" ,;")
    return value.strip()


def row(group, item, language, question, answer, source, confidence="source"):
    return {
        "group": group,
        "item": item,
        "language": language,
        "user": question,
        "assistant": answer,
        "source": source,
        "confidence": confidence,
    }


def build(en, ko, facts):
    dataset = []
    summary = Counter()
    for section in CATEGORIES:
        e, k = en.get(section, {}), ko.get(section, {})
        for key in sorted(set(e) | set(k)):
            if key.endswith("dsc") or key.endswith("dsc2") or key.endswith("dscfixed"):
                continue
            key_desc = key + "dsc"
            for language, src in (("en", e), ("ko", k)):
                if key not in src or key_desc not in src:
                    continue
                label = clean(src[key], 100)
                description = clean(src[key_desc])
                if not label or not description or len(description) < 10:
                    continue
                group = f"cu:{section}:{key}"
                if language == "en":
                    answer = f"In Casualties: Unknown, {label}: {description}"
                    questions = [
                        f"What is {label} in Casualties: Unknown?",
                        f"How is {label} described in C:U?",
                        f"Tell me about {label} in the game Casualties: Unknown.",
                    ]
                else:
                    answer = f"Casualties: Unknown에서 {label}: {description}"
                    questions = [
                        f"Casualties: Unknown에서 {label}은 뭐야?",
                        f"C:U에 나오는 {label} 설명해줘.",
                        f"게임 속 {label}은 어떤 물건이나 상태야?",
                    ]
                for question in questions:
                    dataset.append(row(group, key, language, question, answer, f"user-upload:{language}:{section}.{key_desc}"))
                summary[f"cu_desc_{language}"] += 1
            if key in e and key in k:
                en_name = clean(e[key], 100)
                ko_name = clean(k[key], 100)
                if en_name and ko_name and en_name.lower() != ko_name.lower():
                    group = f"cu:{section}:{key}"
                    dataset.append(row(group, key, "ko", f"C:U에서 {en_name}을 한국어로 뭐라고 해?", f"제공된 한국어 번역에서는 {en_name}을(를) '{ko_name}'이라고 표기해.", f"user-upload:aligned:{section}.{key}"))
                    dataset.append(row(group, key, "en", f"What is the Korean localization of {en_name} in C:U?", f"The supplied Korean localization calls {en_name} '{ko_name}'.", f"user-upload:aligned:{section}.{key}"))
                    summary["aligned_names"] += 1

    other_en, other_ko = en.get("other", {}), ko.get("other", {})
    for key in sorted(set(other_en) | set(other_ko)):
        if not (re.fullmatch(r"layertitle(?:tutorial|\d+)", key) or re.fullmatch(r"layermodifier\d+", key)):
            continue
        group = f"cu:other:{key}"
        for language, source in (("en", other_en), ("ko", other_ko)):
            if key not in source:
                continue
            title = clean(source[key], 150)
            if not title:
                continue
            if key.startswith("layertitle"):
                layer = key[10:]
                question = f"What is the name of layer {layer} in C:U?" if language == "en" else f"C:U의 {layer}층 이름이 뭐야?"
                answer = f"The English localization names this layer '{title}'." if language == "en" else f"제공된 한국어 번역에서 이 층은 '{title}'이야."
            else:
                question = f"What is the C:U layer modifier '{title}'?" if language == "en" else f"C:U의 층 환경 효과 '{title}'은 뭐야?"
                description = clean(source.get(key + "dsc", ""), 260)
                answer = f"'{title}' is a layer modifier in Casualties: Unknown. {description}".strip() if language == "en" else f"'{title}'은(는) Casualties: Unknown의 층 환경 효과야. {description}".strip()
            dataset.append(row(group, key, language, question, answer, f"user-upload:{language}:other.{key}"))
            summary["layer_rows"] += 1

    for entry in facts:
        if not all(entry.get(k) for k in ("game", "id", "q_en", "a_en", "q_ko", "a_ko", "source")):
            raise ValueError(f"invalid game fact: {entry!r}")
        group = f"game:{entry['game']}:{entry['id']}"
        for language in ("en", "ko"):
            dataset.append(row(group, entry["id"], language, entry["q_" + language], entry["a_" + language], entry["source"], "reviewed"))
            summary["reviewed_game_rows"] += 1

    # Group split: same game fact, localization entry and translation ALWAYS go
    # into one partition, so language cross-over and paraphrases cannot leak.
    seen = set()
    unique = []
    for item in dataset:
        signature = (item["language"], clean(item["user"]).casefold(), clean(item["assistant"]).casefold())
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    train, heldout = [], []
    for item in unique:
        bucket = int(hashlib.sha256(item["group"].encode("utf-8")).hexdigest()[:8], 16) % 100
        (heldout if bucket < 8 else train).append(item)
    train_groups = {r["group"] for r in train}
    holdout_groups = {r["group"] for r in heldout}
    if train_groups & holdout_groups:
        raise RuntimeError("source-group leakage detected")
    return train, heldout, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", required=True, help="Source EN locale (user-provided)")
    parser.add_argument("--ko", required=True, help="Source Korean locale (user-provided)")
    parser.add_argument("--facts", default="game_facts_v13.json")
    parser.add_argument("--output", default="v13_game_data")
    args = parser.parse_args()

    inputs = [Path(args.en), Path(args.ko), Path(args.facts)]
    en, ko, facts = [json.loads(p.read_text(encoding="utf-8-sig")) for p in inputs]
    train, heldout, detail = build(en, ko, facts)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("game_qa_v13_train.jsonl", train), ("game_qa_v13_holdout.jsonl", heldout)):
        (output / filename).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
            encoding="utf-8",
        )
    report = {
        "dataset": "HGAIM-v13-game-qa",
        "english_source_version": en.get("version", "unspecified"),
        "korean_source_version": ko.get("version", "unspecified"),
        "localization_source_is_game_data_not_runtime_verified": True,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "train_qa": len(train),
        "heldout_qa": len(heldout),
        "train_groups": len({r["group"] for r in train}),
        "heldout_groups": len({r["group"] for r in heldout}),
        "by_language_train": dict(Counter(r["language"] for r in train)),
        "by_language_heldout": dict(Counter(r["language"] for r in heldout)),
        "game_counts_train": dict(Counter(r["group"].split(":")[1] for r in train if r["group"].startswith("game:"))),
        "detail": dict(detail),
        "warning": "Three source-linked prompts are not three independent facts. Novel phrasing and factual accuracy need human validation. Korean localization differs from the English source.",
    }
    (output / "REPORT_v13.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
