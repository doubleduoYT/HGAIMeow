#!/usr/bin/env python3
"""v15 bilingual factual-curriculum data: explicit provenance, no leakage of held-out fact groups."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from nyang_style_v14 import nyangify
from prepare_sft_v14 import SYS_EN, SYS_KO

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def chat(lang, question, answer, group, kind):
    return {"language": lang, "kind": kind, "source_group": group,
            "messages": [{"role":"system", "content": SYS_KO if lang=="ko" else SYS_EN},
                         {"role":"user", "content": question.strip()},
                         {"role":"assistant", "content": nyangify(answer.strip(), lang)}]}

def build(args):
    random.seed(args.seed)
    rng=random.Random(args.seed)
    train_v14=read_jsonl(args.v14_train)
    game_train=read_jsonl(args.game_train)
    game_holdout=read_jsonl(args.game_holdout)
    facts=json.loads(Path(args.facts).read_text(encoding="utf-8"))
    train_groups={r["group"] for r in game_train}
    heldout_groups={r["group"] for r in game_holdout}
    assert train_groups and heldout_groups and train_groups.isdisjoint(heldout_groups)
    new=[]
    for f in facts:
        group=f"game:{f['game']}:{f['id']}"
        if group not in train_groups:
            continue
        for lang in ("ko","en"):
            a=f["a_"+lang].strip()
            q=f["q_"+lang].strip()
            new.append(chat(lang,q,a,group,"verified_fact"))
            # A second formulation builds a new question from the same source fact.
            # It is intentionally not counted as independent knowledge.
            title=f["game"]
            if lang=="ko":
                q2=f"{title} 관련해서 '{q}'라는 질문을 받으면 어떻게 답할래?"
            else:
                q2=f"Answer this question about {title} clearly: {q}"
            new.append(chat(lang,q2,a,group,"fact_paraphrase"))
    seen={(x["source_group"],x["language"],x["messages"][-2]["content"].casefold())
          for x in train_v14 if x["messages"][-1]["role"]=="assistant"}
    train=list(train_v14)
    for item in new:
        key=(item["source_group"],item["language"],item["messages"][-2]["content"].casefold())
        if key not in seen:
            train.append(item);seen.add(key)
    assert len(train)>1100 and len(new)>250, (len(train),len(new))
    assert {x["source_group"] for x in train if x["kind"] in ("game","verified_fact","fact_paraphrase")}.isdisjoint(heldout_groups)
    # Dialogue groups are distinct; v13 source IDs get inherited.
    rng.shuffle(train)
    heldout=read_jsonl(args.v14_holdout)
    # Preserve only complete supervised validation records. Independent prompts
    # are a free-generation benchmark, not an assistant answer to score token loss.
    supervised=[x for x in heldout if x.get("kind")!="independent_generation"]
    assert supervised and all(x["source_group"] not in {r["source_group"] for r in train} for x in supervised)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for name, rows in (("train.jsonl",train),("holdout.jsonl",supervised)):
        (out/name).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    report={
        "version":"HGAIM-v15-Qwen3",
        "base_model":"Qwen/Qwen3-0.6B",
        "train_records":len(train), "holdout_records":len(supervised),
        "additional_fact_variants":len(new),
        "distinct_source_fact_groups":len({x["source_group"] for x in new}),
        "language_counts":dict(Counter(x["language"] for x in train)),
        "kind_counts":dict(Counter(x["kind"] for x in train)),
        "fact_group_leakage":False,
        "source_sha256":{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (args.v14_train,args.v14_holdout,args.game_train,args.game_holdout,args.facts)},
        "note":"Factual Q/A provenance from v13 user-provided C:U strings and attributed curated game cards; repeated prompts do not add independent facts. This is not evidence of generated accuracy."
    }
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v14-train",default="v14_data/train_v14.jsonl")
    p.add_argument("--v14-holdout",default="v14_data/heldout_v14.jsonl")
    p.add_argument("--game-train",default="v13_game_data/game_qa_v13_train.jsonl")
    p.add_argument("--game-holdout",default="v13_game_data/game_qa_v13_holdout.jsonl")
    p.add_argument("--facts",default="game_facts_v13.json")
    p.add_argument("--output",default="v15_data")
    p.add_argument("--seed",type=int,default=20261013)
    build(p.parse_args())

if __name__=="__main__": main()
