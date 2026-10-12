#!/usr/bin/env python3
"""Independent bilingual evaluation of HGAI. This never modifies the model."""
import argparse
import json
import re
from pathlib import Path

from hgai_core import HGAIEngine

TESTS = [
    ("What is HGAI?", [["hgai"], ["ai", "transformer", "search"]]),
    ("What is Python?", [["python"], ["programming", "language"]]),
    ("Could you explain JSON?", [["json"], ["data", "format"]]),
    ("Tell me about a GPU.", [["gpu"], ["parallel", "graphics"]]),
    ("What is the difference between TCP and UDP?", [["tcp"], ["udp"]]),
    ("What is GitHub?", [["github"], ["git", "repositories", "code"]]),
    ("Could you explain what RAG is?", [["rag", "retrieval"], ["retriev", "search", "information"]]),
    ("Explain what an APK is.", [["apk"], ["android"]]),
    ("Tell me about Linux.", [["linux"], ["kernel", "operating"]]),
    ("What is a Transformer?", [["transformer"], ["attention"]]),
    ("How would you describe machine learning?", [["machine learning"], ["data", "pattern"]]),
    ("What does DNS do?", [["dns"], ["domain", "ip"]]),
    ("Tell me about game mods.", [["mod"], ["game"]]),
    ("What is a memory leak?", [["memory"], ["leak", "release"]]),
    ("What is a neural network?", [["neural network"], ["layer", "weight", "learn"]]),
]

def check(answer, groups):
    a = re.sub(r"\s+", " ", answer.lower())
    return all(any(k in a for k in choices) for choices in groups) and bool(re.search(r"[A-Za-z]",a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default="hgai_model_v10.pth")
    ap.add_argument("--mode", choices=["search", "hybrid", "neural", "raw-neural"], default="hybrid")
    ap.add_argument("--min-pass", type=int, default=0)
    ap.add_argument("--report", default="")
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()
    engine = HGAIEngine("train.txt", args.model_file, "main", seed=args.seed, load_model=(args.mode != "search"))
    import torch
    rows = []
    passed = 0
    for i,(q,groups) in enumerate(TESTS,1):
        engine.history=[]
        torch.manual_seed(args.seed+i)
        a=engine.reply(q,mode=args.mode,temperature=.58,top_k=20,top_p=.88,max_new_tokens=80)
        good=check(a,groups)
        passed+=bool(good)
        rows.append({"index":i,"question":q,"answer":a,"passed":bool(good)})
        print(("PASS" if good else "FAIL"),i,q,"=>",a)
    result={"mode":args.mode,"model_file":args.model_file,"passed":passed,"total":len(TESTS),"min_pass":args.min_pass,"tests":rows}
    print(f"English {args.mode}: {passed}/{len(TESTS)}")
    if args.report:
        Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if passed>=args.min_pass else 1

if __name__=="__main__":
    raise SystemExit(main())
