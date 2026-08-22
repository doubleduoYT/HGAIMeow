#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import torch
from hgai_core import HGAIEngine

TESTS = [
    ("HGAI가 뭐야?", [["hgai"], ["ai", "고양이"]]),
    ("Casualties Unknown은 어떤 게임이야?", [["게임"], ["생존", "하드코어"]]),
    ("GitHub가 뭐야?", [["github", "git"], ["코드", "저장소", "협업"]]),
    ("Python이 뭐야?", [["python", "파이썬"], ["언어", "프로그래밍"]]),
    ("TCP와 UDP 차이 알려줘", [["tcp"], ["udp"]]),
    ("GPU는 뭐 하는 장치야?", [["gpu"], ["그래픽", "병렬", "연산"]]),
    ("JSON 설명해줘", [["json"], ["데이터", "형식"]]),
    ("RAG가 뭐야?", [["rag"], ["검색", "자료", "외부"]]),
    ("APK가 뭐야?", [["apk"], ["안드로이드", "설치", "패키지"]]),
    ("양자화가 뭐야?", [["양자화"], ["비트", "메모리", "연산"]]),
]

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def semantic_pass(answer, groups):
    a=norm(answer)
    if "순수 신경망 생성이 아직 불안정" in a:
        return False
    return all(any(term.lower() in a for term in group) for group in groups)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-file", default="hgai_model_v10.pth")
    ap.add_argument("--preset", default="main")
    ap.add_argument("--min-pass", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--report", default="", help="optional JSON report path")
    args=ap.parse_args()
    e=HGAIEngine("train.txt", args.model_file, args.preset, seed=args.seed, load_model=True)
    if e.model is None:
        raise SystemExit("model not loaded")
    passed=0
    rows=[]
    for i,(q,groups) in enumerate(TESTS,1):
        torch.manual_seed(args.seed+i)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed+i)
        # raw-neural is intentionally required here: retrieval/curated facts must not answer.
        a=e.reply(q, mode="raw-neural", temperature=.58, top_k=20, top_p=.88, max_new_tokens=80)
        good=semantic_pass(a,groups)
        passed+=good
        rows.append({"index":i,"question":q,"answer":a,"passed":bool(good),"required_groups":groups})
        print(("OK" if good else "FAIL"), i, q, "=>", a)
    report={
        "preset":args.preset,
        "model_file":args.model_file,
        "seed":args.seed,
        "passed":passed,
        "total":len(TESTS),
        "required":args.min_pass,
        "gate_passed":passed >= args.min_pass,
        "tests":rows,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("report:",args.report)
    print(f"raw-neural semantic eval {passed}/{len(TESTS)}; required={args.min_pass}")
    raise SystemExit(0 if passed >= args.min_pass else 1)

if __name__ == "__main__":
    main()
