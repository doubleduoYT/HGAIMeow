#!/usr/bin/env python3
"""Explicit v16 model-promotion gate: unlike v15, lower loss is NOT sufficient."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def assess(baseline: dict, candidate: dict) -> dict:
    cases_b=baseline.get("samples",[])
    cases_c=candidate.get("samples",[])
    pairs_match=(len(cases_b)>=10 and len(cases_b)==len(cases_c) and
                 all(b.get("question")==c.get("question") for b,c in zip(cases_b,cases_c)))
    b_hits=sum(x.get("heuristic_pass") is True for x in cases_b)
    c_hits=sum(x.get("heuristic_pass") is True for x in cases_c)
    contradictions=sum(bool(x.get("contradiction_flags")) for x in cases_c)
    leakage=sum(bool(x.get("english_hangul_mixing")) for x in cases_c)
    # Positive automation approval REQUIRES separate human review in addition
    # to these mechanical checks; no unattended deployment/promotion.
    better=(pairs_match and c_hits>b_hits and c_hits>=max(8,len(cases_c)*.7)
            and contradictions==0 and leakage==0)
    return {
        "questions_match":pairs_match,"baseline_heuristic_hits":b_hits,
        "candidate_heuristic_hits":c_hits,"candidate_contradiction_flags":contradictions,
        "candidate_english_hangul_mixing":leakage,
        "automatic_promote":False,
        "ready_for_human_review":bool(better),
        "reason":("generation regression or insufficient objective improvement"
                  if not better else "mechanical checks improved, but human factual review remains required"),
        "note":"Game keyword matches are only rough diagnostics, NOT factual correctness."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",default="V15_BASE_GENERATION.json")
    p.add_argument("--candidate",default="V15_TUNED_GENERATION.json")
    p.add_argument("--output",default="V16_QUALITY_GATE.json")
    args=p.parse_args()
    report=assess(json.loads(Path(args.baseline).read_text(encoding="utf-8")),
                  json.loads(Path(args.candidate).read_text(encoding="utf-8")))
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    # Reject nonsensical promotion. An expected failed v15 candidate is a
    # successful test OF the quality gate, not a pipeline failure.
if __name__=="__main__":main()
