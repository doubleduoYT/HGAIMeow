#!/usr/bin/env python3
"""HGAIM v16 local hybrid client: sourced facts first, pretrained chat otherwise.

No internet, third-party vector DB, paid model endpoint or live service edits.
The optional Qwen local-only endpoint is used only when a source fact is
not applicable. Unsupported game lore is explicitly marked unverified.
"""
from __future__ import annotations
import argparse
import json

from hgaim_v16_grounded import grounded_answer
from hgaim_v15_client import ask as ask_neural

def answer(question, history=None, *, facts_path="game_facts_v13.json",
           url="http://127.0.0.1:8086", unknown_policy="abstain",
           max_tokens=112, metadata=False):
    if not question or not question.strip():
        raise ValueError("Question is required")
    grounded=grounded_answer(question,facts_path=facts_path,unknown_policy=unknown_policy)
    if grounded is not None:
        return grounded if metadata else grounded["answer"]
    text=ask_neural(question,history=history,url=url,max_tokens=max_tokens)
    result={"route":"neural","source":None,"language":"ko" if any("가"<=c<="힣" for c in question) else "en","answer":text}
    return result if metadata else result["answer"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--question",required=True)
    p.add_argument("--facts",default="game_facts_v13.json")
    p.add_argument("--url",default="http://127.0.0.1:8086")
    p.add_argument("--unknown-policy",choices=["abstain","neural"],default="abstain")
    p.add_argument("--metadata",action="store_true")
    p.add_argument("--max-tokens",type=int,default=112)
    args=p.parse_args()
    result=answer(args.question,facts_path=args.facts,url=args.url,
                  unknown_policy=args.unknown_policy,
                  max_tokens=args.max_tokens,metadata=args.metadata)
    print(json.dumps(result,ensure_ascii=False,indent=2) if args.metadata else result)

if __name__=="__main__":main()
