#!/usr/bin/env python3
"""HGAIM v16 local hybrid client: sourced facts first, pretrained chat otherwise.

No internet, third-party vector DB, paid model endpoint or live service edits.
The optional Qwen local-only endpoint is used only when a source fact is
not applicable. Unsupported game lore is explicitly marked unverified.
"""
from __future__ import annotations
import argparse
import json
import re

from hgaim_v16_grounded import grounded_answer, detect_games
from hgaim_v15_client import ask as ask_neural

FOLLOWUP = re.compile(
    r"그\\s*(?:게임|사람|장비|개체|설정|건|거)|그럼|그리고|그러면|누가|개발자|어떤\\s*언어|무슨\\s*언어|"
    r"털\\s*색|색깔|뭘\\s*옮|누구야|\\bwhat about\\b|\\bwho (?:made|developed|is)\\b|\\band what\\b|\\bwhat language\\b",
    re.I,
)

def resolve_followup(question, history):
    """Add ONLY a uniquely known recent game's name for an explicit follow-up.

    Never infer a subject from unrelated general conversation or from a
    previous message that names multiple different games.
    """
    if detect_games(question) or not history or not FOLLOWUP.search(question):
        return question
    prior = ""
    for message in reversed(history[-8:]):
        if message.get("role")=="user":
            candidate=str(message.get("content","")).strip()
            if candidate and candidate!=question:
                prior=candidate
                break
    games=detect_games(prior)
    if len(games)!=1:
        return question
    game=games[0]
    context_suffix=""
    if game=="Casualties: Unknown":
        entity=re.search(r"\\bosprey\\b|\\bsaw[-_ ]?0?3\\b|노란\\s*(?:피|혈액)|yellow\\s+blood",prior,re.I)
        if entity and re.search(r"그\\s*(?:장비|개체|피|혈액)|\\bit\\b|\\bthat\\b",question,re.I):
            context_suffix=" "+entity.group(0)
    return game + context_suffix + " " + question

def answer(question, history=None, *, facts_path="game_facts_v13.json",
           url="http://127.0.0.1:8086", unknown_policy="abstain",
           max_tokens=112, metadata=False):
    if not question or not question.strip():
        raise ValueError("Question is required")
    resolved = resolve_followup(question, history)
    grounded=grounded_answer(resolved,facts_path=facts_path,unknown_policy=unknown_policy)
    if grounded is not None:
        if resolved != question:
            grounded = {**grounded, "context_resolved": True}
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
