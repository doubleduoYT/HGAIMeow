#!/usr/bin/env python3
"""Local-only Qwen3 v15 client with explicit /no_think and inherited nyang speech.

Requires a manually merged+quantized Qwen3 GGUF served by compatible llama.cpp.
Does NOT modify existing Pi bot/socket protocols, model files, or systemd units.
"""
from __future__ import annotations
import argparse
from hgaim_v14_client import query_local

def ask(question, history=None, *, url="http://127.0.0.1:8086", max_tokens=112):
    q=question.strip()
    if not q:raise ValueError("A question is required")
    # Qwen3 soft-switch remains useful if a llama.cpp build does not accept
    # chat_template_kwargs. This is an instruction, not retrieval.
    if "/no_think" not in q:
        q+="\n/no_think"
    return query_local(q,history=history,url=url,model="local",max_tokens=max_tokens)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--question",required=True)
    p.add_argument("--url",default="http://127.0.0.1:8086")
    p.add_argument("--max-tokens",type=int,default=112)
    args=p.parse_args()
    print(ask(args.question,url=args.url,max_tokens=args.max_tokens))

if __name__=="__main__":main()
