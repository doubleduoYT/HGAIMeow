#!/usr/bin/env python3
"""Optional OFFLINE teacher-reference tool.

It never becomes part of HGAI runtime and it never auto-merges teacher output
into the training set.  The generated JSONL is a review queue only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompts",required=True,help="UTF-8 text file, one question per line")
    ap.add_argument("--output",default="teacher_suggestions.jsonl")
    ap.add_argument("--limit",type=int,default=200)
    args=ap.parse_args()
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except Exception as e:
        raise SystemExit("teacher reference requires: pip install transformers torch\n"+str(e))
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype="auto",device_map="auto")
    prompts=[x.strip() for x in Path(args.prompts).read_text(encoding="utf-8").splitlines() if x.strip()][:args.limit]
    with Path(args.output).open("w",encoding="utf-8") as f:
        for q in prompts:
            messages=[{"role":"system","content":"한국어로 정확하고 짧게 답해. 사실을 모르면 모른다고 해."},{"role":"user","content":q}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inp=tok(text,return_tensors="pt").to(model.device)
            out=model.generate(**inp,max_new_tokens=160,do_sample=False)
            answer=tok.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
            f.write(json.dumps({"question":q,"teacher_model":args.model,"suggestion":answer,"reviewed":False},ensure_ascii=False)+"\n")
            print(q,"=>",answer[:120])
    print("Teacher suggestions are review-only; HGAI runtime never loads the teacher model.")

if __name__=="__main__": main()
