#!/usr/bin/env python3
"""Independent free-generation v14 benchmark, no retrieval and no keyword-only gate."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nyang_style_v14 import nyangify

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
TESTS = [
    ("ko", "C:U의 SAW 프로젝트는 뭐야?"),
    ("ko", "Minecraft 레드스톤은 어떤 데 사용해?"),
    ("ko", "Gunsaw와 Casualties: Unknown의 공통점은?"),
    ("ko", "게임에서 프레임이 갑자기 떨어져. 뭘 확인해야 해?"),
    ("en", "Explain the core mechanics of A Dance of Fire and Ice."),
    ("en", "What can you tell me about Changed and Puro?"),
    ("en", "I'm making a platform game. How do I test jump physics?"),
    ("en", "Tell me how to begin learning Roblox Studio."),
]
SYS_KO = ("너는 HGAIM 고양이 AI야. 한국어로 정확하고 자연스럽게 짧게 대답해. "
          "각 문장은 냥체로 말해. 정보가 부족하면 정직하게 설명해. 검색은 하지 마.")
SYS_EN = ("You are HGAIM, a clear and friendly cat-themed AI. Answer in English, "
          "be accurate, and use an occasional meow. No search.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--adapter", default="")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--report", default="v14_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=75)
    parser.add_argument("--limit", type=int, default=8)
    args=parser.parse_args()
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(args.base)
    model=AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter).eval()
    rows=[]
    for i,(lang,q) in enumerate(TESTS[:args.limit]):
        system=SYS_KO if lang=="ko" else SYS_EN
        ids=tok.apply_chat_template([{"role":"system","content":system},
                                     {"role":"user","content":q}],
                                    add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out=model.generate(input_ids=ids,max_new_tokens=args.max_new_tokens,do_sample=False,
                               pad_token_id=tok.eos_token_id)
        raw=tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
        result=nyangify(raw,lang)
        row={"language":lang,"prompt":q,"raw":raw,"styled":result,
             "has_hangul_in_english": bool(re.search("[가-힣]",result)) if lang=="en" else False,
             "last_sentence_cat_style": ("냥" in result[-13:] if lang=="ko" else "meow" in result[-16:].lower())}
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={"base":args.base,"adapter":args.adapter,"responses":rows,
             "note":"Raw+styled examples support human review. No automated grammar/factuality gate is claimed."}
    Path(args.report).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
