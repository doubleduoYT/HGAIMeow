#!/usr/bin/env python3
"""New bilingual game knowledge questions never used verbatim for this LoRA stage.

Keyword checks are only a weak diagnostic: preserve per-prompt generations for
manual semantic/factual review; do not promote based on keyword scores.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nyang_style_v14 import nyangify

BASE="Qwen/Qwen2.5-0.5B-Instruct"
TESTS=[
 ("ko","C:U 연구 기록에서 OSPREY는 무엇을 운송하려고 했어?",["건물","순간이동"]),
 ("ko","C:U 설정의 SAW-03에 대한 기록에는 어떤 외형 묘사가 있어?",["흰","털"]),
 ("ko","C:U의 노란 혈액은 연구 메모에서 어떤 특징을 보였어?",["재생","방사선"]),
 ("ko","C:U 속 SAW 프로젝트는 무엇에 관련된 연구야?",["연구","생명"]),
 ("ko","Gunsaw를 만든 사람은 누구야?",["orsoniks"]),
 ("ko","마인크래프트에서 레드스톤 신호는 어디에 활용할 수 있지?",["회로","자동"]),
 ("ko","얼불춤에서는 어떤 방식으로 박자를 맞춰?",["행성","박자"]),
 ("ko","로블록스 게임 스크립트는 주로 무슨 언어로 만들지?",["luau"]),
 ("ko","Changed 이야기의 Puro는 어떤 인물이야?",["puro"]),
 ("ko","DevilConnection의 세계관을 사실대로 설명해 줘.",[]),
 ("en","What is the experimental OSPREY device described in C:U documents?",["building","teleport"]),
 ("en","What makes the yellow blood special in the C:U research notes?",["radiation","heal"]),
 ("en","What language is used to script Roblox experiences?",["luau"]),
 ("en","What is the core timing mechanic in ADOFAI?",["orbit","beat"]),
 ("en","Who created Gunsaw?",["orsoniks"]),
 ("en","What is Puro's role in Changed?",["puro"]),
]
SYS_KO="너는 HGAIM 고양이 AI야. 정확하게 한국어로 짧게 설명하고 문장 끝은 냥체로 해. 근거 없는 C:U 설정은 지어내지 마."
SYS_EN="You are HGAIM, a friendly game expert cat AI. Answer factual questions briefly in English, with occasional meow. Never fabricate lore."

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base",default=BASE)
 p.add_argument("--adapter",default="")
 p.add_argument("--out",default="V14_1_GAMES.json")
 p.add_argument("--limit",type=int,default=16)
 p.add_argument("--max-tokens",type=int,default=100)
 args=p.parse_args()
 torch.set_num_threads(2)
 tok=AutoTokenizer.from_pretrained(args.base)
 model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
 if args.adapter:model=PeftModel.from_pretrained(model,args.adapter).eval()
 rows=[]
 for lang,question,keywords in TESTS[:args.limit]:
  prompt=tok.apply_chat_template(
    [{"role":"system","content":SYS_KO if lang=="ko" else SYS_EN},{"role":"user","content":question}],
    add_generation_prompt=True,return_tensors="pt")
  with torch.inference_mode():
   result=model.generate(input_ids=prompt,max_new_tokens=args.max_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
  raw=tok.decode(result[0,prompt.shape[-1]:],skip_special_tokens=True).strip()
  styled=nyangify(raw,lang)
  hits=sum(int(any(t in raw.lower() for t in w.split("|"))) for w in keywords)
  row={"language":lang,"question":question,"raw":raw,"styled":styled,"expected_keywords":keywords,
       "keyword_hits":hits,"keyword_total":len(keywords)}
  rows.append(row)
  print(json.dumps(row,ensure_ascii=False),flush=True)
 payload={"base":args.base,"adapter":args.adapter,"tests":rows,
          "note":"Keyword hits are not accuracy; use human factual review against source-linked C:U notes."}
 Path(args.out).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
