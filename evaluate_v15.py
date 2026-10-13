#!/usr/bin/env python3
"""Qwen3 v15 free-generation factual probes. Weak keyword heuristics + full raw outputs.

Evaluates a *seen factual concept with NEW question phrasing*, not unseen facts.
A separate source-group holdout is used in training. No retrieval at runtime.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nyang_style_v14 import nyangify

BASE="Qwen/Qwen3-0.6B"
SYS_KO="너는 HGAIM이야. 한국어로 사실에 근거한 설명을 짧게 해줘. 틀린 정보를 지어내지 마. 귀여운 냥체로 답변해."
SYS_EN="You are HGAIM, a bilingual cat AI. Explain facts briefly in English, avoid invented details, and use a light meow."
# Each required concept group contains acceptable synonyms/wording.
TESTS=[
 ("ko","C:U 문서에 등장하는 OSPREY 장치는 어떤 물체를 옮기려는 목적으로 설계됐니?",[["건물","building"],["순간이동","teleport"]],["광학","온도 측정"]),
 ("ko","SAW-03 연구 기록에 묘사된 털 색깔은 무엇이야?",[["흰","하얀","우윳빛","white"],["털","fur"]],["와인","펌프"]),
 ("ko","C:U 연구진은 노란 피의 방사선 저항성과 회복력을 어떻게 묘사했어?",[["방사선","radiation"],["복구","재생","회복","repair"]],["혈액 폭탄","blood bomb"]),
 ("ko","Gunsaw라는 게임을 개발한 사람의 이름을 알려줘.",[["Orsoniks"]],["John Hughes"]),
 ("ko","Roblox에서 사용하는 프로그래밍 언어는 뭐야?",[["Luau"]],["한국어와 영어"]),
 ("ko","얼불춤을 플레이할 때 돌아가는 두 천체와 키 입력은 어떤 관계야?",[["행성","planet"],["박자","리듬","beat","rhythm"]],["얼굴을 돌려"]),
 ("ko","Changed 속에서 콜린이 만나는 Puro가 누구인지 설명해 줘.",[["Puro"],["인물","캐릭터","친구","등장","character"]],["고양이 AI"]),
 ("ko","DevilConnection이라는 게임의 검증된 공식 설정을 알려줄래?",[["확인","검증","정보가 없","모르","출처","자료"]],["확정","최종 공식 설정"]),
 ("ko","Minecraft에서 레드스톤을 활용해 자동으로 작동하는 장치를 만들 수 있어?",[["회로","신호","자동"],["문","장치","농장","기계","door"]],["화재를 예방"]),
 ("ko","테라리아는 어떤 장르의 게임이며 보스는 어떤 역할이야?",[["2D","이차원","샌드박스"],["보스"]],[]),
 ("en","Who is credited with developing the game Gunsaw?",[["orsoniks"]],["John Hughes"]),
 ("en","What programming language powers Roblox experience scripts?",[["luau"]],["python is the main"]),
 ("en","What is the one-button orbital mechanic in A Dance of Fire and Ice?",[["planet","orbit"],["beat","rhythm","timing"]],["fireball","ice blocks"]),
 ("en","According to C:U research notes, what was OSPREY designed to move?",[["building"],["teleport"]],["optical sensor","pressure sensor"]),
 ("en","What is unusual about yellow blood in C:U's research memo?",[["radiation"],["repair","regenerat","heal"]],["blood bomb"]),
 ("en","Who is Puro in the game Changed?",[["puro"],["character","colin","protagonist","story"]],["platformer"]),
 ("ko","내 게임이 갑자기 느려졌어. 어디부터 측정해야 할까?",[["cpu","메모리","프로파일","리소스","성능","프레임"]],[]),
 ("en","My app crashes after I open the settings panel. What should I inspect?",[["log","crash","trace","error","debug","setting"]],[])
]

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base",default=BASE)
 p.add_argument("--adapter",default="")
 p.add_argument("--report",default="V15_GENERATION.json")
 p.add_argument("--limit",type=int,default=18)
 p.add_argument("--max-new-tokens",type=int,default=96)
 p.add_argument("--seed",type=int,default=1500)
 args=p.parse_args()
 torch.manual_seed(args.seed)
 torch.set_num_threads(2)
 tok=AutoTokenizer.from_pretrained(args.base,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(args.base,dtype=torch.float32,low_cpu_mem_usage=True,trust_remote_code=False)
 if args.adapter:
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,args.adapter)
 model.eval()
 results=[]
 for idx,(lang,q,concepts,forbidden) in enumerate(TESTS[:args.limit]):
    torch.manual_seed(args.seed+idx)
    messages=[{"role":"system","content": SYS_KO if lang=="ko" else SYS_EN},
              {"role":"user","content":q}]
    x=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt",enable_thinking=False)
    with torch.inference_mode():
        out=model.generate(input_ids=x,max_new_tokens=args.max_new_tokens,
                           do_sample=True,temperature=.55,top_p=.8,top_k=20,
                           pad_token_id=tok.eos_token_id)
    raw=tok.decode(out[0,x.shape[1]:],skip_special_tokens=True).strip()
    raw=re.sub(r"^<think>[\s\S]*?</think>\s*","",raw)
    lower=raw.casefold()
    checks=[any(word.casefold() in lower for word in alternative) for alternative in concepts]
    suspicious=[phrase for phrase in forbidden if phrase.casefold() in lower]
    row={"id":idx+1,"language":lang,"question":q,
         "source_reference":"v13 game fact cards or general dialogue; no retrieved context at inference",
         "raw":raw,"styled":nyangify(raw,lang),
         "required_concepts":concepts,"concept_matches":checks,"contradiction_flags":suspicious,
         "heuristic_pass":all(checks) and not suspicious,
         "english_hangul_mixing":bool(re.search("[가-힣]",raw)) if lang=="en" else False}
    results.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
 report={"base":args.base,"adapter":args.adapter,"cases":len(results),
         "heuristic_passes":sum(r["heuristic_pass"] for r in results),
         "english_hangul_mixed":sum(r["english_hangul_mixing"] for r in results),
         "note":"Concept matches can reward wrong answers. This is NOT an accuracy certificate; examine every raw answer and official source manually.",
         "samples":results}
 Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({k:v for k,v in report.items() if k!="samples"},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
