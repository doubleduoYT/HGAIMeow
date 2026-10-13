#!/usr/bin/env python3
"""Independent, retrieval-free dialogue evaluation. Metrics are diagnostics, not a semantic quality certificate."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
from hgai_core import HGAIEngine, FALLBACK_RESPONSE, FALLBACK_RESPONSE_EN

TESTS=[
 ("한국어","오늘 좀 심심한데 뭐 할까?", "그중에서 음악 관련 활동이 좋아"),
 ("한국어","만든 앱이 실행되지 않아. 어디서부터 볼까?", "오류 로그가 없으면 어떻게 해?"),
 ("한국어","새 프로젝트를 시작할까 고민 중이야", "너무 오래 걸릴까 봐 걱정돼"),
 ("한국어","게임 화면이 자꾸 흔들려", "카메라 설정 문제일 수도 있어?"),
 ("한국어","짧고 재미있는 이야기를 들려줘", "이번에는 다른 분위기로 해줘"),
 ("한국어","잘못 알고 있는 것 같아", "다시 확인해 줄래?"),
 ("English","I've been working on a level editor", "What feature should I test first?"),
 ("English","I want to improve my English writing", "Can you make that advice shorter?"),
 ("English","My computer keeps freezing", "How can I narrow down the cause?"),
 ("English","Tell me something fun to do indoors", "I'd rather make something creative"),
 ("English","I don't agree with that suggestion", "Could you offer another option?"),
 ("English","I'm planning a small community project", "What is a sensible first milestone?"),
]

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--model-file",default="hgai_model_v11.pth")
 parser.add_argument("--report",default="V11_DIALOGUE_EVAL.json")
 parser.add_argument("--seed",type=int,default=20261013)
 args=parser.parse_args()
 torch.set_num_threads(2)
 engine=HGAIEngine("train.txt",args.model_file,"main",seed=args.seed,load_model=True)
 if engine.model is None:
  raise SystemExit("Cannot evaluate: checkpoint did not load")
 rows=[]
 for i,(lang,first,followup) in enumerate(TESTS):
  engine.history=[]
  answers=[]
  for j,q in enumerate((first,followup)):
   torch.manual_seed(args.seed+i*10+j)
   a=engine.reply(q,mode="raw-neural",temperature=.58,top_k=20,top_p=.88,max_new_tokens=80)
   answers.append(a)
  english=lang=="English"
  language_mix=any(bool(re.search(r"[가-힣]",a)) for a in answers) if english else False
  fixed_fallback=any(a in {FALLBACK_RESPONSE, FALLBACK_RESPONSE_EN} for a in answers)
  rows.append(dict(index=i+1,language=lang,prompts=[first,followup],answers=answers,
                   english_answer_contains_korean=language_mix,fixed_fallback=fixed_fallback))
 result={"mode":"raw-neural","model_file":args.model_file,"no_retrieval":True,
         "total_dialogues":len(rows),"fallback_dialogues":sum(int(r["fixed_fallback"]) for r in rows),
         "english_with_korean":sum(int(r["english_answer_contains_korean"]) for r in rows),"tests":rows,
         "note":"Human review is still needed. This metric does not establish factual accuracy or conversational coherence."}
 Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({k:v for k,v in result.items() if k!="tests"},ensure_ascii=False,indent=2))
 for row in rows:
  print(row["language"],row["prompts"],"=>",row["answers"])

if __name__=="__main__":
 main()
