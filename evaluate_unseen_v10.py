#!/usr/bin/env python3
"""HGAI v10 raw-neural unseen/generalization evaluation.

Every prompt is intentionally absent from augmented_train.txt by normalized
exact match.  This is stronger than evaluate_neural_v10.py, whose core-fact
questions mainly check whether the model learned the required knowledge.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from hgai_core import HGAIEngine, normalize

TESTS=[
 ("HGAI는 누가 만들었고 어떤 종류의 AI야?", [["hgai"],["덥듀"],["ai","고양이"]]),
 ("덥듀가 모바일로 포팅 중인 수인 세계관 하드코어 생존 게임을 설명해줘", [["casualties","게임"],["생존","하드코어"]]),
 ("Git 저장소에 PR과 Actions까지 제공하는 서비스는 어떤 역할을 해?", [["github"],["git","코드","저장소"],["pr","actions","협업"]]),
 ("읽기 쉬운 문법과 많은 라이브러리를 가진 범용 프로그래밍 언어 하나를 설명해줘", [["python","파이썬"],["언어","프로그래밍"]]),
 ("데이터 전달에서 순서와 재전송을 챙기는 방식과 가벼운 데이터그램 방식의 차이는?", [["tcp"],["udp"],["순서","재전송","신뢰"],["지연","가벼운","오버헤드"]]),
 ("AI 학습에서 병렬 연산에 강한 그래픽 프로세서를 쓰는 이유가 뭐야?", [["gpu"],["병렬","연산","그래픽"]]),
 ("객체와 배열을 텍스트로 표현해서 프로그램끼리 데이터를 주고받는 형식을 설명해줘", [["json"],["데이터","객체","배열","텍스트"]]),
 ("답을 만들기 전에 관련 문서를 찾아서 참고시키는 AI 기법은 뭐고 어떻게 동작해?", [["rag"],["검색","문서","자료"],["참고","답","생성"]]),
 ("안드로이드 앱을 설치하거나 배포할 때 쓰는 패키지 파일 형식이 뭐야?", [["apk"],["안드로이드"],["설치","배포","패키지"]]),
 ("모델 가중치를 더 적은 비트로 표현해서 메모리와 연산량을 줄이는 방법을 설명해줘", [["양자화"],["비트"],["메모리","연산"]]),
 ("BitNet과 일반적인 사후 양자화는 뭐가 달라?", [["bitnet"],["양자화"],["학습","저비트","1.58","ternary","삼진"]]),
 ("LoRA가 전체 파라미터를 전부 학습하는 방식보다 가벼운 이유는?", [["lora"],["파라미터","가중치"],["저랭크","작은","일부","효율"]]),
 ("DNS가 없으면 도메인 이름으로 서버를 찾기 어려운 이유를 설명해줘", [["dns"],["도메인"],["ip","주소"]]),
 ("HTTPS에서 TLS를 쓰는 핵심 이유가 뭐야?", [["https"],["tls","암호"],["도청","변조","보안"]]),
 ("SSH는 원격 컴퓨터를 다룰 때 어떤 보안 이점을 줘?", [["ssh"],["암호","암호화"],["원격","로그인","명령"]]),
 ("프레임이 갑자기 떨어질 때 CPU GPU GC 발열 중 어떤 것들을 의심해야 해?", [["cpu","gpu"],["gc","발열","병목"]]),
 ("메모리 누수가 오래 실행할수록 문제를 키우는 이유는?", [["메모리"],["누수"],["증가","해제","회수","계속"]]),
 ("검증 데이터와 학습 데이터를 굳이 나누는 이유가 뭐야?", [["검증"],["학습"],["일반화","과적합","성능"]]),
 ("오버피팅이 생기면 처음 보는 질문에서 어떤 문제가 나타나?", [["오버피팅","과적합"],["새","처음","일반화"],["성능","약","떨어"]]),
 ("지식 증류에서 teacher와 student는 각각 무슨 역할이야?", [["교사","teacher"],["학생","student"],["학습","출력","분포","지식"]]),
]

def norm_text(s): return re.sub(r"\s+"," ",str(s).strip().lower())
def semantic_pass(answer,groups):
 a=norm_text(answer)
 if not a or "순수 신경망 생성이 아직 불안정" in a: return False
 return all(any(term.lower() in a for term in group) for group in groups)

def assert_unseen(dataset_file):
 p=Path(dataset_file)
 if not p.exists(): raise SystemExit(f"dataset not found: {p}")
 known={normalize(line.split('=',1)[0]) for line in p.read_text(encoding='utf-8').splitlines() if '=' in line}
 leaks=[q for q,_ in TESTS if normalize(q) in known]
 if leaks:
  raise SystemExit("unseen evaluation leaked into training data: "+repr(leaks))
 return len(known)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model-file',default='hgai_model_v10.pth')
 ap.add_argument('--preset',default='main')
 ap.add_argument('--dataset-file',default='augmented_train.txt')
 ap.add_argument('--min-pass',type=int,default=10,help='default 10/20; do not lower merely to make a model pass')
 ap.add_argument('--seed',type=int,default=20260822)
 ap.add_argument('--report',default='')
 args=ap.parse_args()
 unique=assert_unseen(args.dataset_file)
 print(f'unseen leak check PASS against {unique:,} normalized training questions')
 e=HGAIEngine('train.txt',args.model_file,args.preset,seed=args.seed,load_model=True)
 if e.model is None: raise SystemExit('model not loaded')
 rows=[];passed=0
 for i,(q,groups) in enumerate(TESTS,1):
  e.history=[]
  torch.manual_seed(args.seed+i)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed+i)
  a=e.reply(q,mode='raw-neural',temperature=.50,top_k=16,top_p=.86,max_new_tokens=96)
  good=semantic_pass(a,groups); passed+=good
  rows.append({'index':i,'question':q,'answer':a,'passed':bool(good),'required_groups':groups})
  print(('OK' if good else 'FAIL'),i,q,'=>',a)
 report={'preset':args.preset,'seed':args.seed,'training_question_count':unique,'leak_free':True,'passed':passed,'total':len(TESTS),'required':args.min_pass,'gate_passed':passed>=args.min_pass,'tests':rows}
 if args.report:
  Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(f'unseen raw-neural generalization {passed}/{len(TESTS)}; required={args.min_pass}')
 raise SystemExit(0 if passed>=args.min_pass else 1)

if __name__=='__main__': main()
