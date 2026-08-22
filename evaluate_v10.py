#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from hgai_core import HGAIEngine

ap=argparse.ArgumentParser()
ap.add_argument('--mode',default='hybrid')
ap.add_argument('--model-file',default='hgai_model_v10.pth')
ap.add_argument('--preset',default='main')
ap.add_argument('--file',default='eval_v10.json')
ap.add_argument('--diversity-runs',type=int,default=12)
args=ap.parse_args()

e=HGAIEngine('train.txt',args.model_file,args.preset,seed=123,load_model=(args.mode!='search'))
items=json.loads(Path(args.file).read_text(encoding='utf-8')); ok=0
for i,x in enumerate(items,1):
    a=e.reply(x['q'],mode=args.mode,max_new_tokens=48)
    good=all(m.lower() in a.lower() for m in x.get('must',[])) and not any(f.lower() in a.lower() for f in x.get('forbid',[]))
    ok+=good; print(('OK' if good else 'FAIL'),i,x['q'],'=>',a)

# Regression for the old arr[0] bug: repeated exact prompts with multiple valid
# training answers must not collapse to one permanent response.
diversity_ok=True
if args.mode in {'search','hybrid'} and args.diversity_runs>0:
    q='동물 이름 아무거나 말해봐'
    answers=[e.reply(q,mode=args.mode,max_new_tokens=32) for _ in range(args.diversity_runs)]
    unique=len(set(answers))
    diversity_ok=unique>=3
    print(('OK' if diversity_ok else 'FAIL'),'diversity',f'{unique}/{args.diversity_runs} unique answers')

print(f'{ok}/{len(items)} = {ok/len(items):.1%}; diversity={"PASS" if diversity_ok else "FAIL"}')
raise SystemExit(0 if ok==len(items) and diversity_ok else 1)
