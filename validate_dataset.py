#!/usr/bin/env python3
import re, json, sys
from pathlib import Path
p=Path('train.txt')
text=p.read_text(encoding='utf-8')
pairs=[]; bad=[]; seen=set()
for i,line in enumerate(text.splitlines(),1):
    if not line.strip() or line.startswith('#'): continue
    if '=' not in line:
        bad.append((i,'no_equal',line)); continue
    q,a=line.split('=',1); q=q.strip(); a=a.strip()
    if not q or not a: bad.append((i,'empty',line))
    if ' 냥' in a: bad.append((i,'nyang_spacing',line))
    if (q,a) in seen: bad.append((i,'duplicate',line))
    seen.add((q,a)); pairs.append((q,a))
print(f'pairs={len(pairs):,} unique_questions={len(set(q for q,a in pairs)):,} issues={len(bad):,}')
for b in bad[:40]: print('WARN', *b)
# strict fail only for severe format/spacing issues
severe=[b for b in bad if b[1] in ('no_equal','empty','nyang_spacing')]
sys.exit(1 if severe else 0)
