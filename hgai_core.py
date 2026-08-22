#!/usr/bin/env python3
"""HGAI v10 practical standalone engine. External LLMs are never runtime deps."""
from __future__ import annotations
import ast,difflib,hashlib,math,random,re
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path
from build_dataset import CURATED_FACTS,curated_pairs,expand_pairs,is_obsolete_pair,parse_pairs

BASE_DIR=Path(__file__).resolve().parent
SYSTEM_TEXT="너는 HGAI다냥. 한국어로 정확하고 자연스럽게 답하고 말끝에 냥을 자연스럽게 붙인다냥. 모르는 사실은 지어내지 않는다냥."
SPECIAL=["<pad>","<unk>","<|system|>","<|user|>","<|context|>","<|assistant|>","<|end|>"]
PRESETS={
 "main":{"block_size":256,"batch_size":4,"n_embd":224,"n_head":7,"n_layer":5,"ffn_hidden":640,"dropout":.08,"max_vocab":16000},
 "large":{"block_size":512,"batch_size":2,"n_embd":320,"n_head":8,"n_layer":8,"ffn_hidden":896,"dropout":.08,"max_vocab":24000},
}

def clean_text(x): return re.sub(r"\s+"," ",str(x).strip())
def fix_nyang(x):
 x=clean_text(x); x=re.sub(r"\s+냥(?!냥)","냥",x); x=re.sub(r"냥\s+냥","냥냥",x); return re.sub(r"(냥){4,}","냥냥",x).strip()
WRAPS=["궁금한데","하나 물어볼게","질문 하나 할게","혹시","질문인데","물어볼게"]
def strip_wrap(x):
 x=clean_text(x)
 for _ in range(3):
  hit=False
  for p in WRAPS:
   if x.startswith(p+" "): x=x[len(p):].strip(" ,"); hit=True
  if not hit: break
 return re.sub(r"\s*(좀 알려줘|에 대해 답해줘)\s*$","",x).strip()
SYN=[("캐주얼티즈 언노운","casualtiesunknown"),("casualties: unknown","casualtiesunknown"),("casualties unknown","casualtiesunknown"),("대형 언어 모델","llm"),("대형언어모델","llm"),("깃허브","github"),("깃헙","github"),("파이썬","python"),("리눅스","linux"),("윈도우","windows"),("터먹스","termux"),("씨플플","c++"),("cpp","c++"),("인공지능","ai"),("에이아이","ai"),("램","ram"),("덥듀","doubleduo"),("hg ai","hgai"),("챗지피티","chatgpt"),("카톡","kakaotalk"),("디코","discord")]
def canon(x):
 s=strip_wrap(x).lower()
 for a,b in sorted(SYN,key=lambda z:-len(z[0])): s=s.replace(a,b)
 return s
def normalize(x): return re.sub(r"[\s\.\!\?\,\~\;\:\'\"`\[\]\(\){}<>/\\|_+=\-^…ㅋㅎㅠㅜ]","",canon(x))
def terms(x): return set(re.findall(r"[a-z][a-z0-9_+#.:-]{1,}|[가-힣]{2,}|\d+",canon(x)))
def grams(x,n=3):
 s=normalize(x); return {s[i:i+n] for i in range(max(1,len(s)-n+1))} if s else set()

def load_pairs(train_file):
 base=parse_pairs(Path(train_file).read_text(encoding="utf-8")); ef=Path(train_file).with_name("train_v10_extra.txt")
 if ef.exists(): base+=parse_pairs(ef.read_text(encoding="utf-8"))
 merged=[(q,a) for q,a in base if not is_obsolete_pair(q,a)]+curated_pairs(); out=[]; seen=set()
 for q,a in merged:
  k=(clean_text(q),fix_nyang(a))
  if k not in seen: seen.add(k); out.append(k)
 return out

@dataclass
class Hit: score:float; question:str; answer:str
class Retriever:
 def __init__(self,pairs):
  self.pairs=pairs; self.exact=defaultdict(list); self.n=[]; self.t=[]; self.g=[]; self.inv=defaultdict(set)
  for i,(q,a) in enumerate(pairs):
   n,t,g=normalize(q),terms(q),grams(q); self.n.append(n); self.t.append(t); self.g.append(g); self.exact[n].append(a)
   for z in t:self.inv["t:"+z].add(i)
   for z in g:self.inv["g:"+z].add(i)
 def exact_answers(self,q): return list(dict.fromkeys(self.exact.get(normalize(q),[])))
 def search(self,q,k=5):
  nq,tq,gq=normalize(q),terms(q),grams(q)
  if not nq:return []
  cand=set()
  for z in tq:cand|=self.inv.get("t:"+z,set())
  for z in gq:cand|=self.inv.get("g:"+z,set())
  if not cand:cand=range(len(self.pairs))
  scored=[]
  for i in cand:
   seq=difflib.SequenceMatcher(None,nq,self.n[i]).ratio(); qt,qg=self.t[i],self.g[i]
   tj=len(tq&qt)/max(1,len(tq|qt)); gj=len(gq&qg)/max(1,len(gq|qg)); cont=1.0 if (nq in self.n[i] or self.n[i] in nq) and min(len(nq),len(self.n[i]))>=3 else 0
   s=min(1,seq*.48+gj*.27+tj*.20+cont*.12)
   if s>=.34: scored.append(Hit(s,*self.pairs[i]))
  scored.sort(key=lambda h:h.score,reverse=True); out=[]; seen=set()
  for h in scored:
   if h.answer not in seen: seen.add(h.answer); out.append(h)
   if len(out)>=k: break
  return out

def curated_direct_reply(q):
 t=clean_text(q); nt=normalize(t)
 if not any(x in t for x in ["뭐","설명","알려","뜻","어떤","뭔데","뭐임","뭐 하는","왜","원인","달라","다름","차이","비교","쓰는","용도"]): return None
 if "tcp" in nt and "udp" in nt and any(x in t for x in ["달라","다름","차이","비교"]): return CURATED_FACTS.get("TCP와 UDP 차이")
 low=t.lower(); qterms=terms(t)
 for term in sorted(CURATED_FACTS,key=len,reverse=True):
  if re.fullmatch(r"[A-Za-z0-9+.#:_ -]+",term):
   pat=r"(?<![A-Za-z0-9])"+re.escape(term.lower())+r"(?![A-Za-z0-9])"; nterm=normalize(term); ok=bool(re.search(pat,low)) or (len(nterm)>=2 and nterm in qterms)
  else: nterm=normalize(term); ok=bool(nterm and nterm in nt)
  if ok:return CURATED_FACTS[term]
 return None

BIN={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,ast.Div:lambda a,b:a/b,ast.FloorDiv:lambda a,b:a//b,ast.Mod:lambda a,b:a%b,ast.Pow:lambda a,b:a**b}; UNA={ast.UAdd:lambda a:a,ast.USub:lambda a:-a}
def safe_eval(e):
 e=e.replace("×","*").replace("÷","/").replace("^","**"); e=re.sub(r"(?<=\d)[xX](?=\d)","*",e)
 if not re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\s]+",e):return None
 def ev(n):
  if isinstance(n,ast.Expression):return ev(n.body)
  if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):return n.value
  if isinstance(n,ast.BinOp) and type(n.op) in BIN:return BIN[type(n.op)](ev(n.left),ev(n.right))
  if isinstance(n,ast.UnaryOp) and type(n.op) in UNA:return UNA[type(n.op)](ev(n.operand))
  raise ValueError
 try:
  v=ev(ast.parse(e,mode="eval")); return int(v) if isinstance(v,float) and v.is_integer() else round(v,10)
 except:return None
def calculator_reply(q):
 m=re.fullmatch(r"\s*(-?\d+(?:\.\d+)?\s*(?:[+\-*/xX×÷%^]\s*-?\d+(?:\.\d+)?\s*)+)\s*[=?]?\s*",q) or re.search(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/xX×÷%^]\s*-?\d+(?:\.\d+)?\s*)+)",q)
 if m:
  v=safe_eval(m.group(1)); return f"계산하면 {v}다냥 :3" if v is not None else None
 return None

TOKRE=re.compile(r"<\|[^>]+?\|>|[A-Za-z][A-Za-z0-9_+#.:'-]*|\d+(?:\.\d+)?|[가-힣]+|[ㄱ-ㅎㅏ-ㅣ]+|\s+|.",re.S)
def basic_tokens(x):return [" " if m.group(0).isspace() else m.group(0) for m in TOKRE.finditer(x)]
class Tokenizer:
 def __init__(self,vocab):self.vocab=list(vocab);self.stoi={x:i for i,x in enumerate(vocab)};self.itos={i:x for i,x in enumerate(vocab)};self.pad=self.stoi["<pad>"];self.unk=self.stoi["<unk>"];self.end=self.stoi["<|end|>"]
 def encode(self,x):
  out=[]
  for z in basic_tokens(x):
   if z in self.stoi:out.append(self.stoi[z])
   else:out.extend(self.stoi.get(c,self.unk) for c in z)
  return out
 def decode(self,ids):return "".join(self.itos.get(int(i),"") for i in ids)
def build_vocab(pairs,mx):
 c=Counter(); ch=Counter()
 for q,a in pairs:
  s=f"<|user|> {q}\n<|assistant|> {a}<|end|>"; c.update(basic_tokens(s)); ch.update(s)
 v=[];seen=set()
 def add(x):
  if x not in seen:seen.add(x);v.append(x)
 for x in SPECIAL:add(x)
 for x,_ in ch.most_common():
  if not x.isspace():add(x)
 add(" ");add("\n")
 for x,_ in c.most_common():
  add(x)
  if len(v)>=mx:break
 return v[:mx]

try:
 import torch,torch.nn as nn,torch.nn.functional as F
 TORCH_AVAILABLE=True
except Exception: torch=nn=F=None;TORCH_AVAILABLE=False
if TORCH_AVAILABLE:
 class RMSNorm(nn.Module):
  def __init__(self,d,eps=1e-6):super().__init__();self.weight=nn.Parameter(torch.ones(d));self.eps=eps
  def forward(self,x):return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight
 def rope(x):
  _,_,T,D=x.shape;h=D//2; inv=1/(10000**(torch.arange(h,device=x.device,dtype=torch.float32)/h));ang=torch.outer(torch.arange(T,device=x.device,dtype=torch.float32),inv).to(x.dtype);c=ang.cos()[None,None];s=ang.sin()[None,None];a,b=x[...,:h],x[...,h:];return torch.cat((a*c-b*s,a*s+b*c),-1)
 class CausalSelfAttention(nn.Module):
  def __init__(self,cfg):super().__init__();d=cfg["n_embd"];self.n_head=cfg["n_head"];self.head_dim=d//self.n_head;self.dropout=cfg["dropout"];assert self.head_dim%2==0;self.qkv=nn.Linear(d,3*d,bias=False);self.proj=nn.Linear(d,d,bias=False)
  def forward(self,x):
   B,T,C=x.shape;q,k,v=self.qkv(x).chunk(3,-1);sh=lambda z:z.view(B,T,self.n_head,self.head_dim).transpose(1,2);q,k,v=sh(q),sh(k),sh(v);y=F.scaled_dot_product_attention(rope(q),rope(k),v,dropout_p=self.dropout if self.training else 0,is_causal=True);return self.proj(y.transpose(1,2).contiguous().view(B,T,C))
 class SwiGLU(nn.Module):
  def __init__(self,cfg):super().__init__();d,h=cfg["n_embd"],cfg["ffn_hidden"];self.gate=nn.Linear(d,h,bias=False);self.up=nn.Linear(d,h,bias=False);self.down=nn.Linear(h,d,bias=False)
  def forward(self,x):return self.down(F.silu(self.gate(x))*self.up(x))
 class Block(nn.Module):
  def __init__(self,cfg):super().__init__();d=cfg["n_embd"];self.n1=RMSNorm(d);self.attn=CausalSelfAttention(cfg);self.n2=RMSNorm(d);self.ff=SwiGLU(cfg);self.drop=nn.Dropout(cfg["dropout"])
  def forward(self,x):x=x+self.drop(self.attn(self.n1(x)));return x+self.drop(self.ff(self.n2(x)))
 class HGAIModel(nn.Module):
  def __init__(self,vocab_size,cfg):
   super().__init__();d=cfg["n_embd"];self.cfg=dict(cfg);self.embed=nn.Embedding(vocab_size,d);self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])]);self.norm=RMSNorm(d);self.head=nn.Linear(d,vocab_size,bias=False);self.head.weight=self.embed.weight;self.apply(self._init)
  def _init(self,m):
   if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,0,.02)
  def forward(self,idx,targets=None):
   x=self.embed(idx)
   for b in self.blocks:x=b(x)
   logits=self.head(self.norm(x));loss=None
   if targets is not None:loss=F.cross_entropy(logits[:,:-1].contiguous().view(-1,logits.size(-1)),targets[:,1:].contiguous().view(-1),ignore_index=-100)
   return logits,loss

def count_params(model):return sum(p.numel() for p in model.parameters()) if model is not None else 0
def checkpoint_load(path):
 if not TORCH_AVAILABLE or not Path(path).exists():return None
 try:return torch.load(path,map_location="cpu",weights_only=False)
 except TypeError:return torch.load(path,map_location="cpu")
def training_pairs(train_file,max_variants=12):
 base=parse_pairs(Path(train_file).read_text(encoding="utf-8"));ef=Path(train_file).with_name("train_v10_extra.txt")
 if ef.exists():base+=parse_pairs(ef.read_text(encoding="utf-8"))
 return expand_pairs(base,max_variants=max_variants)
def split_pairs(pairs,pct=2):
 tr=[];va=[]
 for x in pairs:(va if int(hashlib.md5(x[0].encode()).hexdigest()[:8],16)%100<pct else tr).append(x)
 return tr,va
def example(tok,cfg,q,a,ctx=False):
 p=f"<|system|> {SYSTEM_TEXT}<|end|>\n<|user|> {q}\n"+(f"<|context|> {a}<|end|>\n" if ctx else "")+"<|assistant|> ";t=f"{a}<|end|>";pi,ti=tok.encode(p),tok.encode(t);ids=pi+ti;labels=[-100]*len(pi)+ti;ex=max(0,len(ids)-cfg["block_size"]);return ids[ex:],labels[ex:]
def batch(tok,cfg,pairs,device,rng):
 xs=[example(tok,cfg,*pairs[rng.randrange(len(pairs))],rng.random()<.35) for _ in range(cfg["batch_size"])];T=max(len(x[0]) for x in xs);xb=torch.full((len(xs),T),tok.pad,dtype=torch.long);yb=torch.full((len(xs),T),-100,dtype=torch.long)
 for i,(a,b) in enumerate(xs):xb[i,:len(a)]=torch.tensor(a);yb[i,:len(b)]=torch.tensor(b)
 return xb.to(device),yb.to(device)
@torch.no_grad() if TORCH_AVAILABLE else (lambda f:f)
def eval_loss(model,tok,cfg,pairs,device,seed,batches=8):
 if not pairs:return 999.
 r=random.Random(seed);model.eval();vals=[]
 for _ in range(batches):x,y=batch(tok,cfg,pairs,device,r);_,l=model(x,y);vals.append(float(l))
 model.train();return sum(vals)/len(vals)
def train_model(train_file,model_file,preset="main",steps=3000,lr=3e-4,seed=1337,device="auto",resume=False,threads=0,status=print):
 if not TORCH_AVAILABLE:raise RuntimeError("PyTorch가 필요하다냥")
 if threads>0:torch.set_num_threads(threads)
 random.seed(seed);torch.manual_seed(seed);rng=random.Random(seed);cfg=dict(PRESETS[preset]);pairs=training_pairs(train_file);tr,va=split_pairs(pairs);dev="cuda" if device in ("auto","cuda") and torch.cuda.is_available() else "cpu";ck=checkpoint_load(model_file) if resume else None
 if ck and ck.get("version")=="hgai-v10" and ck.get("preset")==preset:
  tok=Tokenizer(ck["vocab"]);model=HGAIModel(len(tok.vocab),ck["config"]).to(dev);model.load_state_dict(ck["model"]);start=int(ck.get("step",0));best=float(ck.get("best_val",999.));cfg=dict(ck["config"])
 else:tok=Tokenizer(build_vocab(tr,cfg["max_vocab"]));model=HGAIModel(len(tok.vocab),cfg).to(dev);start=0;best=999.
 status(f"HGAI v10 preset={preset} params={count_params(model):,} pairs={len(pairs):,} device={dev}");opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.95),weight_decay=.1);total=max(1,start+steps);warm=max(20,min(300,total//20));ev=max(25,min(250,steps//10 if steps>=10 else 1));last=None
 for local in range(steps):
  step=start+local;x,y=batch(tok,cfg,tr,dev,rng);_,loss=model(x,y);last=float(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);p=min(1,max(0,(step-warm)/max(1,total-warm)));scale=(step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*p));[g.update(lr=lr*max(.05,scale)) for g in opt.param_groups];opt.step()
  if local==0 or (local+1)%ev==0 or local==steps-1:
   vl=eval_loss(model,tok,cfg,va,dev,seed+step);best=min(best,vl);status(f"step {step+1} train={last:.4f} val={vl:.4f}");torch.save({"version":"hgai-v10","preset":preset,"config":cfg,"vocab":tok.vocab,"model":model.state_dict(),"step":step+1,"best_val":best,"last_val":vl,"params":count_params(model)},model_file)
 model.eval();return model,tok,dev,{"pairs":len(pairs),"best_val":best,"last_loss":last,"params":count_params(model),"step":start+steps}

@torch.no_grad() if TORCH_AVAILABLE else (lambda f:f)
def neural_generate(model,tok,device,cfg,user,context=None,history=None,temperature=.72,top_k=40,top_p=.92,max_new_tokens=96,repetition_penalty=1.12,seed=None):
 if model is None:return None
 prompt=f"<|system|> {SYSTEM_TEXT}<|end|>\n"+"".join(f"<|user|> {u}\n<|assistant|> {a}<|end|>\n" for u,a in (history or [])[-2:])+f"<|user|> {user}\n"+(f"<|context|> {context}<|end|>\n" if context else "")+"<|assistant|> ";idx=torch.tensor([tok.encode(prompt)[-cfg["block_size"]:]],device=device);gen=[];rg=None
 if seed is not None:rg=torch.Generator(device=device);rg.manual_seed(int(seed))
 for _ in range(max_new_tokens):
  z=model(idx[:,-cfg["block_size"]:])[0][:,-1].float()
  for i in set(gen[-64:]):v=z[0,i];z[0,i]=torch.where(v<0,v*repetition_penalty,v/repetition_penalty)
  z/=max(.05,float(temperature))
  if top_k:
   v,_=torch.topk(z,min(top_k,z.size(-1)));z[z<v[:,-1,None]]=-float("inf")
  p=F.softmax(z,-1)
  if 0<top_p<1:
   sp,si=torch.sort(p,descending=True);cm=torch.cumsum(sp,-1);mask=cm>top_p;mask[...,1:]=mask[...,:-1].clone();mask[...,0]=False;sp=sp.masked_fill(mask,0);sp/=sp.sum(-1,keepdim=True);pick=torch.multinomial(sp,1,generator=rg);nxt=si.gather(-1,pick)
  else:nxt=torch.multinomial(p,1,generator=rg)
  i=int(nxt.item())
  if i==tok.end:break
  gen.append(i);idx=torch.cat((idx,nxt),1)
 out=fix_nyang(tok.decode(gen).replace("<unk>",""));return re.split(r"<\|(?:user|context|assistant|system|end)\|>",out)[0].strip()
def quality_ok(a,context=None):
 if not a or len(a)<3 or len(a)>600 or a.count("냥")>14:return False
 if context and terms(a) and terms(context) and not (terms(a)&terms(context)):return False
 return not bool(re.search(r"(.{2,12})\1\1\1",a))

class HGAIEngine:
 def __init__(self,train_file=BASE_DIR/"train.txt",model_file=BASE_DIR/"hgai_model_v10.pth",preset="main",device="auto",seed=None,load_model=True):
  self.train_file=Path(train_file);self.model_file=Path(model_file);self.preset=preset;self.cfg=dict(PRESETS[preset]);self.rng=random.Random(seed if seed is not None else random.SystemRandom().randrange(1<<63));self.pairs=load_pairs(self.train_file);self.retriever=Retriever(self.pairs);self.history=[];self.model=self.tok=None;self.device="cpu";self.neural_ready=False;self.checkpoint_meta={}
  if load_model and TORCH_AVAILABLE:
   c=checkpoint_load(self.model_file)
   if c and c.get("version")=="hgai-v10" and c.get("preset")==preset:
    self.device="cuda" if device in ("auto","cuda") and torch.cuda.is_available() else "cpu";self.tok=Tokenizer(c["vocab"]);self.cfg=dict(c["config"]);self.model=HGAIModel(len(self.tok.vocab),self.cfg).to(self.device);self.model.load_state_dict(c["model"]);self.model.eval();self.checkpoint_meta={"step":int(c.get("step",0)),"best_val":float(c.get("best_val",999)),"last_val":float(c.get("last_val",999))};self.neural_ready=self.checkpoint_meta["step"]>=1500 and self.checkpoint_meta["best_val"]<=2.8
 def remember_rule(self,q):
  if any(x in q for x in ["내가 방금 뭐라고","방금 내가 뭐라고","내가 아까 뭐라고"]):return f"방금 너는 '{self.history[-1][0]}'라고 말했다냥" if self.history else "아직 앞에서 한 말이 없다냥"
  m=re.search(r"(?:내 이름은|내이름은)\s*([가-힣A-Za-z0-9_\-]{1,24}?)(?:이야|야|이다|다)?$",q)
  if m:self.user_name=m.group(1);return f"기억했다냥 이제 {self.user_name}라고 부르겠다냥 :3"
  if re.search(r"내\s*이름.*(뭐|기억)",q):return f"네 이름은 {self.user_name}다냥" if hasattr(self,"user_name") else "아직 이름을 못 들었다냥"
 def _save(self,q,a):self.history.append((q,a));self.history=self.history[-8:]
 def reply(self,user,mode="hybrid",temperature=.72,top_k=40,top_p=.92,max_new_tokens=96):
  q=clean_text(user)
  if not q:return "냥? 뭔가 말해줘라냥 :3"
  for fn in (self.remember_rule,calculator_reply,curated_direct_reply):
   a=fn(q)
   if a:self._save(q,a);return a
  if mode=="raw-neural":
   g=neural_generate(self.model,self.tok,self.device,self.cfg,q,None,self.history,temperature,top_k,top_p,max_new_tokens) if self.model else None;a=fix_nyang(g) if quality_ok(g) else "순수 신경망 생성이 아직 불안정하다냥";self._save(q,a);return a
  ex=self.retriever.exact_answers(q)
  if ex:a=fix_nyang(self.rng.choice(ex));self._save(q,a);return a
  hits=self.retriever.search(q);best=hits[0] if hits else None
  if mode=="search":a=best.answer if best and best.score>=.52 else "잘 모르겠다냥 새로 배우고 싶다냥 :3";self._save(q,a);return fix_nyang(a)
  if best and best.score>=.86:a=fix_nyang(self.rng.choice([h.answer for h in hits if h.score>=best.score-.03] or [best.answer]));self._save(q,a);return a
  if best and best.score>=.50:
   g=neural_generate(self.model,self.tok,self.device,self.cfg,q,best.answer,self.history,temperature,top_k,top_p,max_new_tokens) if self.model and (self.neural_ready or mode=="neural") else None;a=g if quality_ok(g,best.answer) else best.answer;self._save(q,a);return fix_nyang(a)
  if self.model and (mode=="neural" or (mode=="hybrid" and self.neural_ready)):
   g=neural_generate(self.model,self.tok,self.device,self.cfg,q,None,self.history,temperature,top_k,top_p,max_new_tokens)
   if quality_ok(g):self._save(q,g);return fix_nyang(g)
  a="그건 아직 확실하게 모르겠다냥 아는 척 지어내기보단 새로 배우는 게 낫다냥 :3";self._save(q,a);return a
 def model_info(self):return {"version":"hgai-v10","preset":self.preset,"runtime_pairs":len(self.pairs),"model_loaded":self.model is not None,"params":count_params(self.model),"neural_ready":self.neural_ready,"checkpoint":self.checkpoint_meta,"device":self.device,"external_model_required":False}
