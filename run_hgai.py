#!/usr/bin/env python3
# HGAI v6.2 Hybrid-Token
# - Exact/keyword/fuzzy search + calculator + memory + token-based Transformer
# - Keeps train.txt compatibility: 질문=답변
# - Default uses Torch when available; fallback to Lite when unavailable

import argparse, ast, difflib, hashlib, json, math, os, random, re, sys
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = BASE_DIR / "train.txt"
DEFAULT_MODEL_FILE = BASE_DIR / "hgai_model.pth"
DEFAULT_KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"
MEMORY_FILE = BASE_DIR / "memory.json"

# lazy torch
torch = None
nn = None
F = None
TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = None

PRESETS = {
    "tiny":      {"block_size": 80,  "batch_size": 10, "n_embd": 80,  "n_head": 4, "n_layer": 2, "dropout": 0.10, "max_vocab": 6000},
    "phone":     {"block_size": 96,  "batch_size": 8,  "n_embd": 112, "n_head": 4, "n_layer": 2, "dropout": 0.10, "max_vocab": 8000},
    "mid-safe":  {"block_size": 128, "batch_size": 8,  "n_embd": 160, "n_head": 5, "n_layer": 3, "dropout": 0.12, "max_vocab": 12000},
    "mid-plus":  {"block_size": 160, "batch_size": 6,  "n_embd": 192, "n_head": 6, "n_layer": 4, "dropout": 0.12, "max_vocab": 16000},
    "pc":        {"block_size": 192, "batch_size": 8,  "n_embd": 256, "n_head": 8, "n_layer": 4, "dropout": 0.12, "max_vocab": 20000},
}

parser = argparse.ArgumentParser(description="HGAI v6.2 hybrid-token Korean cat chatbot")
parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
parser.add_argument("--knowledge-file", default=str(DEFAULT_KNOWLEDGE_FILE))
parser.add_argument("--preset", default="mid-safe", choices=sorted(PRESETS.keys()))
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--auto-train", action="store_true")
parser.add_argument("--lite", action="store_true", help="Torch를 전혀 쓰지 않고 검색/룰만 사용")
parser.add_argument("--torch-only", action="store_true", help="Torch/모델이 없으면 종료")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
parser.add_argument("--once", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.70)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--max-new-tokens", type=int, default=60)
parser.add_argument("--fuzzy-threshold", type=float, default=0.66)
parser.add_argument("--generation-mode", default="safe", choices=["safe","creative","off"], help="기본 safe: 정확/계산/검색 우선, 필요할 때 생성")
parser.add_argument("--reply-mode", default="hybrid", choices=["hybrid","search-first","torch-first","search-only","torch-only"])
parser.add_argument("--info", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--validate", action="store_true")
parser.add_argument("--install-help", action="store_true")
args = parser.parse_args()
CONFIG = dict(PRESETS[args.preset])
random.seed(args.seed)

def status(*x):
    if args.once is None:
        print(*x)

def load_torch():
    global torch, nn, F, TORCH_AVAILABLE, TORCH_IMPORT_ERROR
    if TORCH_AVAILABLE:
        return True
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch, nn, F = _torch, _nn, _F
        TORCH_AVAILABLE = True
        TORCH_IMPORT_ERROR = None
        return True
    except Exception as e:
        TORCH_IMPORT_ERROR = e
        return False

def print_install_help():
    print("""
[Termux Torch]
pkg update
pkg install python python-torch
python -c "import torch; print(torch.__version__)"

[GitHub Actions]
저장소에 push하면 .github/workflows/train.yml 이 자동 학습을 돌리고 artifact로 모델 zip을 올린다냥.

[폰 실행]
python run_hgai.py --preset mid-safe
python run_hgai.py --lite

[학습]
python run_hgai.py --retrain --preset phone --steps 800 --threads 2
python run_hgai.py --retrain --preset mid-safe --steps 5000 --threads 2
""".strip())

if args.install_help:
    print_install_help(); raise SystemExit(0)

if not args.lite:
    load_torch()
if TORCH_AVAILABLE:
    torch.manual_seed(args.seed)
    if args.threads > 0:
        try: torch.set_num_threads(args.threads)
        except Exception: pass

TRAIN_FILE = Path(args.train_file)
MODEL_FILE = Path(args.model_file)
KNOWLEDGE_FILE = Path(args.knowledge_file)
if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"train file not found: {TRAIN_FILE}")
raw_train = TRAIN_FILE.read_text(encoding="utf-8")
try:
    KNOWLEDGE = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8")) if KNOWLEDGE_FILE.exists() else {}
except Exception:
    KNOWLEDGE = {}

# ---------- text cleanup / dataset ----------
def fix_nyang(text):
    text = str(text).strip()
    text = re.sub(r"\s+냥(?!냥)", "냥", text)
    text = re.sub(r"냥\s+냥", "냥냥", text)
    text = re.sub(r"(냥){4,}", "냥냥", text)
    text = text.replace("언 어", "언어").replace("파이 썬", "파이썬")
    text = re.sub(r"\b(\S{2,})\s+\1\b", r"\1", text)
    text = re.sub(r"(\S{2,6})(?:\s+\1){1,}", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def clean_text(text):
    return re.sub(r"\s+", " ", str(text).strip())

SYNONYMS = [
    ("대형 언어 모델","llm"),("대형언어모델","llm"),("large language model","llm"),("llm","llm"),
    ("깃허브","github"),("깃헙","github"),("기트허브","github"),("github","github"),
    ("파이썬","python"),("python","python"),("윈도우","windows"),("windows","windows"),
    ("리눅스","linux"),("linux","linux"),("텀ux","termux"),("터먹스","termux"),("termux","termux"),
    ("라즈베리 파이","raspberrypi"),("라즈베리파이","raspberrypi"),("라즈베리","raspberrypi"),
    ("씨언어","c언어"),("씨 언어","c언어"),("c 언어","c언어"),("c언어","c언어"),
    ("씨플플","c++"),("c플플","c++"),("cpp","c++"),("c++","c++"),
    ("에이아이","ai"),("인공지능","ai"),("ai","ai"),("램","ram"),("ram","ram"),
    ("퍼리","furry"),("수인","furry"),("furry","furry"),
    ("hg company","hgcompany"),("hg사","hgcompany"),("hg 회사","hgcompany"),("hg컴퍼니","hgcompany"),("hgcompany","hgcompany"),
    ("덥듀","doubleduo"),("doubleduo","doubleduo"),("hg ai","hgai"),("hgai","hgai"),
]
STRONG_TERMS = sorted({b for _,b in SYNONYMS} | {"api","json","html","css","java","javascript","torch","pytorch","git","cpu","gpu","os","apk","zip"}, key=len, reverse=True)

def normalize(s):
    s = clean_text(s).lower()
    for a,b in sorted(SYNONYMS, key=lambda x:-len(x[0])):
        s = s.replace(a,b)
    s = re.sub(r"[\s\.\!\?\,\~\;\:\'\"`\[\]\(\){}<>/\\|_+=\-^…ㅋㅎㅠㅜ]", "", s)
    return s

def words_for_terms(s):
    low = clean_text(s).lower()
    canon = low
    for a,b in sorted(SYNONYMS, key=lambda x:-len(x[0])):
        canon = canon.replace(a,b)
    found = set()
    for t in STRONG_TERMS:
        if t in canon:
            found.add(t)
    # also add latin words and korean noun-ish chunks
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_+#.]*|[가-힣]{2,}", canon):
        if len(w) >= 2 and w not in {"뭐야","알려줘","설명해줘","대해","쉽게","아무거나","말해봐"}:
            found.add(w)
    return found

def parse_pairs(text):
    pairs=[]
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        q,a=line.split("=",1)
        q=clean_text(q); a=fix_nyang(a)
        if q and a:
            pairs.append((q,a))
    return pairs
PAIRS = parse_pairs(raw_train)
PAIR_HASH = hashlib.sha256("\n".join(f"{q}={a}" for q,a in PAIRS).encode()).hexdigest()
by_exact=defaultdict(list)
for q,a in PAIRS:
    by_exact[normalize(q)].append(a)
q_norms=[normalize(q) for q,a in PAIRS]
q_terms=[words_for_terms(q) for q,a in PAIRS]
questions=[q for q,a in PAIRS]
answers=[a for q,a in PAIRS]

def stable_choice(items, seed=""):
    if not items: return None
    h=int(hashlib.md5((seed+"|"+str(len(items))).encode()).hexdigest(),16)
    return items[h % len(items)]

# ---------- calculator ----------
ALLOWED_BINOPS={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,ast.Div:lambda a,b:a/b,ast.FloorDiv:lambda a,b:a//b,ast.Mod:lambda a,b:a%b,ast.Pow:lambda a,b:a**b}
ALLOWED_UNARY={ast.UAdd:lambda a:a, ast.USub:lambda a:-a}
def safe_eval_expr(expr):
    expr=expr.replace("×","*").replace("x","*").replace("X","*").replace("÷","/").replace("^","**")
    if not re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\s]+", expr): return None
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS: return ALLOWED_BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY: return ALLOWED_UNARY[type(node.op)](ev(node.operand))
        raise ValueError("bad expr")
    try:
        val=ev(ast.parse(expr, mode="eval"))
        if abs(val) > 10**30: return None
        return int(val) if isinstance(val,float) and val.is_integer() else round(val,10)
    except Exception: return None

def is_math_query(text):
    t=clean_text(text)
    if re.search(r"계산\s*(하지마|하지 마|말고|안해|안 해|금지)", t) or ("계산" in t and any(x in t for x in ["하지", "말고", "안해", "안 해", "금지"])):
        return False
    if not re.search(r"\d", t):
        return False
    expr = re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*(?:[+\-*/xX×÷%^]\s*-?\d+(?:\.\d+)?\s*)+\s*", t)
    if expr:
        return True
    if re.search(r"\d+\s*부터\s*\d+\s*까지\s*(더해|합|더하면|합계)", t):
        return True
    if re.search(r"\d+\s*단(\s|$|[을를]?\s*(외워|알려|출력|보여))", t):
        return True
    if re.search(r"\d+\s*팩토리얼", t):
        return True
    if re.search(r"\d+(?:\.\d+)?\s*의\s*\d+(?:\.\d+)?\s*%", t):
        return True
    return False

def calc_reply(user):
    t=user.strip()
    if not is_math_query(t):
        return None
    m=re.fullmatch(r"\s*(-?\d+(?:\.\d+)?\s*(?:[+\-*/xX×÷/%^]\s*-?\d+(?:\.\d+)?\s*)+)\s*", t)
    if not m:
        m=re.search(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/xX×÷/%^]\s*-?\d+(?:\.\d+)?\s*)+)", t)
    if m:
        v=safe_eval_expr(m.group(1))
        if v is not None: return f"계산하면 {v}다냥 :3"
    m=re.search(r"(\d+)\s*부터\s*(\d+)\s*까지\s*(더해|합|더하면|합계)", t)
    if m:
        a,b=map(int,m.group(1,2));
        if abs(b-a) <= 1000000:
            n=abs(b-a)+1; return f"계산하면 {(a+b)*n//2}다냥 :3"
    m=re.search(r"(\d+)\s*단", t)
    if m:
        n=int(m.group(1));
        if 1<=n<=99: return " / ".join(f"{n}×{i}={n*i}" for i in range(1,10)) + " 다냥"
    m=re.search(r"(\d+)\s*팩토리얼", t)
    if m:
        n=int(m.group(1));
        if 0<=n<=100: return f"계산하면 {math.factorial(n)}다냥 :3"
    return None

# ---------- memory / random rules ----------
def load_memory():
    try: return json.loads(MEMORY_FILE.read_text(encoding="utf-8")) if MEMORY_FILE.exists() else {}
    except Exception: return {}
def save_memory(mem):
    try: MEMORY_FILE.write_text(json.dumps(mem,ensure_ascii=False,indent=2),encoding="utf-8")
    except Exception: pass

def rule_reply(user):
    t=clean_text(user)
    nt=normalize(t)
    if re.search(r"계산\s*(하지마|하지 마|말고|안해|안 해|금지)", t) or ("계산" in t and any(x in t for x in ["하지", "말고", "안해", "안 해", "금지"])):
        return "알겠다냥 이번엔 계산 안 하겠다냥"
    if is_noise_input(t):
        return "냥? 한 글자나 의미 없는 입력은 어렵다냥 조금 더 말해줘라냥 :3"
    m=re.search(r"내 이름은\s*([가-힣A-Za-z0-9_\-]{1,20})", t)
    if m:
        mem=load_memory(); mem["user_name"]=m.group(1); save_memory(mem)
        return f"기억했다냥 이제 {m.group(1)}라고 불러주겠다냥 :3"
    if "내이름" in nt and ("뭐" in nt or "기억" in nt):
        name=load_memory().get("user_name")
        return f"네 이름은 {name}다냥 내가 기억하고 있다냥" if name else "아직 이름을 못 들었다냥 알려주면 기억하겠다냥"
    if ("동물이름" in nt or "아무동물" in nt):
        animal=random.choice(KNOWLEDGE.get('random',{}).get('animals',['고양이']))
        return f"{animal}이다냥 :3"
    if ("게임이름" in nt or "아무게임" in nt):
        game=random.choice(KNOWLEDGE.get('random',{}).get('games',['마인크래프트']))
        return f"{game} 어떠냐냥"
    if "죽고싶" in t or "자해" in t or "사라지고싶" in t:
        return "지금은 혼자 버티지 말고 바로 주변 어른이나 긴급 도움에 연락해야 한다냥 네 안전이 제일 중요하다냥"
    if len(nt) <= 1:
        return "냥? 한 글자만으론 어렵다냥 조금 더 말해줘라냥 :3"
    if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ가-힣]{2,}", t.replace(" ","")) and len(set(t.replace(" ","")))<=2 and len(t)>=4:
        return "냥냥고로롱... 그 말은 새로 배우고 싶다냥"
    return None

# ---------- search ----------
def exact_reply(user):
    arr=by_exact.get(normalize(user))
    return arr[0] if arr else None

def is_noise_input(user):
    t=clean_text(user)
    compact=re.sub(r"\s+", "", t)
    if not compact:
        return True
    if len(compact) <= 1:
        return True
    if re.fullmatch(r"""[\.\?\!~`'"ㆍᆞ·…\-_=+*/\\|,;:()\[\]{}<>]+""", compact):
        return True
    if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣㆍᆞ·]+", compact):
        return True
    if len(compact) <= 3 and re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ가-힣ㆍᆞ·]+", compact) and not re.search(r"[가-힣]{2,}", compact):
        return True
    return False

def protected_reply(user):
    t=clean_text(user)
    nt=normalize(t)
    terms=words_for_terms(t)
    req=KNOWLEDGE.get("required_answers", {})
    concepts=KNOWLEDGE.get("concepts", {})
    # 정확해야 하는 세계관/정체성 답변
    if "hgcompany" in terms:
        return req.get("HG Company가 뭐야") or "여러 IT 주제를 다루거나 특이한 거 개발도 하는 회사다냥"
    if "doubleduo" in terms and any(k in t for k in ["누구", "뭐야", "만든", "개발"]):
        return "덥듀는 나 HGAI를 만든 개발자이자 HG Company를 굴리는 사람이다냥"
    if ("hgai" in terms or re.search(r"너(는|가)?\s*(누구|뭐야|정체)", t)) and any(k in t for k in ["누구", "뭐야", "정체", "소개"]):
        return req.get("너는 뭐야") or "나는 HGAI다냥 한국어로 대화하는 작은 고양이 AI다냥 :3"
    if any(k in t for k in ["어디서", "어디에서", "만들어졌", "태어났"]) and ("너" in t or "HGAI" in t or "hgai" in nt):
        return req.get("너는 어디에서 만들어졌어") or "나는 HG Company라는 곳에서 덥듀라는 사람에게 개발되었다냥"
    if "furry" in terms:
        if "너" in t or "HGAI" in t:
            return "나는 현실의 퍼리는 아니고 고양이 말투를 쓰는 HGAI다냥 :3"
        return req.get("퍼리가 뭐야") or "동물을 의인화한 캐릭터 또는 그런 것들에 끌리는 취향을 말한다냥"
    # 지식 용어는 엉뚱한 용어로 섞이지 않게 직접 답변
    concept_map=[
        (("llm",), "LLM"), (("github",), "GitHub"), (("git",), "Git"),
        (("ram",), "RAM"), (("cpu",), "CPU"), (("gpu",), "GPU"),
        (("python",), "Python"), (("c언어",), "C언어"), (("c++",), "C++"),
        (("linux",), "리눅스"), (("windows",), "윈도우"), (("termux",), "Termux"),
        (("torch", "pytorch"), "PyTorch"), (("api",), "API"), (("json",), "JSON"),
        (("html",), "HTML"), (("css",), "CSS"), (("javascript",), "JavaScript"),
        (("token", "토큰"), "토큰"), (("parameter", "파라미터"), "파라미터"), (("epoch", "에포크"), "에포크"),
    ]
    if any(k in t for k in ["뭐야", "뭐임", "뜻", "설명", "알려", "대해", "란"]):
        for aliases,key in concept_map:
            if any(a in terms or a in nt for a in aliases):
                return concepts.get(key) or concepts.get(key.upper())
    return None

def fuzzy_reply(user, threshold=None):
    threshold = args.fuzzy_threshold if threshold is None else threshold
    nu=normalize(user); tu=words_for_terms(user)
    if not nu: return None
    best=(-1,None)
    is_knowledge = any(x in user for x in ["뭐야","뭐임","뜻","설명","알려","대해","란"])
    for i, qn in enumerate(q_norms):
        if not qn: continue
        tq=q_terms[i]
        # Strong term guard: if user asks about LLM/GitHub/etc, never match RAM/Python/etc without overlap.
        strong_u = {x for x in tu if x in STRONG_TERMS or re.fullmatch(r"[a-zA-Z0-9_+#.]{2,}", x)}
        strong_q = {x for x in tq if x in STRONG_TERMS or re.fullmatch(r"[a-zA-Z0-9_+#.]{2,}", x)}
        if is_knowledge and strong_u and strong_q and not (strong_u & strong_q):
            continue
        seq=difflib.SequenceMatcher(None, nu, qn).ratio()
        overlap=len(tu & tq) / max(1, len(tu | tq))
        prefix = 0.08 if (nu.startswith(qn) or qn.startswith(nu)) else 0
        score=seq*0.70 + overlap*0.25 + prefix
        if score > best[0]: best=(score, answers[i])
    if best[0] >= threshold:
        return best[1]
    return None

# ---------- tokenization ----------
SPECIAL=["<pad>","<unk>","<|user|>","<|assistant|>","<|end|>"]
def basic_tokenize(text):
    toks=[]
    for m in re.finditer(r"<\|[^>]+?\|>|[A-Za-z][A-Za-z0-9_+#.]*|\d+(?:\.\d+)?|[가-힣]+|[ㄱ-ㅎㅏ-ㅣ]+|\s+|.", text, re.S):
        x=m.group(0)
        toks.append(" " if x.isspace() else x)
    return toks

def train_corpus():
    return "\n".join(f"<|user|> {q}\n<|assistant|> {a}\n<|end|>" for q,a in PAIRS) + "\n"

def build_vocab(corpus, max_vocab):
    cnt=Counter(basic_tokenize(corpus))
    vocab=[]; seen=set()
    for t in SPECIAL:
        vocab.append(t); seen.add(t)
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789가나다라마바사아자차카타파하냥 .,!?:;+-=*/()[]{}_<>/\\'\"#@%&|^`~…♡♥★☆✨ㅠㅜㅋㅎ":
        if ch not in seen: vocab.append(ch); seen.add(ch)
    for tok,_ in cnt.most_common(max_vocab):
        if tok not in seen:
            vocab.append(tok); seen.add(tok)
            if len(vocab)>=max_vocab: break
    return vocab

class Tokenizer:
    def __init__(self, vocab):
        self.vocab=list(vocab); self.stoi={t:i for i,t in enumerate(self.vocab)}; self.itos={i:t for i,t in enumerate(self.vocab)}
        self.unk=self.stoi.get("<unk>",1)
    def encode(self, text):
        out=[]
        for tok in basic_tokenize(text):
            if tok in self.stoi: out.append(self.stoi[tok])
            else:
                for ch in tok:
                    out.append(self.stoi.get(ch,self.unk))
        return out
    def decode(self, ids):
        return "".join(self.itos.get(int(i),"") for i in ids)

def make_tokenizer_from_ckpt_or_data(ckpt=None):
    if ckpt and "vocab" in ckpt:
        return Tokenizer(ckpt["vocab"])
    return Tokenizer(build_vocab(train_corpus(), CONFIG.get("max_vocab",12000)))

# ---------- model ----------
def ensure_torch_or_exit():
    if not TORCH_AVAILABLE:
        msg=f"Torch 사용 불가: {TORCH_IMPORT_ERROR}"
        if args.torch_only: raise SystemExit(msg)
        return False
    return True

if TORCH_AVAILABLE:
    class Head(nn.Module):
        def __init__(self, n_embd, head_size, block_size, dropout):
            super().__init__(); self.key=nn.Linear(n_embd,head_size,bias=False); self.query=nn.Linear(n_embd,head_size,bias=False); self.value=nn.Linear(n_embd,head_size,bias=False)
            self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size))); self.dropout=nn.Dropout(dropout)
        def forward(self,x):
            B,T,C=x.shape; k=self.key(x); q=self.query(x); wei=q@k.transpose(-2,-1)*(k.shape[-1]**-0.5); wei=wei.masked_fill(self.tril[:T,:T]==0, float('-inf')); wei=F.softmax(wei,dim=-1); wei=self.dropout(wei); return wei@self.value(x)
    class MultiHeadAttention(nn.Module):
        def __init__(self,n_embd,n_head,block_size,dropout):
            super().__init__(); hs=n_embd//n_head; self.heads=nn.ModuleList([Head(n_embd,hs,block_size,dropout) for _ in range(n_head)]); self.proj=nn.Linear(n_embd,n_embd); self.dropout=nn.Dropout(dropout)
        def forward(self,x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))
    class FeedForward(nn.Module):
        def __init__(self,n_embd,dropout): super().__init__(); self.net=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.GELU(),nn.Linear(4*n_embd,n_embd),nn.Dropout(dropout))
        def forward(self,x): return self.net(x)
    class Block(nn.Module):
        def __init__(self,cfg): super().__init__(); self.sa=MultiHeadAttention(cfg['n_embd'],cfg['n_head'],cfg['block_size'],cfg['dropout']); self.ff=FeedForward(cfg['n_embd'],cfg['dropout']); self.ln1=nn.LayerNorm(cfg['n_embd']); self.ln2=nn.LayerNorm(cfg['n_embd'])
        def forward(self,x): x=x+self.sa(self.ln1(x)); x=x+self.ff(self.ln2(x)); return x
    class HGAITokenModel(nn.Module):
        def __init__(self,vocab_size,cfg):
            super().__init__(); self.cfg=cfg; self.token_embedding=nn.Embedding(vocab_size,cfg['n_embd']); self.position_embedding=nn.Embedding(cfg['block_size'],cfg['n_embd']); self.blocks=nn.Sequential(*[Block(cfg) for _ in range(cfg['n_layer'])]); self.ln_f=nn.LayerNorm(cfg['n_embd']); self.head=nn.Linear(cfg['n_embd'],vocab_size,bias=False); self.head.weight=self.token_embedding.weight
        def forward(self,idx,targets=None):
            B,T=idx.shape; tok=self.token_embedding(idx); pos=self.position_embedding(torch.arange(T,device=idx.device)); x=tok+pos; x=self.blocks(x); x=self.ln_f(x); logits=self.head(x)
            if targets is None: return logits,None
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), targets.reshape(-1))
            return logits,loss

def get_device():
    if not TORCH_AVAILABLE: return "cpu"
    if args.device=="cuda" and torch.cuda.is_available(): return "cuda"
    if args.device=="auto" and torch.cuda.is_available(): return "cuda"
    return "cpu"

def count_params(model):
    seen=set(); total=0
    for p in model.parameters():
        if id(p) not in seen: total+=p.numel(); seen.add(id(p))
    return total

def load_ckpt():
    if not MODEL_FILE.exists() or not TORCH_AVAILABLE: return None
    try:
        try: return torch.load(MODEL_FILE, map_location="cpu")
        except TypeError: return torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    except Exception as e:
        status("모델 로드 실패:", e); return None

def compatible(ckpt):
    return isinstance(ckpt,dict) and ckpt.get("version") == "v6.2-hybrid-token" and ckpt.get("pair_hash") == PAIR_HASH and ckpt.get("preset") == args.preset

def train_model():
    if not ensure_torch_or_exit(): return None,None,"cpu"
    device=get_device(); tok=make_tokenizer_from_ckpt_or_data(None); ids=tok.encode(train_corpus())
    data=torch.tensor(ids,dtype=torch.long)
    if len(data) < CONFIG['block_size']+2: raise RuntimeError("dataset too small")
    model=HGAITokenModel(len(tok.vocab),CONFIG).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr)
    status(f"학습 시작 v6 preset={args.preset} params={count_params(model):,} vocab={len(tok.vocab):,} tokens={len(data):,} device={device} steps={args.steps}")
    model.train()
    for step in range(args.steps):
        ix=torch.randint(0, len(data)-CONFIG['block_size']-1, (CONFIG['batch_size'],))
        xb=torch.stack([data[i:i+CONFIG['block_size']] for i in ix]).to(device)
        yb=torch.stack([data[i+1:i+CONFIG['block_size']+1] for i in ix]).to(device)
        _,loss=model(xb,yb); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step % max(1,args.steps//20)==0 or step==args.steps-1: status(f"step {step:5d} loss {loss.item():.4f}")
    ckpt={"version":"v6.2-hybrid-token","model":model.state_dict(),"vocab":tok.vocab,"config":CONFIG,"pair_hash":PAIR_HASH,"preset":args.preset,"params":count_params(model)}
    torch.save(ckpt, MODEL_FILE); status("학습 완료! 저장됨:", MODEL_FILE)
    model.eval(); return model,tok,device

def load_model_runtime():
    if args.lite or args.generation_mode=="off" or not TORCH_AVAILABLE: return None,None,"cpu"
    ckpt=load_ckpt()
    if ckpt and compatible(ckpt):
        tok=make_tokenizer_from_ckpt_or_data(ckpt); device=get_device(); model=HGAITokenModel(len(tok.vocab),ckpt.get("config",CONFIG)).to(device); model.load_state_dict(ckpt["model"]); model.eval(); status("저장된 v6 Torch 모델 불러옴:", MODEL_FILE); return model,tok,device
    if args.auto_train or args.retrain:
        return train_model()
    status("호환 v6 모델 없음. 검색+룰 모드로 실행함. 학습하려면 --retrain 또는 --auto-train 사용")
    if args.torch_only: raise SystemExit("호환 v6 모델 없음")
    return None,None,"cpu"

@torch.no_grad() if TORCH_AVAILABLE else (lambda f:f)
def torch_generate(model,tok,device,user_text):
    if model is None or tok is None: return None
    prompt=f"<|user|> {clean_text(user_text)}\n<|assistant|> "
    idx=torch.tensor([tok.encode(prompt)[-CONFIG['block_size']:]], dtype=torch.long, device=device)
    for _ in range(args.max_new_tokens):
        logits,_=model(idx[:,-CONFIG['block_size']:]); logits=logits[:,-1,:]/max(0.05,args.temperature)
        if args.top_k and args.top_k>0:
            v,_=torch.topk(logits, min(args.top_k, logits.size(-1))); logits[logits<v[:,[-1]]] = -float('inf')
        probs=F.softmax(logits,dim=-1); nxt=torch.multinomial(probs,1); idx=torch.cat([idx,nxt],dim=1)
        piece=tok.decode([nxt.item()])
        if piece == "<|end|>": break
    out=tok.decode(idx[0].tolist())
    if "<|assistant|>" in out: out=out.split("<|assistant|>",1)[1]
    out=out.split("<|end|>",1)[0]
    out=out.replace("<|user|>","").replace("<unk>","").strip()
    out=fix_nyang(out)
    return out.split("\n")[0].strip()

def quality_ok(ans,user):
    if not ans or len(ans) < 3 or len(ans) > 160: return False
    if ans.startswith("계산하면") and not is_math_query(user): return False
    if ans.count("냥") > 6: return False
    if re.search(r"(.{1,4})\1\1\1", ans): return False
    if len(set(ans.replace(" ",""))) < max(3, len(ans.replace(" ",""))//8): return False
    # For knowledge questions, avoid answer that contains a different strong term and not the asked strong term.
    tu=words_for_terms(user); ta=words_for_terms(ans)
    strong_u={x for x in tu if x in STRONG_TERMS or re.fullmatch(r"[a-zA-Z0-9_+#.]{2,}",x)}
    strong_a={x for x in ta if x in STRONG_TERMS or re.fullmatch(r"[a-zA-Z0-9_+#.]{2,}",x)}
    if any(k in user for k in ["뭐야","뜻","설명","대해","알려"]) and strong_u and strong_a and not (strong_u & strong_a):
        # allow identity answers
        if not ({"hgai","hgcompany","doubleduo"} & (strong_u|strong_a)):
            return False
    return True

MODEL=None; TOKENIZER=None; DEVICE="cpu"
if args.retrain:
    MODEL,TOKENIZER,DEVICE=train_model()
else:
    MODEL,TOKENIZER,DEVICE=load_model_runtime()

def reply(user):
    user=clean_text(user)
    # 1) 안전/기본 룰과 명확한 계산은 항상 먼저 처리
    for fn in (rule_reply, calc_reply, exact_reply, protected_reply):
        r=fn(user)
        if r: return fix_nyang(r)
    # 2) 검색 전용/생성 끔
    if args.reply_mode == "search-only" or args.generation_mode == "off":
        return fix_nyang(fuzzy_reply(user) or "잘 모르겠다냥 새로 배우고 싶다냥 :3")
    # 3) 지식 질문이나 강한 키워드가 있으면 검색을 먼저 강하게 시도해서 용어 섞임 방지
    is_knowledge = any(x in user for x in ["뭐야","뭐임","뜻","설명","알려","대해","란"])
    has_strong = bool(words_for_terms(user) & set(STRONG_TERMS))
    if is_knowledge or has_strong:
        r=fuzzy_reply(user, threshold=min(args.fuzzy_threshold, 0.62))
        if r: return fix_nyang(r)
    # 4) Torch 생성 모드
    if args.reply_mode == "torch-only":
        g=torch_generate(MODEL,TOKENIZER,DEVICE,user)
        return fix_nyang(g if quality_ok(g,user) else "잘 모르겠다냥 더 학습이 필요하다냥")
    if args.reply_mode == "torch-first" or args.generation_mode == "creative":
        g=torch_generate(MODEL,TOKENIZER,DEVICE,user)
        if quality_ok(g,user): return fix_nyang(g)
        r=fuzzy_reply(user)
        return fix_nyang(r or "잘 모르겠다냥 새로 배우고 싶다냥 :3")
    # 5) safe hybrid: 검색 → 생성 → fallback
    r=fuzzy_reply(user)
    if r: return fix_nyang(r)
    g=torch_generate(MODEL,TOKENIZER,DEVICE,user)
    if quality_ok(g,user): return fix_nyang(g)
    return "잘 모르겠다냥 새로 배우고 싶다냥 :3"

def validate_dataset():
    bad=[]
    for q,a in PAIRS:
        if not q or not a or " 냥" in a: bad.append((q,a,"spacing"))
        tq=words_for_terms(q); ta=words_for_terms(a)
        # only flag, don't fail too hard
        if any(x in q for x in ["뭐야","뜻","설명"]) and tq and ta and (tq & set(STRONG_TERMS)) and (ta & set(STRONG_TERMS)) and not ((tq&ta) or ({"hgai","hgcompany","doubleduo"}&(tq|ta))):
            bad.append((q,a,"possible mismatch"))
    print(f"pairs={len(PAIRS):,} unique_questions={len(set(q_norms)):,} possible_issues={len(bad):,}")
    for row in bad[:20]: print("WARN", row[2], row[0], "=>", row[1])
    return len(bad)

if args.info:
    tok=make_tokenizer_from_ckpt_or_data(load_ckpt()) if TORCH_AVAILABLE else None
    if TORCH_AVAILABLE and tok:
        tmp=HGAITokenModel(len(tok.vocab), CONFIG)
        params=count_params(tmp)
        vocab=len(tok.vocab)
    else:
        params=0; vocab=0
    print(f"version: v6.2-hybrid-token")
    print(f"train pairs: {len(PAIRS):,}")
    print(f"train chars: {len(raw_train):,}")
    print(f"pair hash: {PAIR_HASH[:12]}")
    print(f"torch available: {TORCH_AVAILABLE}")
    print(f"model loaded: {MODEL is not None}")
    print(f"preset: {args.preset}")
    print(f"config: {CONFIG}")
    print(f"vocab size: {vocab:,}")
    print(f"params approx: {params:,}")
    raise SystemExit(0)

if args.validate:
    validate_dataset(); raise SystemExit(0)

if args.benchmark:
    tests=["안녕","너는 누구니","퍼리가 뭐야","너는 어디에서 만들어졌어?","HG Company가 뭐야","LLM이 뭐야","깃허브가 뭐야","C언어가 뭐야","동물 이름 아무거나 말해봐","10000×10000"]
    for t in tests: print("Q:",t,"\nA:",reply(t),"\n")
    raise SystemExit(0)

if args.once is not None:
    print(reply(args.once)); raise SystemExit(0)

print("\nHGAI v6.2 Hybrid-Token 대화 시작! 종료하려면 exit 입력")
print(f"pairs={len(PAIRS):,}, torch={TORCH_AVAILABLE and MODEL is not None and not args.lite}, preset={args.preset}, mode={args.generation_mode}/{args.reply_mode}")
while True:
    try: user=input("너: ").strip()
    except (EOFError,KeyboardInterrupt): print(); break
    if user.lower() in ["exit","quit","종료"]: break
    print("HG AI:", reply(user))
