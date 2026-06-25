#!/usr/bin/env python3
# HGAI MidTorch v5-clean
# Korean cat-tone mini chatbot
# - Phone/PC middle-grade Transformer presets
# - Torch training when torch is available
# - Lite exact/fuzzy/search mode when torch is unavailable
# - Safe calculator and Korean arithmetic helper
# - Automatic "냥" spacing cleanup

import argparse
import ast
import difflib
import hashlib
import math
import os
import random
import re
import sys
from pathlib import Path

# Torch는 --lite 실행을 빠르게 하기 위해 필요할 때만 불러온다.
torch = None
nn = None
F = None
TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = BASE_DIR / "train.txt"
DEFAULT_MODEL_FILE = BASE_DIR / "hgai_model.pth"

PRESETS = {
    "tiny":  {"block_size": 48,  "batch_size": 8,  "n_embd": 64,  "n_head": 4, "n_layer": 1, "dropout": 0.10},
    "phone": {"block_size": 64,  "batch_size": 8,  "n_embd": 96,  "n_head": 4, "n_layer": 2, "dropout": 0.10},
    "mid":   {"block_size": 96,  "batch_size": 12, "n_embd": 160, "n_head": 5, "n_layer": 3, "dropout": 0.12},
    "mid-safe": {"block_size": 96,  "batch_size": 12, "n_embd": 160, "n_head": 5, "n_layer": 3, "dropout": 0.12},
    "mid-plus": {"block_size": 128, "batch_size": 10, "n_embd": 192, "n_head": 6, "n_layer": 4, "dropout": 0.12},
    "pc":    {"block_size": 128, "batch_size": 20, "n_embd": 224, "n_head": 7, "n_layer": 4, "dropout": 0.12},
}

parser = argparse.ArgumentParser(description="HGAI MidTorch v5-clean Korean cat-tone chatbot")
parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
parser.add_argument("--preset", default="mid-safe", choices=sorted(PRESETS.keys()), help="모델 크기: tiny/phone/mid/mid-safe/mid-plus/pc")
parser.add_argument("--retrain", action="store_true", help="기존 모델을 무시하고 다시 학습")
parser.add_argument("--auto-train", action="store_true", help="호환 모델이 없으면 자동 학습")
parser.add_argument("--lite", action="store_true", help="Torch를 쓰지 않고 검색/룰 모드만 사용")
parser.add_argument("--torch-only", action="store_true", help="Torch가 없으면 fallback 없이 종료")
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--threads", type=int, default=0, help="CPU 스레드 수. 0이면 기본값")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--once", type=str, default=None, help="대화창 없이 한 문장만 답변")
parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top-k", type=int, default=16)
parser.add_argument("--max-new-tokens", type=int, default=70)
parser.add_argument("--fuzzy-threshold", type=float, default=0.62)
parser.add_argument("--no-fuzzy", action="store_true")
parser.add_argument("--info", action="store_true", help="데이터/모델 정보만 출력")
parser.add_argument("--generation-mode", default="off", choices=["safe", "creative", "off"], help="Torch 생성 사용 방식. 폰 속도 때문에 기본값은 off")
parser.add_argument("--install-help", action="store_true", help="Termux Torch 설치 힌트 출력")
args = parser.parse_args()

CONFIG = dict(PRESETS[args.preset])


def load_torch():
    global torch, nn, F, TORCH_AVAILABLE, TORCH_IMPORT_ERROR
    if TORCH_AVAILABLE:
        return True
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F
        TORCH_AVAILABLE = True
        TORCH_IMPORT_ERROR = None
        return True
    except Exception as e:
        torch = None
        nn = None
        F = None
        TORCH_AVAILABLE = False
        TORCH_IMPORT_ERROR = e
        return False


def status(*items):
    if args.once is None:
        print(*items)


def print_install_help():
    print("""
[Termux Torch 힌트]
1) 먼저 Termux 패키지를 확인해봐:
   pkg update
   pkg install python
   pkg search torch
   pkg install python-torch
   python -c "import torch; print(torch.__version__)"

2) python-torch가 없거나 import가 깨지면 proot Ubuntu 쪽이 더 나을 수 있어:
   pkg install proot-distro
   proot-distro install ubuntu
   proot-distro login ubuntu
   apt update
   apt install python3 python3-pip
   python3 -m pip install torch

3) 그래도 실패하면 Lite 모드는 항상 가능해:
   python run_hgai.py --lite

폰 학습 추천:
   python run_hgai.py --retrain --preset phone --steps 300 --threads 2
PC/노트북 추천:
   python run_hgai.py --retrain --preset mid --steps 2000
""".strip())

if args.install_help:
    print_install_help()
    raise SystemExit(0)

random.seed(args.seed)
# 기본 대화는 폰에서 빨리 켜지도록 Torch import를 건너뛴다.
# 학습/모델 생성/정보 확인이 필요할 때만 Torch를 불러온다.
if args.torch_only or args.retrain or args.auto_train or args.generation_mode != "off" or (args.info and not args.lite):
    load_torch()
if TORCH_AVAILABLE:
    torch.manual_seed(args.seed)
    if args.threads and args.threads > 0:
        try:
            torch.set_num_threads(args.threads)
        except Exception:
            pass

TRAIN_FILE = Path(args.train_file)
MODEL_FILE = Path(args.model_file)
if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"학습 파일을 찾을 수 없음: {TRAIN_FILE}")

raw_text = TRAIN_FILE.read_text(encoding="utf-8")

EXTRA_CHARS = """
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ
 .,!?~:;=+-_()[]{}<>/\\'\"*#@%&|^`₩…♡♥★☆✨
"""


def fix_nyang_spacing(text: str) -> str:
    text = str(text).strip()
    text = text.replace("네르지 마세요", "깨물지는 마라냥")
    text = text.replace("아야하지 마라 냥", "아야하니까 살살 해라냥")
    text = re.sub(r"\s+냥(?!냥)", "냥", text)
    text = re.sub(r"냥\s+냥", "냥냥", text)
    text = text.replace("냥냥 고로롱", "냥냥고로롱")
    text = text.replace("냥냥 고롱", "냥냥고롱")
    text = text.replace("언 어", "언어").replace("파이 썬", "파이썬").replace("C 언어", "C언어")
    text = re.sub(r"(냥){4,}", "냥냥", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_user_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def match_key(text: str) -> str:
    text = clean_user_text(text).lower()
    text = re.sub(r"[\s\.\!?~,;:'\"`\[\]\(\){}<>/\\|_+=\-^…ㅋㅎㅠㅜ]", "", text)
    return text


def parse_pairs(text: str):
    pairs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        q, a = line.split("=", 1)
        q = clean_user_text(q)
        a = fix_nyang_spacing(a)
        if q and a:
            pairs.append((q, a))
    return pairs

PAIRS = parse_pairs(raw_text)
PAIR_HASH = hashlib.sha256("\n".join(f"{q}={a}" for q, a in PAIRS).encode("utf-8")).hexdigest()

by_exact = {}
for q, a in PAIRS:
    by_exact.setdefault(match_key(q), []).append(a)
questions = [q for q, _ in PAIRS]
keys = [match_key(q) for q, _ in PAIRS]


def dataset_text():
    return "\n".join(f"{q}={a}" for q, a in PAIRS) + "\n"


def stable_choice(items, seed_text=""):
    if not items:
        return None
    h = int(hashlib.md5((seed_text + "|" + str(len(items))).encode("utf-8")).hexdigest(), 16)
    return items[h % len(items)]

# ------------------------- safe calculator -------------------------
ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
ALLOWED_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil, "sin": math.sin, "cos": math.cos, "tan": math.tan,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("제곱이 너무 크다냥")
        result = ALLOWED_BINOPS[type(node.op)](left, right)
        if abs(result) > 10**12:
            raise ValueError("결과가 너무 크다냥")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
        return ALLOWED_UNARY[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS:
        args = [_safe_eval_node(a) for a in node.args]
        if len(args) > 2:
            raise ValueError("함수 인자가 너무 많다냥")
        return ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError("허용되지 않은 계산식이다냥")


def format_number(x):
    if isinstance(x, float):
        if math.isfinite(x) and abs(x - round(x)) < 1e-10:
            return str(int(round(x)))
        return (f"{x:.10f}".rstrip("0").rstrip("."))
    return str(x)


def korean_to_expr(text):
    t = text.lower()
    t = t.replace("×", "*").replace("÷", "/").replace("^", "**")
    t = re.sub(r"(더하기|플러스|더한)", "+", t)
    t = re.sub(r"(빼기|마이너스|뺀)", "-", t)
    t = re.sub(r"(곱하기|곱한|곱해|x)", "*", t)
    t = re.sub(r"(나누기|나눈|나눠)", "/", t)
    t = re.sub(r"(제곱)", "**2", t)
    t = re.sub(r"루트\s*([0-9\.]+)", r"sqrt(\1)", t)
    # Keep only math-ish characters and function letters.
    candidates = re.findall(r"[0-9\.\+\-\*/%\(\)\s]+|sqrt\([0-9\.]+\)|abs\([0-9\.\-]+\)|round\([0-9\.\-]+\)", t)
    expr = "".join(candidates).strip()
    expr = re.sub(r"\s+", "", expr)
    return expr


def maybe_calculate(user_text):
    text = clean_user_text(user_text)
    compact = text.replace(" ", "")
    nums_for_stats = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if ("평균" in text or "평균값" in text) and len(nums_for_stats) >= 2:
        return f"평균은 {format_number(sum(nums_for_stats)/len(nums_for_stats))}다냥 :3"
    if ("최대" in text or "제일 큰" in text) and len(nums_for_stats) >= 2:
        return f"가장 큰 값은 {format_number(max(nums_for_stats))}다냥 :3"
    if ("최소" in text or "제일 작은" in text) and len(nums_for_stats) >= 2:
        return f"가장 작은 값은 {format_number(min(nums_for_stats))}다냥 :3"
    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:의|에서)\s*(\d+(?:\.\d+)?)\s*%", text)
    if m_pct:
        a, p = float(m_pct.group(1)), float(m_pct.group(2))
        return f"계산하면 {format_number(a*p/100)}다냥 :3"
    m_pct2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:은|는)\s*(\d+(?:\.\d+)?)\s*의\s*몇\s*퍼센트", text)
    if m_pct2:
        a, b = float(m_pct2.group(1)), float(m_pct2.group(2))
        if b == 0:
            return "0을 기준으로 퍼센트를 구할 수 없다냥 >_<"
        return f"{format_number(a)}는 {format_number(b)}의 {format_number(a/b*100)}%다냥 :3"
    m_fact = re.search(r"(\d+)\s*!", compact) or re.search(r"(\d+)\s*팩토리얼", text)
    if m_fact:
        n = int(m_fact.group(1))
        if n > 20:
            return "팩토리얼이 너무 커진다냥 20 이하로 해줘라냥"
        return f"계산하면 {math.factorial(n)}다냥 :3"

    m = re.search(r"(\d+)\s*부터\s*(\d+)\s*까지\s*(더해|합|더하기)", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if abs(b - a) > 1000000:
            return "범위가 너무 크다냥 백만 칸 안쪽으로 해줘라냥"
        lo, hi = (a, b) if a <= b else (b, a)
        s = (lo + hi) * (hi - lo + 1) // 2
        return f"{a}부터 {b}까지 더하면 {s}다냥 :3"
    m = re.search(r"(\d+)\s*단", text)
    if "구구단" in text or m:
        n = int(m.group(1)) if m else None
        if n is None:
            return "몇 단인지 말해주면 구구단을 불러주겠다냥 예: 7단냥"
        if n < 0 or n > 99:
            return "너무 큰 단은 냥손으로 세기 힘들다냥 0부터 99단까지만 해보자냥"
        return f"{n}단: " + ", ".join(f"{n}x{i}={n*i}" for i in range(1, 10)) + "냥"
    if not any(x in compact for x in ["계산", "+", "-", "*", "/", "×", "÷", "%", "!", "더하기", "빼기", "곱하기", "나누기", "루트", "제곱", "평균", "최대", "최소", "퍼센트", "팩토리얼"]):
        return None
    expr = korean_to_expr(text)
    if not expr or not re.search(r"\d", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree)
        return f"계산하면 {format_number(result)}다냥 :3"
    except ZeroDivisionError:
        return "0으로 나누면 우주 고양이도 곤란하다냥 >_<"
    except Exception as e:
        msg = str(e)
        if "냥" in msg:
            return msg
        return "계산식이 조금 헷갈린다냥 숫자랑 + - * / 로 다시 써줘라냥"

# ------------------------- rules/search -------------------------

def safety_reply(user_text):
    t = match_key(user_text)
    danger = ["죽고싶", "사라지고싶", "없어지고싶", "자해", "다치게하고싶", "살기싫", "위험한생각"]
    if any(x in t for x in danger):
        return "지금은 혼자 버티지 마라냥 가까운 사람이나 긴급 도움을 바로 불러라냥 주인의 안전이 제일 중요하다냥 ;w;"
    return None


def rule_reply(user_text):
    raw = clean_user_text(user_text)
    t = match_key(raw)

    # 너무 짧은 입력은 Transformer에 넘기지 않고 안전하게 처리한다.
    short_alias = {
        "ㅎㅇ": "하이냥 반갑다냥 :3",
        "hi": "Hi냥 한국어도 영어도 조금씩 받아준다냥 :3",
        "hello": "Hello냥 반갑다냥 ^w^",
        "ㄱ": "ㄱㄱ냥 준비됐다냥 무슨 말부터 해볼까냥",
        "ㅋ": "웃겼냐냥 나도 고로롱 웃겠다냥",
        "ㅠ": "울지 마라냥 내가 옆에 있겠다냥 ;w;",
    }
    if t in short_alias:
        return short_alias[t]

    if ("이름" in t or "누구" in t or "정체" in t) and ("너" in t or "너의" in raw or len(t) <= 8):
        return "나는 HGAI다냥 한국어로 대화하는 작은 고양이 AI다냥 :3"
    if ("아는" in t and "지식" in t) or "뭐할수있" in t or "할수있는것" in t:
        return "나는 감정 위로, 공부 응원, 기본 코딩 지식, 컴퓨터 용어, GitHub, Termux, 간단한 계산을 할 수 있다냥 :3"


    if any(x in t for x in ["어디서만들", "어디에서만들", "어디서개발", "어디에서개발", "누가만들", "개발자누구", "만든사람"]):
        return "나는 HG Company라는 곳에서 덥듀라는 사람에게 개발되었다냥"
    if "hgcompany" in t or "hg컴퍼니" in t:
        return "HG Company는 여러 IT 주제를 다루거나 특이한 거 개발도 하는 회사다냥"
    if "퍼리" in t or "furry" in t or "수인" in t:
        if "너" in t or "hgai" in t:
            return "나는 현실의 퍼리는 아니고 고양이 말투를 쓰는 HGAI다냥 :3"
        return "동물을 의인화한 캐릭터 또는 그런 것들에 끌리는 취향을 말한다냥"

    quick_knowledge = {
        "윈도우": "마이크로소프트가 만든 PC 운영체제다냥 프로그램 실행하고 파일 관리하는 데 많이 쓴다냥",
        "리눅스": "서버와 개발자가 많이 쓰는 자유로운 운영체제다냥 터미널이 강한 게 매력이다냥",
        "파이썬": "문법이 쉬운 프로그래밍 언어다냥 AI 자동화 웹 서버까지 다양하게 만들 수 있다냥",
        "c언어": "컴퓨터에 가까운 저수준 프로그래밍 언어다냥 빠르지만 포인터 때문에 조심해야 한다냥",
        "자바스크립트": "웹 브라우저에서 많이 쓰는 언어다냥 요즘은 서버도 만들 수 있다냥",
        "깃허브": "Git 저장소를 온라인에 올리고 협업하는 서비스다냥 Actions로 자동 학습도 가능하다냥",
        "github": "Git 저장소를 온라인에 올리고 협업하는 서비스다냥 Actions로 자동 학습도 가능하다냥",
        "termux": "안드로이드에서 리눅스 터미널처럼 명령을 실행하게 해주는 앱이다냥",
        "라즈베리파이": "작은 싱글보드 컴퓨터다냥 서버 봇 자동화 실험용으로 좋다냥",
        "트랜스포머": "문장 안의 관계를 attention으로 보는 AI 구조다냥 요즘 언어모델의 핵심이다냥",
        "파라미터": "모델이 학습하면서 조정하는 숫자들이다냥 많을수록 표현력은 커지지만 데이터와 연산도 더 필요하다냥",
        "퍼리": "동물을 의인화한 캐릭터 또는 그런 것들에 끌리는 취향을 말한다냥",
        "수인": "동물 특징이 있는 사람형 캐릭터를 말한다냥 퍼리 캐릭터와 비슷하게 쓰일 때가 있다냥",
        "hgcompany": "여러 IT 주제를 다루거나 특이한 거 개발도 하는 회사다냥",
    }
    if any(w in t for w in ["뭐야", "무엇", "설명", "알려", "뜻", "대해"]):
        for k, a in quick_knowledge.items():
            if match_key(k) in t:
                return a

    spicy = ["수위높", "매운맛", "세게말", "디스", "팩폭", "욕해", "개빡", "개짜증", "멘탈나갔", "킹받", "현타"]
    if any(x in t for x in spicy):
        return stable_choice([
            "살짝 매운맛만 간다냥 지금 주인 멘탈 거의 과열된 라즈베리파이다냥",
            "팩폭하자면 고민 100개보다 실행 1개가 더 세다냥 지금 3분만 움직여라냥",
            "욕 대신 냥냥펀치 날린다냥 퍽퍽! 이제 정신 차리자냥",
            "세상아 왜 이렇게 구리게 구냐냥 그래도 주인은 안 진다냥",
            "매운맛으로 말하면 지금 뇌가 탭 999개 열린 브라우저다냥 하나씩 닫아라냥",
        ], user_text)
    if "추천" in t and "노래" in t:
        return "신나는 Phonk나 NCS 스타일 음악 어떠냐냥 기분 전환에 좋다냥 :3"
    return None


def exact_reply(user_text):
    items = by_exact.get(match_key(user_text), [])
    if not items:
        return None
    # 정확히 배운 문장은 첫 번째 고품질 답변을 우선 사용해서 이상한 랜덤 답변을 줄인다.
    return items[0]


def fuzzy_reply(user_text):
    if args.no_fuzzy or not PAIRS:
        return None
    mk = match_key(user_text)
    best_i = -1
    best_score = 0.0
    for i, k in enumerate(keys):
        if not k:
            continue
        # exact substring gets a boost
        score = difflib.SequenceMatcher(None, mk, k).ratio()
        if mk in k or k in mk:
            score = max(score, 0.78)
        # token/char overlap boost
        sa, sb = set(mk), set(k)
        if sa and sb:
            j = len(sa & sb) / len(sa | sb)
            score = max(score, 0.55 * score + 0.45 * j)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i >= 0 and best_score >= args.fuzzy_threshold:
        q = questions[best_i]
        options = by_exact.get(match_key(q), [PAIRS[best_i][1]])
        return stable_choice(options, user_text)
    return None


def lite_reply(user_text):
    user_text = clean_user_text(user_text)
    for fn in (maybe_calculate, safety_reply, exact_reply, rule_reply, fuzzy_reply):
        r = fn(user_text)
        if r:
            return fix_nyang_spacing(r)
    return stable_choice([
        "잘 모르겠다냥 그래도 같이 생각해보자냥",
        "그건 아직 학습이 부족하다냥 train.txt에 추가하면 더 똑똑해진다냥",
        "흠냥 조금 더 자세히 말해주면 대답해보겠다냥 :3",
        "냥냥고로롱... 그 말은 새로 배우고 싶다냥",
    ], user_text)

# ------------------------- Transformer -------------------------
model = None
stoi = itos = chars = data = None


def build_vocab():
    global stoi, itos, chars, data
    text = dataset_text()
    chars = sorted(list(set(text + EXTRA_CHARS)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor(encode(text), dtype=torch.long)


def encode(s):
    space_id = stoi.get(" ", 0)
    return [stoi.get(c, space_id) for c in str(s)]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


def make_model_class():
    block_size = CONFIG["block_size"]
    n_embd = CONFIG["n_embd"]
    n_head = CONFIG["n_head"]
    n_layer = CONFIG["n_layer"]
    dropout = CONFIG["dropout"]

    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            return wei @ v

    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super().__init__()
            head_size = n_embd // n_head
            self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
            self.proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            x = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(x))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.sa = MultiHeadAttention()
            self.ffwd = FeedForward()
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class MiniHGAI(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, n_embd)
            self.position_embedding = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size)
        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok = self.token_embedding(idx)
            pos = self.position_embedding(torch.arange(T, device=idx.device))
            x = tok + pos
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)
            if targets is None:
                return logits, None
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))
            return logits, loss
    return MiniHGAI


def choose_device():
    if args.device == "cpu" or not TORCH_AVAILABLE:
        return "cpu"
    if args.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if args.device == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(device):
    block_size = CONFIG["block_size"]
    batch_size = CONFIG["batch_size"]
    if len(data) <= block_size + 1:
        raise ValueError("train.txt가 너무 짧다냥")
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def init_model():
    global model
    build_vocab()
    MiniHGAI = make_model_class()
    model = MiniHGAI(len(chars))
    return model


def ckpt_meta():
    return {"chars": chars, "config": CONFIG, "pair_hash": PAIR_HASH, "preset": args.preset}


def load_model_if_possible(device):
    global model
    if not MODEL_FILE.exists():
        return False
    try:
        ckpt = torch.load(MODEL_FILE, map_location=device)
        if ckpt.get("chars") != chars:
            status("저장 모델과 글자 목록이 달라서 다시 학습 필요")
            return False
        if ckpt.get("config") != CONFIG:
            status("저장 모델 preset/config가 달라서 다시 학습 필요")
            return False
        if ckpt.get("pair_hash") != PAIR_HASH:
            status("train.txt가 바뀌어서 다시 학습 권장")
            return False
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        status("저장된 Torch 모델 불러옴:", MODEL_FILE)
        return True
    except Exception as e:
        status("모델 불러오기 실패:", e)
        return False


def train_model(device):
    global model
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    status(f"학습 시작 preset={args.preset} params={count_params(model):,} device={device} steps={args.steps}")
    for step in range(args.steps):
        xb, yb = get_batch(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            status(f"step {step:5d} loss {loss.item():.4f}")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **ckpt_meta()}, MODEL_FILE)
    status("학습 완료! 저장됨:", MODEL_FILE)
    model.eval()


def torch_generate(user_text, device, max_new_tokens=None, temperature=None, top_k=None):
    if not TORCH_AVAILABLE or model is None:
        return None
    with torch.no_grad():
        max_new_tokens = max_new_tokens or args.max_new_tokens
        temperature = temperature or args.temperature
        top_k = args.top_k if top_k is None else top_k
        prompt = clean_user_text(user_text) + "="
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        block_size = CONFIG["block_size"]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            ch = decode([next_id.item()])
            if ch == "\n":
                break
        result = decode(idx[0].detach().cpu().tolist())
        reply = result[len(prompt):]
        reply = reply.split("\n")[0].split("=")[0]
        reply = reply.replace("사용자:", "").replace("AI:", "").replace("HG AI:", "").strip()
        reply = fix_nyang_spacing(reply)
        if not reply or len(reply) < 2:
            return None
        return reply




def generated_is_bad(reply: str) -> bool:
    if not reply:
        return True
    r = fix_nyang_spacing(reply)
    compact = match_key(r)
    if len(compact) < 4:
        return True
    # character-level model sometimes repeats syllables or makes broken spacing.
    if re.search(r"(.)\1{5,}", r):
        return True
    if r.count("냥") >= 5 and len(r) < 40:
        return True
    bad_bits = ["뭐야뭐야", "좋다좋다", "배터리 배터", "내는냥", "고 있 뭐야", "오리 버", "언 어", "파이썬에 대해 밥"]
    if any(x in r for x in bad_bits):
        return True
    # In safe mode, answers should look like HGAI: Korean-ish and usually cat-toned.
    if args.generation_mode == "safe" and "냥" not in r:
        return True
    hangul = len(re.findall(r"[가-힣]", r))
    if hangul < 2:
        return True
    return False

def generate_reply(user_text, device="cpu"):
    # Stable tools first, Transformer as guarded creative fallback.
    r = maybe_calculate(user_text) or safety_reply(user_text) or exact_reply(user_text) or rule_reply(user_text) or fuzzy_reply(user_text)
    if r:
        return fix_nyang_spacing(r)
    if args.generation_mode != "off" and TORCH_AVAILABLE and not args.lite and model is not None:
        temp = args.temperature if args.generation_mode == "creative" else min(args.temperature, 0.55)
        topk = args.top_k if args.generation_mode == "creative" else min(args.top_k, 8)
        tr = torch_generate(user_text, device, temperature=temp, top_k=topk)
        if tr and not generated_is_bad(tr):
            return fix_nyang_spacing(tr)
    return lite_reply(user_text)


def main():
    if args.info:
        print(f"train pairs: {len(PAIRS):,}")
        print(f"train chars: {len(dataset_text()):,}")
        print(f"vocab hash: {PAIR_HASH[:12]}")
        print(f"torch available: {TORCH_AVAILABLE}")
        if TORCH_AVAILABLE:
            init_model()
            print(f"preset: {args.preset}")
            print(f"config: {CONFIG}")
            print(f"params: {count_params(model):,}")
        return

    device = "cpu"
    if args.torch_only and not TORCH_AVAILABLE:
        print("Torch import 실패다냥")
        print("원인:", TORCH_IMPORT_ERROR)
        print_install_help()
        raise SystemExit(1)

    need_torch_runtime = args.retrain or args.auto_train or args.generation_mode != "off"
    use_torch = TORCH_AVAILABLE and not args.lite and need_torch_runtime
    if use_torch:
        device = choose_device()
        init_model()
        loaded = False
        if not args.retrain:
            loaded = load_model_if_possible(device)
        if args.retrain or (args.auto_train and not loaded):
            train_model(device)
        elif not loaded:
            status("호환 Torch 모델 없음. 지금은 Lite+룰 모드로 실행함. 학습하려면 --retrain 또는 --auto-train 사용")
    else:
        if not TORCH_AVAILABLE:
            status("Torch import 실패. Lite 모드로 실행함:", TORCH_IMPORT_ERROR)
            status("Torch 학습을 원하면 --install-help를 실행해봐")

    if args.once is not None:
        print(generate_reply(args.once, device))
        return

    print()
    print("HGAI MidTorch v5-clean 대화 시작! 종료하려면 exit 입력")
    print(f"pairs={len(PAIRS):,}, torch={TORCH_AVAILABLE and not args.lite}, preset={args.preset}")
    while True:
        try:
            user = input("너: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ["exit", "quit", "종료"]:
            break
        if not user:
            continue
        print("HG AI:", generate_reply(user, device))

if __name__ == "__main__":
    main()
