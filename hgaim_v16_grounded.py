#!/usr/bin/env python3
"""HGAIM v16 factual guardrail: safe offline game Q/A before neural generation.

Game facts come from game_facts_v13.json (reviewed records with source paths).
This is an in-process compact source registry, NOT web search or a pretrained
model claiming to remember held-out text. High-confidence intent rules answer
verbatim from those records; unsupported factual requests are not fabricated.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional

from nyang_style_v14 import nyangify

GAME_ALIASES = {
    "Casualties: Unknown": (r"casualties\s*:\s*unknown", r"\bc\s*:\s*u\b", r"\bsaw[-_ ]?0[0-9]\b", r"\bosprey\b", r"노란\s*(?:피|혈액)", r"\byellow blood\b"),
    "Minecraft": (r"마인크래프트", r"\bminecraft\b"),
    "Roblox": (r"로블록스", r"\broblox\b"),
    "ADOFAI": (r"얼불춤", r"\badofai\b", r"a dance of fire and ice"),
    "Gunsaw": (r"\bgunsaw\b"),
    "Changed": (r"\bchanged\b", r"\bpuro\b", r"\bdr\.\s*k\b"),
    "DevilConnection": (r"\bdevil\s*connection\b",),
    "Terraria": (r"테라리아", r"\bterraria\b"),
}
FACT_QUESTION = re.compile(
    r"누구|누가|뭐|무엇|어떤|알려|설명|목적|특징|이름|방법|언어|관계|왜|정체|"
    r"\bwho\b|\bwhat\b|\bwhich\b|\bwhy\b|\bhow\b|\bdescribe\b|\bexplain\b|\btell me\b|\bcan you\b|\bis\b|\bare\b",
    re.I,
)
STYLE_UNKNOWN = {
    "ko": "지금 가진 게임 원본 자료에는 그 내용을 확인할 근거가 없다냥. 정확한 설정이나 출처를 알려주면 확인할 수 있다냥.",
    "en": "I don't have enough verified source material to confirm that game detail, meow. Please provide an official reference.",
}
# Each rule requires both a game anchor and the exact fact intent. Unknown
# questions about a related item deliberately do not return an unrelated fact.
INTENT_RULES = [
    ("Casualties: Unknown", "lore-osprey", r"\bosprey\b", r"옮기|운송|이동|순간이동|건물|목적|설계|어떤|뭐|what|move|transport|building|teleport|purpose|designed"),
    ("Casualties: Unknown", "lore-saw03", r"\bsaw[-_ ]?0?3\b", r"털|색|외형|모습|특징|fur|color|colour|appearance|look"),
    ("Casualties: Unknown", "lore-yellowblood", r"노란\s*(?:피|혈액)|yellow\s*blood", r"방사선|회복|재생|복구|치유|특징|저항|radiation|heal|repair|regenerat|remarkable|special"),
    ("Gunsaw", "developer", r"\bgunsaw\b", r"개발자|제작자|만든\s*사람|누가\s*만들|who\s*(?:made|created|developed)|developer|creator|developed\s+by"),
    ("Roblox", "scripting", r"로블록스|\broblox\b", r"스크립트|프로그래밍\s*언어|개발\s*언어|\bluau\b|scripting|programming\s+language|script\s+language"),
    ("ADOFAI", "mechanic", r"얼불춤|\badofai\b|a dance of fire and ice", r"조작|박자|행성|천체|공전|리듬|타이밍|눌러|입력|mechanic|one.button|orbital|planet|timing|rhythm|beat|control|play"),
    ("Changed", "puro", r"\bpuro\b", r"누구|뭐|어떤|인물|캐릭터|정체|who|what|role|character|colin|콜린|describe"),
    ("DevilConnection", "identity", r"\bdevil\s*connection\b", r"세계관|공식|설정|뭐|무슨|어떤|게임|소개|정체|lore|official|setting|game|what|about"),
    ("Minecraft", "redstone", r"마인크래프트|\bminecraft\b", r"레드스톤|redstone"),
    ("Terraria", "genre", r"테라리아|\bterraria\b", r"장르|어떤\s*게임|무슨\s*게임|genre|what kind|what is"),
]

def language_of(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"

def detect_games(text: str):
    return [game for game,patterns in GAME_ALIASES.items() if any(re.search(p,text,re.I) for p in patterns)]

@lru_cache(maxsize=4)
def load_facts(path: str = "game_facts_v13.json"):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data,list): raise ValueError("Expected fact list")
    result={}
    for item in data:
        key=(item["game"],item["id"])
        if key in result: raise ValueError(f"Duplicate fact: {key}")
        for attr in ("q_ko","a_ko","q_en","a_en","source"):
            if not isinstance(item.get(attr),str) or not item[attr].strip():
                raise ValueError(f"Malformed fact: {key}")
        result[key]=item
    return result

def _normalized(text):
    return re.sub(r"[^a-z0-9가-힣]+","",text.casefold())

def find_fact(question: str, *, facts_path: str = "game_facts_v13.json"):
    if not question.strip(): return None
    registry=load_facts(facts_path)
    games=detect_games(question)
    for game,identifier,subject,intent in INTENT_RULES:
        if game not in games or not re.search(subject,question,re.I) or not re.search(intent,question,re.I):
            continue
        item=registry.get((game,identifier))
        if item:return item
    # For other reviewed game facts, accept only almost-identical
    # questions, with an explicit game anchor; never promote weak matches.
    if len(games)!=1:return None
    game=games[0]
    lang=language_of(question)
    target=_normalized(question)
    scored=[]
    for (name,_),item in registry.items():
        if name!=game:continue
        question_text=_normalized(item["q_"+lang])
        ratio=SequenceMatcher(None,target,question_text,autojunk=False).ratio()
        scored.append((ratio,item))
    scored.sort(key=lambda pair:pair[0],reverse=True)
    if scored and scored[0][0]>=.88 and (len(scored)==1 or scored[0][0]-scored[1][0]>=.10):
        return scored[0][1]
    return None

def is_game_fact_request(question: str) -> bool:
    return bool(detect_games(question) and FACT_QUESTION.search(question))

def grounded_answer(question: str, *, facts_path: str = "game_facts_v13.json", unknown_policy: str = "abstain") -> Optional[dict]:
    """Return an answer with provenance or None for neural-only questions.

    Returns response metadata with route='source' or 'abstain'. The code does
    not claim an unsourced completion is verified.
    """
    if unknown_policy not in {"abstain","neural"}:
        raise ValueError("unknown_policy must be abstain or neural")
    lang=language_of(question)
    match=find_fact(question,facts_path=facts_path)
    if match:
        return {
          "route":"source", "game":match["game"],"fact_id":match["id"],
          "source":match["source"],"language":lang,
          "answer":nyangify(match["a_"+lang],lang),
        }
    if is_game_fact_request(question) and unknown_policy=="abstain":
        return {"route":"abstain","game":detect_games(question)[0],
                "source":None,"language":lang,"answer":STYLE_UNKNOWN[lang]}
    return None
