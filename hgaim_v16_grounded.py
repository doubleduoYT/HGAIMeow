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

MODULE_DIR = Path(__file__).resolve().parent

GAME_ALIASES = {
    "Casualties: Unknown": (r"casualties\s*:\s*unknown", r"(?<![A-Za-z0-9])c\s*:\s*u(?![A-Za-z0-9])", r"(?<![A-Za-z0-9])saw[-_ ]?0[0-9](?![A-Za-z0-9])", r"(?<![A-Za-z0-9])osprey(?![A-Za-z0-9])", r"노란\s*(?:피|혈액)", r"\byellow blood\b"),
    "Minecraft": (r"마인크래프트", r"(?<![A-Za-z0-9])minecraft(?![A-Za-z0-9])"),
    "Roblox": (r"로블록스", r"(?<![A-Za-z0-9])roblox(?![A-Za-z0-9])"),
    "ADOFAI": (r"얼불춤", r"(?<![A-Za-z0-9])adofai(?![A-Za-z0-9])", r"a dance of fire and ice"),
    "Gunsaw": (r"(?<![A-Za-z0-9])gunsaw(?![A-Za-z0-9])",),
    "Changed": (r"(?<![A-Za-z0-9])puro(?![A-Za-z0-9])", r"\bdr\.\s*k\b"),
    "DevilConnection": (r"(?<![A-Za-z0-9])devil\s*connection(?![A-Za-z0-9])",),
    "Terraria": (r"테라리아", r"(?<![A-Za-z0-9])terraria(?![A-Za-z0-9])"),
}
FACT_QUESTION = re.compile(
    r"누구|누가|뭐|무엇|어떤|알려|설명|목적|특징|이름|방법|언어|관계|왜|정체|언제|출시|연도|날짜|가격|얼마|몇|"
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
    ("Casualties: Unknown", "lore-osprey", r"(?<![A-Za-z0-9])osprey(?![A-Za-z0-9])", r"옮기|옮겨|옮겼|운송|이동|순간이동|건물|목적|설계|어떤|뭐|what|move|transport|building|teleport|purpose|designed"),
    ("Casualties: Unknown", "lore-saw03", r"(?<![A-Za-z0-9])saw[-_ ]?0?3(?![A-Za-z0-9])", r"털|색|외형|모습|특징|fur|color|colour|appearance|look"),
    ("Casualties: Unknown", "lore-yellowblood", r"노란\s*(?:피|혈액)|yellow\s*blood", r"방사선|회복|재생|복구|치유|특징|저항|radiation|heal|repair|regenerat|remarkable|special"),
    ("Gunsaw", "developer", r"(?<![A-Za-z0-9])gunsaw(?![A-Za-z0-9])", r"개발자|제작자|만든\s*사람|누가\s*만들|who\s*(?:made|created|developed)|developer|creator|developed\s+by"),
    ("Roblox", "scripting", r"로블록스|(?<![A-Za-z0-9])roblox(?![A-Za-z0-9])", r"스크립트|프로그래밍\s*언어|개발\s*언어|무슨\s*언어|어떤\s*언어|\bluau\b|scripting|programming\s+language|script\s+language|what\s+language"),
    ("ADOFAI", "mechanic", r"얼불춤|(?<![A-Za-z0-9])adofai(?![A-Za-z0-9])|a dance of fire and ice", r"조작|박자|행성|천체|공전|리듬|타이밍|눌러|입력|mechanic|one.button|orbital|planet|timing|rhythm|beat|control|play"),
    ("Changed", "puro", r"(?<![A-Za-z0-9])puro(?![A-Za-z0-9])", r"누구|뭐|어떤|인물|캐릭터|정체|who|what|role|character|colin|콜린|describe"),
    ("DevilConnection", "identity", r"(?<![A-Za-z0-9])devil\s*connection(?![A-Za-z0-9])", r"세계관|공식|설정|뭐|무슨|어떤|게임|소개|정체|lore|official|setting|game|what|about"),
    ("Minecraft", "redstone", r"마인크래프트|(?<![A-Za-z0-9])minecraft(?![A-Za-z0-9])", r"레드스톤|redstone"),
    ("Terraria", "genre", r"테라리아|(?<![A-Za-z0-9])terraria(?![A-Za-z0-9])", r"장르|어떤\s*게임|무슨\s*게임|genre|what kind|what is"),
]

def language_of(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"

def detect_games(text: str):
    # 'changed' is an ordinary English verb. Recognize the game only by a
    # title-cased proper noun or explicit game-specific context.
    found = [game for game, patterns in GAME_ALIASES.items()
             if any(re.search(pattern, text, re.I) for pattern in patterns)]
    named_changed = bool(re.search(r"(?<![A-Za-z0-9])Changed(?![A-Za-z0-9])", text))
    verb_changed = bool(re.search(r"\\bi\\s+changed\\b|\\bchanged\\s+(?:my|the|our|a|an)\\b", text, re.I))
    game_context = bool(re.search(r"\\b(?:the\\s+game|game|puro|colin)\\b|게임|주인공", text, re.I))
    if ((named_changed and (not verb_changed or game_context)) or
            (game_context and re.search(r"\\bchanged\\b", text, re.I))):
        if "Changed" not in found:
            found.append("Changed")
    return found

@lru_cache(maxsize=4)
def load_facts(path: str = "game_facts_v13.json"):
    source = Path(path)
    # The daemon may have an unrelated WorkingDirectory. Always anchor the
    # bundled default registry to this Python module, not the current shell.
    if path == "game_facts_v13.json":
        source = MODULE_DIR / source
    data=json.loads(source.read_text(encoding="utf-8"))
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
    if is_open_ended_request(question):
        return None
    registry=load_facts(facts_path)
    games=detect_games(question)
    # The existing Gunsaw relation card covers an explicit comparison of two
    # games. All other multi-game fact questions are left unresolved.
    if len(games) > 1:
        if (set(games) == {"Gunsaw", "Casualties: Unknown"} and
                re.search(r"관계|관련|공통|차이|연결|비슷|relation|related|linked|connect|similar|between|difference", question, re.I)):
            return registry.get(("Gunsaw", "relation"))
        return None
    # Ignore intent mismatches that would confidently return the wrong fact.
    excluded=re.search(
        r"성우|목소리|더빙|voice\s+actor|voice\s+of|발매일|출시일|release\s+date|price|가격|비용|나이|몇\s*살|생일|태어|\bage\b|\bbirthday\b|\bborn\b|"
        r"레벨\s*에디터.{0,18}(?:누가|만들|제작자|개발자)|level\s+editor.{0,28}(?:who|made|creator|developer)",
        question, re.I)
    # Mentioning a level editor is NOT evidence of who created that editor.
    editor_authorship = (
        bool(re.search(r"레벨\s*에디터|level\s+editor", question, re.I)) and
        bool(re.search(r"누가|만들|제작자|개발자|who|made|created|developer|creator", question, re.I))
    )
    if editor_authorship:
        return None
    for game,identifier,subject,intent in INTENT_RULES:
        if excluded or game not in games or not re.search(subject,question,re.I) or not re.search(intent,question,re.I):
            continue
        # The presence of an item name alone does not prove that a generic
        # description answers a request for specific construction instructions.
        if (identifier == "redstone" and
                re.search(r"회로도|회로\\s*설계|설치법|제작법|step.by.step|tutorial|wiring|배선", question, re.I)):
            continue
        item=registry.get((game,identifier))
        if item:return item
    # For other reviewed game facts, accept only almost-identical
    # questions, with an explicit game anchor; never promote weak matches.
    if excluded or len(games)!=1:return None
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

def is_open_ended_request(question: str) -> bool:
    return bool(re.search(
        r"추천|아이디어|뭐\s*만들|뭘\s*만들|만들면\s*좋|어떻게\s*만들|입문|연습|사용법|도와|"
        r"suggest|recommend|ideas?|how\s+(?:can|should)\s+i\s+(?:build|make|start)",
        question, re.I))

def is_game_fact_request(question: str) -> bool:
    return bool(detect_games(question) and
                not is_open_ended_request(question) and
                FACT_QUESTION.search(question))

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
