#!/usr/bin/env python3
"""Small offline follow-up resolver for HGAIM's existing v10 conversation history.

Only adds a prior game when the last user message has exactly one recognized
game, and the current question explicitly refers back to that subject.
"""
from __future__ import annotations
import re
from hgaim_v16_grounded import detect_games

FOLLOWUP = re.compile(
    r"그\s*(?:게임|사람|장비|개체|설정|건|거)|그럼|그리고|그러면|누가|개발자|어떤\s*언어|무슨\s*언어|"
    r"털\s*색|색깔|뭘\s*옮|누구야|\bwhat about\b|\bwho (?:made|developed|is)\b|\band what\b|\bwhat language\b",
    re.I,
)

def resolve_followup(question, history):
    """history: list of user/assistant role/content dictionaries. No persistent state."""
    if detect_games(question) or not history or not FOLLOWUP.search(question):
        return question
    prior = ""
    for message in reversed(history[-8:]):
        if message.get("role") == "user":
            candidate = str(message.get("content", "")).strip()
            if candidate and candidate != question:
                prior = candidate
                break
    games = detect_games(prior)
    if len(games) != 1:
        return question
    game = games[0]
    context_suffix = ""
    if game == "Casualties: Unknown":
        entity = re.search(
            r"(?<![A-Za-z0-9])osprey(?![A-Za-z0-9])|"
            r"(?<![A-Za-z0-9])saw[-_ ]?0?3(?![A-Za-z0-9])|"
            r"노란\s*(?:피|혈액)|yellow\s+blood", prior, re.I
        )
        if entity and re.search(r"그\s*(?:장비|개체|피|혈액)|\bit\b|\bthat\b", question, re.I):
            context_suffix = " " + entity.group(0)
    return game + context_suffix + " " + question
