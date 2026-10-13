#!/usr/bin/env python3
"""Conservative Korean/English cat-speech formatting for HGAIM v14.

This is a surface-form guarantee, not a substitute for language learning.
Never run it on quoted code blocks, commands, URLs, or data values by themselves.
"""
from __future__ import annotations

import re

SPECIAL = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
LINE = re.compile(r"(?<=[.!?。！？])(?=\s|$)")
HAS_KO = re.compile(r"[가-힣]")
HAS_EN = re.compile(r"[A-Za-z]")

ENDING_REWRITES = (
    ("있어요", "있다냥"), ("없어요", "없다냥"),
    ("이에요", "이다냥"), ("예요", "이다냥"),
    ("좋아요", "좋다냥"), ("맞아요", "맞다냥"),
    ("해요", "한다냥"), ("돼요", "된다냥"),
    ("있어", "있다냥"), ("없어", "없다냥"),
    ("좋아", "좋다냥"), ("맞아", "맞다냥"),
    ("아니야", "아니다냥"), ("이야", "이다냥"),
    ("거야", "거다냥"), ("싶어", "싶다냥"),
    ("했어", "했다냥"), ("했어요", "했다냥"),
    ("이야", "이다냥"), ("야", "다냥"),
    ("해", "한다냥"), ("돼", "된다냥"),
    ("됩니다", "된다냥"), ("합니다", "한다냥"),
    ("입니다", "이다냥"),
)


def _fix_clause(sentence: str, language: str) -> str:
    if not sentence.strip():
        return sentence
    leading = sentence[:len(sentence)-len(sentence.lstrip())]
    trailing = sentence[len(sentence.rstrip()):]
    body = sentence.strip()
    if not body or body.lstrip().startswith((">", "#", "|", "http://", "https://")):
        return sentence
    if language == "en":
        if not HAS_EN.search(body) or body.endswith((":", ";")):
            return sentence
        if re.search(r"\bmeow\b\W*$", body, flags=re.I):
            return sentence
        if len(body) < 8 or not re.search(r"[a-zA-Z]", body):
            return sentence
        match = re.search(r"([.!?。！？]+|[~]+)?$", body)
        punct = (match.group(0) if match else "") or ""
        head = body[:-len(punct)] if punct else body
        if not head.strip() or head.endswith((":", ";")):
            return sentence
        return leading + head.rstrip() + ", meow" + punct + trailing

    if not HAS_KO.search(body) or body.endswith((":", ";")):
        return sentence
    match = re.search(r"([.!?。！？~]+)$", body)
    punct = match.group(0) if match else ""
    head = body[:-len(punct)] if punct else body
    head = head.rstrip()
    if not head or head.endswith(("냥", "냥ㅋ", "냥ㅎ")):
        return sentence
    # If response ends in a closing quote, preserve the quoted source verbatim
    # and add tone outside it.
    if head.endswith(("'", '"', "”", "’", "」", "』")):
        return leading + head + "이다냥" + punct + trailing
    if not HAS_KO.search(head[-6:]):
        return sentence
    for original, replacement in ENDING_REWRITES:
        if head.endswith(original):
            head = head[:-len(original)] + replacement
            break
    else:
        head = head + "냥"
    return leading + head + punct + trailing


def nyangify(text: str, language: str = "auto") -> str:
    if not text:
        return text
    language = language.lower()
    if language == "auto":
        language = "ko" if HAS_KO.search(text) else "en"
    if language not in {"ko", "en"}:
        raise ValueError("language must be ko, en, or auto")
    # Treat fenced/inline code as immutable. Process individual sentences in prose.
    parts = SPECIAL.split(text)
    result = []
    for i, part in enumerate(parts):
        if i % 2:
            result.append(part)
            continue
        output = []
        for paragraph in part.splitlines(keepends=True):
            if not paragraph.strip():
                output.append(paragraph)
                continue
            m = re.search(r"(\r?\n)$", paragraph)
            ending = m.group(1) if m else ""
            payload = paragraph[:-len(ending)] if ending else paragraph
            # Preserve heading/list markup by editing only natural-language tail.
            clauses = LINE.split(payload)
            output.append("".join(_fix_clause(c, language) for c in clauses) + ending)
        result.append("".join(output))
    return "".join(result)


if __name__ == "__main__":
    for x in ("마인크래프트는 샌드박스 게임이야.", "확인해 보자.", "네, 맞아요.", "영어를 사용해.", "I like games."):
        print(x, "=>", nyangify(x))
