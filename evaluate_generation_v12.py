#!/usr/bin/env python3
"""Independent neural-only dialogue sampling and transparent surface-quality metrics."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
from hgai_core import HGAIEngine, FALLBACK_RESPONSE, FALLBACK_RESPONSE_EN

TESTS = [
 ("ko", "그림 파일을 편집했는데 열리지 않아", "다른 프로그램으로 열어보면 될까?"),
 ("ko", "이 게임에서 속도감을 더 주고 싶어", "카메라부터 바꾸는 게 좋을까?"),
 ("ko", "내 음악이 너무 비슷하게 들려", "새로운 악기를 넣으면 달라질까?"),
 ("ko", "수정한 코드를 실험하고 싶어", "안전하게 되돌리는 방법도 알려줘"),
 ("ko", "게임에서 적이 너무 멀리 보여", "조준하기 쉽게 만들려면?"),
 ("ko", "번역을 좀 더 부드럽게 바꾸고 싶어", "대사를 너무 길게 하지 않는 게 좋겠지?"),
 ("ko", "내 프로그램이 메모리를 계속 사용해", "어디서 누수가 생기는지 알 수 있을까?"),
 ("ko", "협업할 때 작업이 자꾸 충돌해", "다른 브랜치에서 작업해도 될까?"),
 ("ko", "어제 만든 기능이 오늘 동작하지 않아", "어떤 로그부터 봐야 할까?"),
 ("ko", "간단한 채팅 게임을 개발하고 있어", "플레이어가 다시 접속하면 기록은 어떻게 하지?"),
 ("en", "I'm building a tiny platform game", "How should I test its difficulty?"),
 ("en", "I want to organize my audio samples", "Would names based on instruments help?"),
 ("en", "My app's settings panel loads slowly", "How can I find the expensive operation?"),
 ("en", "Can you help me write a friendly note?", "Please make the tone less formal."),
 ("en", "I want to compare two AI checkpoints", "What should I hold constant during the evaluation?"),
 ("en", "My game has too many buttons", "How can I simplify the controls?"),
 ("en", "I'm recording sound effects at home", "How can I avoid background noise?"),
 ("en", "I'm thinking of joining a game jam", "What should I build in the first day?"),
 ("en", "I built a level but it feels empty", "Where should I place the first landmark?"),
 ("en", "The server is responsive but players lag", "What network measurements are useful?")
]


def english_lexical_ratio(text):
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        raise SystemExit("Install wordfreq for the English lexical-quality diagnostic")
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    return sum(zipf_frequency(w.lower(), "en") >= 2.0 for w in words) / len(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default="hgai_model_v12.pth")
    parser.add_argument("--report", default="V12_GENERATION_EVAL.json")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    torch.set_num_threads(2)
    engine = HGAIEngine("train.txt", args.model_file, "main", seed=args.seed, load_model=True)
    if engine.model is None:
        raise SystemExit("Checkpoint failed to load")
    rows = []
    for index, (language, q1, q2) in enumerate(TESTS):
        engine.history = []
        answers = []
        for turn, question in enumerate((q1, q2)):
            torch.manual_seed(args.seed + index * 10 + turn)
            result = engine.reply(question, mode="raw-neural", temperature=.60,
                                  top_k=24, top_p=.88, max_new_tokens=100)
            answers.append(result)
        txt = " ".join(answers)
        fallback = sum(answer in (FALLBACK_RESPONSE, FALLBACK_RESPONSE_EN) for answer in answers)
        mixed = bool(re.search("[가-힣]", txt)) if language == "en" else False
        lexical_ratio = english_lexical_ratio(txt) if language == "en" else None
        row = dict(index=index+1, language=language, prompts=[q1,q2], answers=answers,
                   fallback_count=fallback, mixed_language=mixed, english_lexical_ratio=lexical_ratio)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    english = [r for r in rows if r["language"] == "en"]
    result = dict(model=args.model_file, mode="raw-neural", independent_prompts=True, no_retrieval=True,
                  cases=len(rows), responses=len(rows)*2,
                  fallback_count=sum(r["fallback_count"] for r in rows),
                  english_korean_mixed=sum(int(r["mixed_language"]) for r in english),
                  english_lexical_ratio=sum(r["english_lexical_ratio"] for r in english)/len(english),
                  note="English lexical frequency is a surface spelling indicator, NOT comprehension. Human review required.",
                  tests=rows)
    Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="tests"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
