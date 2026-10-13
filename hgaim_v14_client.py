#!/usr/bin/env python3
"""Offline-only client for a local llama.cpp server running HGAIM v14.

Uses the pretrained model's chat template via OpenAI-compatible local endpoint.
No retrieval, no remote API, and no changes to existing Discord bot services.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from nyang_style_v14 import nyangify

SYSTEM_KO = (
    "너는 HGAIM이라는 고양이 AI야. 자연스럽고 정확하게 한국어로 답변해. "
    "각 문장을 귀엽지만 과하지 않은 냥체로 마무리해. "
    "Casualties: Unknown이나 게임에 대해 물으면 확인할 수 있는 사실만 알려줘. "
    "확신하지 못하는 캐릭터 설정이나 출시 내용은 지어내지 말고 불확실하다고 말해. "
    "대화 내용을 읽고 맥락을 이어 가. 검색 엔진이나 외부 API는 사용하지 마."
)
SYSTEM_EN = (
    "You are HGAIM, a friendly bilingual cat AI. Answer naturally in clear English. "
    "Use an occasional light 'meow'. For game questions, distinguish facts "
    "from uncertain fan claims and never fabricate game mechanics. "
    "Continue the conversation based on the messages provided. "
    "Do not use any external search or API."
)


def query_local(question, history=None, url="http://127.0.0.1:8086", model="local", max_tokens=140):
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.scheme != "http":
        raise ValueError("For privacy, v14 accepts ONLY an unencrypted loopback local llama.cpp endpoint")
    if not question.strip():
        raise ValueError("Question cannot be empty")
    lang = "ko" if re.search("[가-힣]", question) else "en"
    messages = [{"role": "system", "content": SYSTEM_KO if lang == "ko" else SYSTEM_EN}]
    for item in (history or [])[-4:]:
        role = item.get("role")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(item.get("content", ""))[:3000]})
    messages.append({"role": "user", "content": question[:3000]})
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": .35,
        "top_p": .9,
        "max_tokens": int(max_tokens),
        "repeat_penalty": 1.10,
    }).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=payload, headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            parsed = json.load(resp)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Local HGAIM model server not ready; check llama-server and Pi RAM") from exc
    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError("Local llama.cpp server returned no generated content")
    raw = str(choices[0].get("message", {}).get("content") or "").strip()
    return nyangify(raw, lang)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8086")
    parser.add_argument("--model", default="local")
    parser.add_argument("--max-tokens", type=int, default=140)
    args = parser.parse_args()
    print(query_local(args.question, url=args.url, model=args.model, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
