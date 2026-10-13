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
    "너는 HGAIM이야. 한국어로 짧고 명확하게 대답해. "
    "사실을 확신할 수 없다면 모른다고 말하고 인물이나 게임 설정을 지어내지 마. "
    "대화 문맥을 따라가고, 답변에 지시문이나 말투 설명을 반복하지 마. "
    "출력 스타일은 프로그램이 따로 처리하므로 보통의 자연스러운 문장으로 답해."
)
SYSTEM_EN = (
    "You are HGAIM. Reply naturally and concisely in English. "
    "Never invent an unknown developer, game mechanic, date or fictional lore. "
    "Say when you cannot verify a fact. Do not echo style instructions. "
    "Answer content first; the app applies character speech style separately."
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
