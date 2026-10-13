#!/usr/bin/env python3
"""Dependency-free regression tests for the v14 bilingual style and local client."""
import json
import unittest
from unittest.mock import patch
from io import BytesIO

from nyang_style_v14 import nyangify
from hgaim_v14_client import query_local


class FakeResponse:
    def __init__(self, answer):
        self.value = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
        self.status = 200

    def __enter__(self):
        return BytesIO(self.value)

    def __exit__(self, *args):
        return False


class NyangTests(unittest.TestCase):
    def test_common_sentences(self):
        cases = [
            ("마인크래프트는 샌드박스 게임이야.", "마인크래프트는 샌드박스 게임이다냥."),
            ("확인해 보자.", "확인해 보자냥."),
            ("맞아요.", "맞다냥."),
            ("장비가 있어.", "장비가 있다냥."),
            ("잘 모르겠어.", "잘 모르겠어냥."),
            ("간단한 질문이야?","간단한 질문이다냥?"),
        ]
        for source, target in cases:
            self.assertEqual(nyangify(source, "ko"), target)
        self.assertEqual(nyangify("좋다냥.", "ko"), "좋다냥.")

    def test_multiple_lines(self):
        t = nyangify("안녕. 나는 HGAIM이야!\n또 이야기하자.", "ko")
        self.assertEqual(t, "안녕냥. 나는 HGAIM이다냥!\n또 이야기하자냥.")

    def test_code_preservation(self):
        original = "코드를 확인해.\n```python\nprint('hi')\n```\n다시 실행해."
        fixed = nyangify(original, "ko")
        self.assertIn("```python\nprint('hi')\n```", fixed)
        self.assertTrue(fixed.endswith("다시 실행한다냥."))
        self.assertIn("코드를 확인한다냥.", fixed)

    def test_english(self):
        self.assertEqual(nyangify("The answer is four.", "en"), "The answer is four, meow.")
        self.assertEqual(nyangify("I can help, meow.", "en"), "I can help, meow.")

    def test_local_only(self):
        with self.assertRaises(ValueError):
            query_local("hi", url="https://api.openai.com")
        with self.assertRaises(ValueError):
            query_local("hi", url="http://192.168.0.1:8086")

    def test_local_generation_style(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse("안녕. 오늘 뭘 할까?")) as m:
            result = query_local("안녕")
            self.assertEqual(result, "안녕냥. 오늘 뭘 할까냥?")
            self.assertEqual(m.call_count, 1)
            request = m.call_args.args[0]
            self.assertTrue(request.full_url.startswith("http://127.0.0.1:8086/"))
            payload = json.loads(request.data)
            self.assertEqual(payload["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
