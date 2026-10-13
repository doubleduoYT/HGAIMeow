#!/usr/bin/env python3
"""Verify that proven game routing is really active in the original v10 engine.

Uses the repository's existing v10 dataset with neural loading disabled.
Does not require torch, a GPU, a llama.cpp server or a pretrained checkpoint.
"""
from __future__ import annotations
import unittest
from unittest.mock import patch

from hgai_core import HGAIEngine, FALLBACK_RESPONSE


class StableRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=HGAIEngine(load_model=False)

    def setUp(self):
        self.engine.history.clear()

    def test_source_game_developer(self):
        response=self.engine.reply("Gunsaw 개발자는 누구야?",mode="hybrid")
        self.assertIn("Orsoniks",response)
        self.assertIn("냥",response)
        self.assertEqual(self.engine.history[-1][1],response)

    def test_cu_osprey_and_preserved_context(self):
        first=self.engine.reply("C:U의 OSPREY는 뭘 옮겨?",mode="hybrid")
        self.assertIn("건물 전체",first)
        follow=self.engine.reply("그 장비는 뭘 옮겨?",mode="hybrid")
        self.assertIn("건물 전체",follow)

    def test_original_search_supports_new_source(self):
        answer=self.engine.reply("Roblox에서 사용하는 프로그래밍 언어는 뭐야?",mode="search")
        self.assertIn("Luau",answer)

    def test_english_keeps_english_and_meow(self):
        answer=self.engine.reply("Who developed Gunsaw?",mode="hybrid")
        self.assertIn("Orsoniks",answer)
        self.assertIn("meow",answer.lower())
        self.assertNotRegex(answer,r"[가-힣]")

    def test_explicitly_unsupported_release_not_invented(self):
        response=self.engine.reply("Gunsaw 출시일은 언제야?",mode="hybrid")
        self.assertIn("근거",response)
        self.assertNotIn("2020",response)

    def test_raw_neural_is_still_raw_for_unbiased_evaluation(self):
        with patch.object(HGAIEngine,"_reply_core",return_value="일반적인 응답이야."):
            answer=self.engine.reply("테스트",mode="raw-neural")
            self.assertEqual(answer,"일반적인 응답이야.")
            styled=self.engine.reply("테스트",mode="hybrid")
            self.assertEqual(styled,"일반적인 응답이다냥.")

    def test_general_advice_still_uses_existing_engine(self):
        with patch("hgai_core.grounded_answer",side_effect=AssertionError("Never call on raw-neural")):
            self.engine.reply("Minecraft에서 뭘 만들면 좋을까?",mode="raw-neural")

if __name__=="__main__":
    unittest.main(verbosity=2)
