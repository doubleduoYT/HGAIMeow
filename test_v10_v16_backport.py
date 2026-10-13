#!/usr/bin/env python3
"""Regression tests for safe v16.1 factual hotfix on the original v10 engine."""
import os
import tempfile
import unittest
from unittest.mock import patch

from hgai_core import HGAIEngine, FALLBACK_RESPONSE
from hgaim_v16_grounded import grounded_answer, load_facts
from hgaim_v16_context import resolve_followup


class StableV10BackportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicitly avoid a Torch checkpoint so this tests the Pi-safe
        # source route even when no model weights are installed.
        cls.engine = HGAIEngine(load_model=False, seed=1234)

    def setUp(self):
        self.engine.history=[]

    def test_original_v10_cli_and_engine_signature(self):
        self.assertFalse(self.engine.neural_ready)
        self.assertTrue(callable(self.engine.reply))
        self.assertEqual(self.engine.model_info()["version"],"hgai-v10")
        self.assertGreater(self.engine.model_info()["runtime_pairs"],0)

    def test_source_cu_entity_with_korean_particle(self):
        a=self.engine.reply("C:U의 OSPREY는 뭘 옮겨?",mode="hybrid")
        self.assertIn("건물 전체",a)
        self.assertTrue(a.endswith("냥."),a)

    def test_source_gunsaw_developer(self):
        a=self.engine.reply("Gunsaw 개발자 누구야?",mode="hybrid")
        self.assertIn("Orsoniks",a)

    def test_source_roblox_luau(self):
        a=self.engine.reply("Roblox에서 사용하는 프로그래밍 언어 뭐야?",mode="search")
        self.assertIn("Luau",a)

    def test_source_advises_changed(self):
        a=self.engine.reply("Changed 속 Puro는 누구야?")
        self.assertIn("콜린",a)

    def test_source_english(self):
        a=self.engine.reply("Who developed Gunsaw?",mode="hybrid")
        self.assertIn("Orsoniks",a)
        self.assertNotIn("한글",a)
        self.assertIn("meow",a)

    def test_followup_in_existing_engine(self):
        a=self.engine.reply("Gunsaw 개발자 누구야?")
        b=self.engine.reply("그 게임 개발자는 누구야?")
        self.assertIn("Orsoniks",a)
        self.assertIn("Orsoniks",b)
        self.assertEqual(self.engine.history[-1][0],"그 게임 개발자는 누구야?")

    def test_cu_context_followup(self):
        self.engine.reply("C:U의 OSPREY는 뭘 옮겨?")
        second=self.engine.reply("그 장비는 뭘 옮기도록 설계됐어?")
        self.assertIn("건물",second)

    def test_unverified_release_date_refused(self):
        a=self.engine.reply("Gunsaw 출시일은 언제야?")
        self.assertIn("확인할 근거",a)
        self.assertNotIn("2025",a)

    def test_no_opinion_roulette(self):
        a=self.engine.reply("Casualties: Unknown 게임이 뭐야?")
        self.assertIn("하드코어",a)

    def test_raw_neural_quality_evaluation_is_not_source_lookup(self):
        # raw-neural must remain retrieval-free for future model benchmarks.
        a=self.engine.reply("Who developed Gunsaw?",mode="raw-neural")
        self.assertEqual(a,FALLBACK_RESPONSE)

    def test_source_path_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d)
                a=self.engine.reply("Gunsaw 개발자 누구야?")
                self.assertIn("Orsoniks",a)
                self.assertIn(("Gunsaw","developer"),load_facts())
            finally:
                os.chdir(old)

    def test_general_chat_uses_original_v10_engine(self):
        with patch.object(self.engine, "_legacy_reply",return_value="간단한 예제부터 만들어 보자."):
            a=self.engine.reply("새 게임을 만들고 싶어")
        self.assertIn("예제",a)
        self.assertIn("냥",a)

    def test_no_external_model_or_network_needed(self):
        with patch("urllib.request.urlopen",side_effect=AssertionError("no network allowed")):
            a=self.engine.reply("C:U의 OSPREY는 뭐야?")
        self.assertIn("건물",a)

    def test_followup_context_is_not_invented(self):
        self.assertEqual(resolve_followup("개발자는 누구야?",[]),"개발자는 누구야?")
        self.assertEqual(resolve_followup("안녕",[{"role":"user","content":"Gunsaw 개발자 누구야?"}]),"안녕")

    def test_grounded_provenance_survives(self):
        r=grounded_answer("Gunsaw 개발자 누구야?")
        self.assertEqual(r["route"],"source")
        self.assertTrue(r["source"].startswith("https://"))
        self.assertEqual(r["fact_id"],"developer")


if __name__=="__main__":
    unittest.main(verbosity=2)
