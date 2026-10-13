#!/usr/bin/env python3
"""Adversarial regression tests for v16.1 intent routing and multi-turn game Q&A."""
import os
import tempfile
import unittest
from unittest.mock import patch

from hgaim_v16_client import answer, resolve_followup
from hgaim_v16_grounded import detect_games, find_fact, is_game_fact_request, load_facts


def sourced(question, fact_id, history=None):
    response = answer(question, history=history, url="http://127.0.0.1:1", metadata=True)
    assert response["route"] == "source", response
    assert response["fact_id"] == fact_id, response
    assert response["source"], response
    return response


class RoutingReliabilityTests(unittest.TestCase):
    def test_game_name_followed_by_korean_particle(self):
        self.assertIn("Casualties: Unknown", detect_games("C:U의 게임 특징 알려줘"))
        self.assertIn("Gunsaw", detect_games("Gunsaw라는 게임"))
        self.assertIn("Roblox", detect_games("Roblox에서"))
        self.assertIn("Changed", detect_games("Changed는 무슨 게임이야?"))

    def test_changed_verb_is_not_game_title(self):
        for sentence in ("I changed my settings.", "We changed the server config.", "changed my mind"):
            with self.subTest(sentence=sentence):
                self.assertNotIn("Changed", detect_games(sentence))
        self.assertIn("Changed", detect_games("What is Changed?"))
        self.assertIn("Changed", detect_games("Who is Puro?"))

    def test_cu_korean_possessive_and_related_games(self):
        self.assertEqual(sourced("C:U의 OSPREY가 옮기도록 설계된 건 뭐야?", "lore-osprey")["game"], "Casualties: Unknown")
        related=sourced("Gunsaw와 C:U의 관계가 뭐야?", "relation")
        self.assertEqual(related["game"], "Gunsaw")
        self.assertIn("팬 위키", related["answer"])
        self.assertIsNone(find_fact("Gunsaw와 Changed 개발자가 같은 사람이야?"))

    def test_programming_language_context(self):
        sourced("Roblox에서 사용하는 프로그래밍 언어는 무엇이야?", "scripting")
        sourced("What language is Roblox scripting written in?", "scripting")

    def test_source_paths_not_tied_to_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=os.getcwd()
            try:
                os.chdir(folder)
                result=sourced("Gunsaw 개발자 누구야?", "developer")
                self.assertIn("Orsoniks", result["answer"])
                self.assertGreater(len(load_facts()),100)
            finally:
                os.chdir(previous)

    def test_editor_authorship_not_developer_attribution(self):
        self.assertIsNone(find_fact("Gunsaw 레벨 에디터 만든 사람 누구야?"))
        self.assertIsNone(find_fact("Who created the Gunsaw level editor?"))
        response=answer("Gunsaw 레벨 에디터 만든 사람 누구야?", metadata=True, url="http://127.0.0.1:1")
        self.assertEqual(response["route"], "abstain")
        self.assertNotIn("Orsoniks", response["answer"])

    def test_unknown_dates_refused(self):
        for q in ["Gunsaw의 출시일은 언제야?", "Roblox Studio 출시일 알려줘", "Gunsaw 가격이 얼마야?"]:
            with self.subTest(question=q):
                response=answer(q, metadata=True, url="http://127.0.0.1:1")
                self.assertEqual(response["route"], "abstain")

    def test_open_ended_requests_go_to_neural(self):
        questions=["Minecraft에서 뭐 만들면 좋을까?", "Minecraft 레드스톤 아이디어 추천해줘",
                   "Roblox Studio 사용법 알려줘", "How should I build a house in Minecraft?"]
        with patch("hgaim_v16_client.ask_neural",return_value="좋아, 작은 연습부터 해 보자냥.") as mock:
            for q in questions:
                with self.subTest(question=q):
                    response=answer(q,metadata=True)
                    self.assertEqual(response["route"],"neural",response)
            self.assertEqual(mock.call_count,len(questions))

    def test_followup_uses_only_unambiguous_prior_user(self):
        hist=[{"role":"user","content":"Gunsaw는 어떤 게임이야?"},
              {"role":"assistant","content":"2D 게임이다냥."}]
        result=sourced("그 게임 개발자는 누구야?", "developer", history=hist)
        self.assertTrue(result["context_resolved"])
        self.assertIn("Orsoniks",result["answer"])
        self.assertEqual(resolve_followup("안녕",hist),"안녕")

    def test_followup_game_with_korean_particles(self):
        hist=[{"role":"user","content":"Roblox는 무슨 게임이야?"}]
        result=sourced("그 게임은 무슨 언어로 스크립트를 작성해?", "scripting", history=hist)
        self.assertTrue(result["context_resolved"])

    def test_followup_in_world_specific_item(self):
        hist=[{"role":"user","content":"C:U의 OSPREY라는 장비를 보고 있어."}]
        result=sourced("그 장비는 뭘 옮기도록 설계됐어?", "lore-osprey", history=hist)
        self.assertIn("건물 전체",result["answer"])

    def test_followup_unknown_attribute_does_not_guess(self):
        hist=[{"role":"user","content":"Gunsaw는 어떤 게임이야?"}]
        result=answer("그 개발자는 몇 살이야?",history=hist,metadata=True,url="http://127.0.0.1:1")
        self.assertEqual(result["route"],"abstain")

    def test_no_context_leakage_multiple_games(self):
        hist=[{"role":"user","content":"Gunsaw와 Changed 비교해줘"}]
        with patch("hgaim_v16_client.ask_neural",return_value="추가 문맥이 필요하다냥.") as mock:
            result=answer("그 게임의 개발자는 누구야?",history=hist,metadata=True)
            self.assertEqual(result["route"],"neural")
            mock.assert_called_once()

    def test_existing_local_source_answer_not_fake_neural_accuracy(self):
        with patch("hgaim_v16_client.ask_neural",side_effect=AssertionError("must not call neural")):
            v=sourced("C:U의 OSPREY는 뭘 옮겨?", "lore-osprey")
            self.assertEqual(v["route"],"source")


if __name__=="__main__":
    unittest.main(verbosity=2)
