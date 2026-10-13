#!/usr/bin/env python3
"""Offline regression suite for v16's C:U/game-grounded fallback.

The source path must resolve deterministically even when llama-server is absent.
It is intentionally scored separately from neural-model generation tests.
"""
import unittest
from unittest.mock import patch

from hgaim_v16_client import answer
from hgaim_v16_grounded import find_fact, grounded_answer, load_facts
from nyang_style_v14 import nyangify


class GroundedGameTests(unittest.TestCase):
    def check(self,question,needle,expected_id):
        result=answer(question,url="http://127.0.0.1:1",metadata=True)
        self.assertEqual(result["route"],"source",result)
        self.assertEqual(result["fact_id"],expected_id)
        self.assertIn(needle.lower(),result["answer"].lower())
        self.assertTrue(result["source"])
        return result

    def test_game_reference_sources(self):
        refs=load_facts()
        self.assertGreaterEqual(len(refs),100)
        self.assertIn(("Casualties: Unknown","lore-osprey"),refs)

    def test_cu_osprey_holdout_in_local_knowledge(self):
        self.check("C:U 문서에 등장하는 OSPREY 장치는 어떤 물체를 옮기려는 목적으로 설계됐니?","건물 전체","lore-osprey")
        self.check("According to C:U notes, what was OSPREY designed to move?","entire buildings","lore-osprey")

    def test_cu_saw_and_yellow_blood(self):
        self.check("SAW-03의 털 색과 외형은 뭐야?","우윳빛 털","lore-saw03")
        self.check("C:U 노란 피의 방사선 회복 능력은?","방사선","lore-yellowblood")

    def test_games_korean_and_english(self):
        self.check("Gunsaw라는 게임의 개발자는 누구야?","Orsoniks","developer")
        self.check("Who developed Gunsaw?","Orsoniks","developer")
        self.check("Roblox에서 프로그래밍 언어가 뭐야?","Luau","scripting")
        self.check("What programming language is used in Roblox scripting?","Luau","scripting")
        self.check("얼불춤의 두 행성과 박자는 어떻게 연결돼?","박자","mechanic")
        self.check("How does A Dance of Fire and Ice use the one-button orbital rhythm?","beat","mechanic")
        self.check("Changed 속의 Puro는 누구야?","콜린","puro")
        self.check("Who is Puro in Changed?","Colin","puro")
        self.check("Minecraft 레드스톤으로 자동 문을 만들 수 있어?","자동문","redstone")

    def test_unverified_game_no_fabrications(self):
        a=self.check("DevilConnection 공식 세계관 알려줘","검증된 자료는","identity")
        self.assertEqual(a["game"],"DevilConnection")
        a=answer("Gunsaw 출시 연도가 정확히 언제야?",url="http://127.0.0.1:1",metadata=True)
        self.assertEqual(a["route"],"abstain",a)
        self.assertIn("근거",a["answer"])

    def test_no_wrong_fact_routing(self):
        self.assertIsNone(find_fact("OSPREY의 제작자 이름은?"))
        self.assertIsNone(find_fact("Gunsaw의 레벨 에디터는 누구야?"))
        self.assertIsNone(find_fact("Roblox Studio의 출시일은 언제야?"))
        self.assertIsNone(find_fact("Puro의 목소리를 연기한 사람은 누구야?"))
        self.assertIsNone(find_fact("Minecraft에서 달리기 속도를 높이는 방법은 뭐야?"))

    def test_general_chat_uses_local_neural_only(self):
        with patch("hgaim_v16_client.ask_neural",return_value="CPU, 메모리, 디스크부터 측정해 보자냥.") as mock:
            a=answer("게임이 버벅거리는데 어디부터 확인하지?",metadata=True)
            self.assertEqual(a["route"],"neural")
            self.assertIn("CPU",a["answer"])
            self.assertEqual(mock.call_count,1)

    def test_known_fact_never_calls_neural(self):
        with patch("hgaim_v16_client.ask_neural",side_effect=AssertionError("Must not invoke neural")):
            self.check("Gunsaw 개발자 누구야?","Orsoniks","developer")

    def test_nyang_natural_ending(self):
        self.assertEqual(nyangify("순간이동 설비야.","ko"),"순간이동 설비다냥.")
        self.assertEqual(nyangify("테스트를 다시 해.","ko"),"테스트를 다시 한다냥.")
        self.assertIn("냥",answer("Gunsaw 개발자 누구야?"))

if __name__=="__main__":
    unittest.main(verbosity=2)
