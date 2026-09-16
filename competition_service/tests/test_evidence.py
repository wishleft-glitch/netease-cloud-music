from __future__ import annotations

import unittest
from pathlib import Path

from competition_emotion.evidence import build_evidence
from competition_emotion.rubric import load_rubric


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubric = load_rubric(Path(__file__).parents[1] / "rubric" / "emotion_rubric.json")

    def test_reports_only_lyrics_when_lyrics_were_scored(self) -> None:
        self.assertEqual(
            build_evidence(lyric_used=True, title_used=False),
            "基于可用歌词文本进行情绪判定。",
        )

    def test_reports_only_song_name_when_it_was_the_fallback_input(self) -> None:
        self.assertEqual(
            build_evidence(lyric_used=False, title_used=True),
            "仅基于歌曲名称进行情绪判定。",
        )

    def test_reports_metadata_when_the_metadata_aware_model_is_used(self) -> None:
        self.assertEqual(
            build_evidence(lyric_used=True, title_used=False, metadata_used=True),
            "基于歌曲名称、艺人、专辑元数据与可用歌词进行情绪判定。",
        )

    def test_adds_only_a_verified_rubric_cue_quote(self) -> None:
        evidence = build_evidence(
            lyric_used=True,
            title_used=False,
            metadata_used=True,
            lyric_text="一个人寂寞独处，没人理解我的沉默。",
            label="孤独",
            rubric=self.rubric,
        )
        self.assertIn("歌词证据：\u201c寂寞独处\u201d", evidence)

    def test_does_not_add_a_quote_when_no_rubric_cue_occurs(self) -> None:
        evidence = build_evidence(
            lyric_used=True,
            title_used=False,
            metadata_used=True,
            lyric_text="今天天气很好。",
            label="孤独",
            rubric=self.rubric,
        )
        self.assertIn("输入歌词片段：\u201c今天天气很好。\u201d", evidence)
        self.assertIn("标签定义：", evidence)

    def test_excerpt_is_taken_verbatim_from_the_supplied_lyrics(self) -> None:
        evidence = build_evidence(
            lyric_used=True,
            title_used=False,
            lyric_text="[00:01.00]窗外有一盏灯，风吹过街道。\n第二行歌词",
            label="思念",
            rubric=self.rubric,
        )
        self.assertIn("输入歌词片段：\u201c窗外有一盏灯，风吹过街道。\u201d", evidence)


if __name__ == "__main__":
    unittest.main()
