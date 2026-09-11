from __future__ import annotations

import unittest

from competition_emotion.evidence import build_evidence


class EvidenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
