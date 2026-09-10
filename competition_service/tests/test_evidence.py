from __future__ import annotations

import unittest

from competition_emotion.evidence import build_evidence


class EvidenceTests(unittest.TestCase):
    def test_reports_lyrics_only_when_a_real_lyric_source_contributed(self) -> None:
        self.assertEqual(
            build_evidence(
                text_lyric="  lyric  ",
                lrc_lyric=None,
                lrc_translation=None,
            ),
            "基于歌曲名称、艺人及可用歌词文本进行情绪判定。",
        )

    def test_reports_metadata_only_when_all_lyric_sources_are_blank(self) -> None:
        self.assertEqual(
            build_evidence(
                text_lyric=" ",
                lrc_lyric="[00:01]  ",
                lrc_translation=None,
            ),
            "仅基于歌曲名称和艺人元数据进行情绪判定。",
        )


if __name__ == "__main__":
    unittest.main()
