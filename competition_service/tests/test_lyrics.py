from __future__ import annotations

import unittest

from competition_emotion.lyrics import compose_lyrics


class ComposeLyricsTests(unittest.TestCase):
    def test_preserves_cleaned_sources_in_fixed_order(self) -> None:
        result = compose_lyrics(
            "  plain   lyric ",
            "[00:01] original\n[00:02.5] line",
            " [00:01] translated   line ",
        )

        self.assertEqual(result, "plain lyric\noriginal line\ntranslated line")

    def test_skips_exact_duplicate_cleaned_portions(self) -> None:
        result = compose_lyrics(
            "same lyric",
            "[00:01] same   lyric",
            "translation",
        )

        self.assertEqual(result, "same lyric\ntranslation")


if __name__ == "__main__":
    unittest.main()
