from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from competition_emotion.models import (
    TextScorer,
    load_text_scorer,
    save_text_scorer,
)
from competition_emotion.types import Song


LABELS = ("狂欢", "孤独")


def song(song_id: str, labels: set[str], name: str, text: str) -> Song:
    return Song(
        song_id=song_id,
        labels=frozenset(labels),
        name=name,
        artists="",
        genre="",
        text=text,
        audio_url="",
    )


class TextScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.songs = [
            song("1", {"狂欢"}, "夏日狂欢", "派对 跳舞 欢呼 happy dance"),
            song("2", {"狂欢"}, "夜晚派对", "朋友 一起 狂欢 音乐 dance"),
            song("3", {"孤独"}, "一个人", "一个人 孤独 没有你 眼泪"),
            song("4", {"孤独"}, "空房间", "寂寞 夜晚 思念 lonely"),
        ]
        self.scorer = TextScorer().fit(self.songs, LABELS)

    def test_bilingual_fixture_scores_loneliness_higher_for_lonely_text(self) -> None:
        scores = self.scorer.score("一个人孤独没有你")

        self.assertGreater(scores["孤独"], scores["狂欢"])

    def test_score_preserves_label_order_and_probability_bounds(self) -> None:
        scores = self.scorer.score("friends dance together")

        self.assertEqual(tuple(scores), LABELS)
        self.assertEqual(set(scores), set(LABELS))
        self.assertTrue(all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in scores.values()))

    def test_blank_input_is_safe(self) -> None:
        scores = self.scorer.score("   ")

        self.assertEqual(tuple(scores), LABELS)
        self.assertTrue(all(np.isfinite(value) for value in scores.values()))

    def test_score_many_has_expected_shape_and_matches_single_score(self) -> None:
        texts = ["一个人孤独没有你", "朋友跳舞狂欢"]

        scores = self.scorer.score_many(texts)

        self.assertEqual(scores.shape, (2, len(LABELS)))
        np.testing.assert_allclose(scores[0], list(self.scorer.score(texts[0]).values()))

    def test_score_many_accepts_an_empty_batch(self) -> None:
        scores = self.scorer.score_many([])

        self.assertEqual(scores.shape, (0, len(LABELS)))

    def test_fit_rejects_labels_with_only_one_class(self) -> None:
        songs = [
            song("1", {"狂欢", "孤独"}, "a", "party alone"),
            song("2", {"狂欢", "孤独"}, "b", "dance lonely"),
        ]

        with self.assertRaisesRegex(ValueError, "狂欢.*孤独"):
            TextScorer().fit(songs, LABELS)

    def test_fit_rejects_invalid_labels_and_empty_song_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            TextScorer().fit(self.songs, ())
        with self.assertRaisesRegex(ValueError, "unique"):
            TextScorer().fit(self.songs, ("狂欢", "狂欢"))
        with self.assertRaisesRegex(ValueError, "blank"):
            TextScorer().fit(self.songs, ("狂欢", " "))
        with self.assertRaisesRegex(ValueError, "songs"):
            TextScorer().fit([], LABELS)

    def test_save_load_roundtrip_preserves_scores(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "text_scorer.joblib"
            expected = self.scorer.score("一个人孤独没有你")

            save_text_scorer(self.scorer, path)
            loaded = load_text_scorer(path)

            self.assertEqual(loaded.labels, LABELS)
            np.testing.assert_allclose(
                list(loaded.score("一个人孤独没有你").values()),
                list(expected.values()),
            )

    def test_load_rejects_malformed_payload(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.joblib"
            invalid_payloads = (
                ({"schema_version": 2}, "schema"),
                (
                    {
                        "schema_version": 1,
                        "labels": LABELS,
                        "vectorizer": TfidfVectorizer(),
                        "classifier": OneVsRestClassifier(LogisticRegression()),
                    },
                    "components",
                ),
                (
                    {
                        "schema_version": 1,
                        "labels": list(LABELS),
                        "vectorizer": self.scorer.vectorizer,
                        "classifier": self.scorer.classifier,
                    },
                    "labels",
                ),
            )
            for payload, message in invalid_payloads:
                with self.subTest(payload=payload):
                    joblib.dump(payload, path)

                    with self.assertRaisesRegex(ValueError, message):
                        load_text_scorer(path)


if __name__ == "__main__":
    unittest.main()
