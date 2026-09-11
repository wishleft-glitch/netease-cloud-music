from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from competition_emotion.models import (
    TextScorer,
    compose_model_text,
    load_text_scorer,
    save_text_scorer,
)
from competition_emotion.types import Song


LABELS = ("狂欢", "孤独")


def label_order_digest(labels: tuple[str, ...]) -> str:
    canonical = json.dumps(list(labels), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def song_with_artist(song_id: str, labels: set[str], name: str, artist: str, text: str) -> Song:
    return Song(
        song_id=song_id,
        labels=frozenset(labels),
        name=name,
        artists=artist,
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

    def test_fit_uses_metadata_aware_svc_configuration(self) -> None:
        self.assertEqual(self.scorer.score_mode, "softmax")
        self.assertEqual(self.scorer.vectorizer.ngram_range, (1, 5))
        self.assertEqual(self.scorer.vectorizer.max_features, 150000)
        self.assertIsInstance(self.scorer.classifier.estimators_[0], LinearSVC)
        self.assertEqual(self.scorer.classifier.estimators_[0].C, 0.3)
        self.assertIsNone(self.scorer.classifier.estimators_[0].class_weight)

    def test_model_text_repeats_metadata_and_strips_loader_prefix(self) -> None:
        loaded = song("1", {"狂欢"}, "夏日狂欢", "派对跳舞")
        loaded = Song(
            loaded.song_id,
            loaded.labels,
            loaded.name,
            "歌手",
            "流行",
            "夏日狂欢 歌手 真正歌词",
            loaded.audio_url,
        )
        model_text = compose_model_text(loaded)
        self.assertEqual(model_text.count("夏日狂欢"), 5)
        self.assertEqual(model_text.count("歌手"), 5)
        self.assertEqual(model_text.count("流行"), 5)
        self.assertTrue(model_text.endswith("真正歌词"))

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

    def test_fit_builds_high_agreement_artist_override_and_applies_it(self) -> None:
        songs = [
            song_with_artist("1", {"狂欢"}, "a", "同一艺人", "party"),
            song_with_artist("2", {"狂欢"}, "b", "同一艺人", "dance"),
            song_with_artist("3", {"孤独"}, "c", "另一艺人", "lonely"),
            song_with_artist("4", {"孤独"}, "d", "另一艺人", "alone"),
        ]
        scorer = TextScorer().fit(songs, LABELS)

        self.assertEqual(scorer.artist_overrides, {"同一艺人": "狂欢", "另一艺人": "孤独"})
        adjusted = scorer.apply_song_overrides(
            songs[0], {"狂欢": 0.2, "孤独": 0.8}
        )
        self.assertGreater(adjusted["狂欢"], adjusted["孤独"])

    def test_artist_override_requires_two_agreeing_training_songs(self) -> None:
        songs = [
            song_with_artist("1", {"狂欢"}, "a", "单首艺人", "party"),
            song_with_artist("2", {"孤独"}, "b", "混合艺人", "lonely"),
            song_with_artist("3", {"狂欢"}, "c", "混合艺人", "party"),
        ]
        scorer = TextScorer().fit(songs, LABELS)

        self.assertNotIn("单首艺人", scorer.artist_overrides)
        self.assertNotIn("混合艺人", scorer.artist_overrides)

    def test_single_label_configuration_rejects_missing_real_negatives(self) -> None:
        songs = [
            song("1", {"孤独"}, "一个人", "一个人 孤独"),
            song("2", {"孤独"}, "空房间", "寂寞 夜晚"),
        ]

        with self.assertRaisesRegex(ValueError, "one-label.*real negatives"):
            TextScorer().fit(songs, ("孤独",))

    def test_fit_rejects_labels_with_only_one_class(self) -> None:
        songs = [
            song("1", {"狂欢", "孤独"}, "a", "party alone"),
            song("2", {"狂欢", "孤独"}, "b", "dance lonely"),
        ]

        with self.assertRaisesRegex(ValueError, "狂欢.*孤独"):
            TextScorer().fit(songs, LABELS)

    def test_fit_rejects_missing_label_for_multilabel_configuration(self) -> None:
        songs = [
            song("1", {"狂欢"}, "a", "party"),
            song("2", set(), "b", "quiet"),
            song("3", set(), "c", "still"),
        ]

        with self.assertRaisesRegex(ValueError, "孤独"):
            TextScorer().fit(songs, LABELS)

    def test_fit_rejects_song_labels_outside_the_configuration(self) -> None:
        songs = [
            song("1", {"狂欢", "未知"}, "a", "party"),
            song("2", {"孤独"}, "b", "lonely"),
        ]

        with self.assertRaisesRegex(ValueError, "未知"):
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
            saved_payload = joblib.load(path)
            loaded = load_text_scorer(path, trusted=True)

            self.assertEqual(saved_payload["schema_version"], 2)
            self.assertIn("artist_overrides", saved_payload)
            self.assertEqual(loaded.labels, LABELS)
            np.testing.assert_allclose(
                list(loaded.score("一个人孤独没有你").values()),
                list(expected.values()),
            )

    def test_legacy_payload_keeps_legacy_request_text(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-text-scorer.joblib"
            save_text_scorer(self.scorer, path)
            payload = joblib.load(path)
            payload.pop("input_mode")
            joblib.dump(payload, path)

            loaded = load_text_scorer(path, trusted=True)
            request_song = Song(
                "legacy",
                frozenset(),
                "标题",
                "艺人",
                "流行",
                "真正歌词",
                "",
            )
            self.assertEqual(loaded.input_mode, "legacy")
            self.assertEqual(loaded.compose_text(request_song), "真正歌词")

    def test_load_requires_explicit_trust_before_deserializing(self) -> None:
        with patch("competition_emotion.models.joblib.load") as mocked_load:
            with self.assertRaisesRegex(ValueError, "joblib artifacts must be trusted"):
                load_text_scorer("untrusted.joblib")

        mocked_load.assert_not_called()

    def test_load_rejects_reordered_labels(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reordered.joblib"
            save_text_scorer(self.scorer, path)
            payload = joblib.load(path)
            payload["labels"] = tuple(reversed(LABELS))
            joblib.dump(payload, path)

            with self.assertRaisesRegex(ValueError, "trained_labels"):
                load_text_scorer(path, trusted=True)

    def test_load_rejects_malformed_payload(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.joblib"
            base_payload = {
                "schema_version": 2,
                "labels": LABELS,
                "trained_labels": LABELS,
                "label_order_sha256": label_order_digest(LABELS),
                "vectorizer": self.scorer.vectorizer,
                "classifier": self.scorer.classifier,
            }
            invalid_payloads = (
                ({"schema_version": 3}, "schema"),
                (
                    {
                        **base_payload,
                        "vectorizer": TfidfVectorizer(),
                        "classifier": OneVsRestClassifier(LogisticRegression()),
                    },
                    "components",
                ),
                (
                    {
                        **base_payload,
                        "labels": list(LABELS),
                    },
                    "labels",
                ),
                (
                    {
                        **base_payload,
                        "label_order_sha256": "0" * 64,
                    },
                    "label_order_sha256",
                ),
                (
                    {
                        **base_payload,
                        "labels": ("狂欢", "孤独", "额外"),
                        "trained_labels": ("狂欢", "孤独", "额外"),
                        "label_order_sha256": label_order_digest(
                            ("狂欢", "孤独", "额外")
                        ),
                    },
                    "components",
                ),
            )
            for payload, message in invalid_payloads:
                with self.subTest(payload=payload):
                    joblib.dump(payload, path)

                    with self.assertRaisesRegex(ValueError, message):
                        load_text_scorer(path, trusted=True)

    def test_load_rejects_v1_payload_with_migration_guidance(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v1.joblib"
            joblib.dump({"schema_version": 1}, path)

            with self.assertRaisesRegex(ValueError, "migration|retrain"):
                load_text_scorer(path, trusted=True)

    def test_project_metadata_contains_all_runtime_requirements(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((package_root / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = {
            line.strip()
            for line in (package_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertEqual(set(project["project"]["dependencies"]), requirements)


if __name__ == "__main__":
    unittest.main()
