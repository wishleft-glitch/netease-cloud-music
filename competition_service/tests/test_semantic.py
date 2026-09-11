from __future__ import annotations

from unittest.mock import MagicMock, patch
import unittest

from competition_emotion.semantic import (
    SemanticReviewConfig,
    review_candidates,
)
from competition_emotion.rubric import DEFAULT_RUBRIC_PATH, load_rubric
from competition_emotion.types import Song


class SemanticReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.song = Song("1", frozenset(), "歌名", "艺人", "流行", "歌词", "")
        self.config = SemanticReviewConfig("http://reviewer.internal/review")

    def test_config_rejects_credentials_and_bad_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            SemanticReviewConfig("http://user:pass@reviewer.internal/review")
        with self.assertRaisesRegex(ValueError, "min_score_gap"):
            SemanticReviewConfig("http://reviewer.internal/review", min_score_gap=2)
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            SemanticReviewConfig("http://reviewer.internal/review", candidate_count=1)
        with self.assertRaisesRegex(ValueError, "internal host"):
            SemanticReviewConfig("https://reviewer.example.com/review")

    def test_config_defaults_to_cost_bounded_review_gap(self) -> None:
        self.assertEqual(self.config.min_score_gap, 0.10)
        self.assertEqual(self.config.candidate_count, 7)

    def test_review_sends_song_context_and_accepts_only_a_candidate(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "label": "孤独",
            "confidence": 0.91,
            "evidence": "歌词反复表达独处和隔离。",
            "quotes": ["一个人走在夜里"],
        }
        response.raise_for_status.return_value = None
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        with patch("competition_emotion.semantic.httpx.Client", return_value=client):
            result = review_candidates(self.song, "一个人走在夜里", ["孤独", "思念"], self.config)

        self.assertEqual((result.label, result.confidence), ("孤独", 0.91))
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["song_id"], "1")
        self.assertEqual(payload["candidates"], ["孤独", "思念"])
        self.assertEqual([item["label"] for item in payload["rubric"]], ["孤独", "思念"])
        self.assertTrue(payload["rubric"][0]["rule_ids"])
        self.assertNotIn("audio_url", payload)

    def test_review_rejects_fabricated_lyric_quote(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "label": "孤独",
            "confidence": 0.8,
            "evidence": "看起来孤独。",
            "quotes": ["这句歌词不在输入里"],
        }
        response.raise_for_status.return_value = None
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        with patch("competition_emotion.semantic.httpx.Client", return_value=client):
            with self.assertRaisesRegex(ValueError, "invalid"):
                review_candidates(self.song, "一个人走在夜里", ["孤独", "思念"], self.config, load_rubric(DEFAULT_RUBRIC_PATH))

    def test_review_rejects_a_label_outside_candidates(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "label": "悲伤",
            "confidence": 0.8,
            "evidence": "不在候选集中",
        }
        response.raise_for_status.return_value = None
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        with patch("competition_emotion.semantic.httpx.Client", return_value=client):
            with self.assertRaisesRegex(ValueError, "invalid"):
                review_candidates(self.song, "歌词", ["孤独", "思念"], self.config)


if __name__ == "__main__":
    unittest.main()
