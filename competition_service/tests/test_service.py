from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from competition_emotion.models import TextScorer, save_text_scorer
from competition_emotion.service import create_app
from competition_emotion.types import Song


LABELS = ("狂欢", "孤独")


def _write_bundle(root: Path) -> Path:
    version_dir = root / "versions" / "test-version"
    version_dir.mkdir(parents=True)
    songs = [
        Song("1", frozenset({"狂欢"}), "派对", "甲", "流行", "跳舞 欢呼 热闹", ""),
        Song("2", frozenset({"狂欢"}), "庆祝", "乙", "流行", "派对 狂欢 开心", ""),
        Song("3", frozenset({"孤独"}), "夜晚", "丙", "流行", "一个人 寂寞 孤单", ""),
        Song("4", frozenset({"孤独"}), "离开", "丁", "流行", "无人 陪伴 安静", ""),
    ]
    scorer = TextScorer().fit(songs, LABELS)
    save_text_scorer(scorer, version_dir / "model.joblib")
    (version_dir / "report.json").write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "model_type": "lyrics_tfidf_logreg",
                "model_version": "test-v1",
                "labels": list(LABELS),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "current.json").write_text(
        json.dumps(
            {"pointer_schema_version": 1, "active_bundle": "versions/test-version"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


class ServiceTests(unittest.TestCase):
    def test_health_and_recognition_contract_with_configured_tie_order(self) -> None:
        with TemporaryDirectory() as directory:
            app = create_app(_write_bundle(Path(directory) / "bundle"))
            client = TestClient(app)

            health = client.get("/healthz")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(
                health.json(),
                {"ready": True, "model_type": "lyrics_tfidf_logreg", "model_version": "test-v1", "label_count": 2},
            )

            response = client.post(
                "/api/v1/emotion/recognize",
                json={"song_id": "abc", "title": "新歌", "artists": "歌手", "genre": "流行", "lyrics": "任意"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(set(response.json()), {"song_id", "emotion_label", "confidence", "model_type", "model_version"})
            self.assertEqual(response.json()["song_id"], "abc")
            self.assertIn(response.json()["emotion_label"], LABELS)
            self.assertTrue(0.0 <= response.json()["confidence"] <= 1.0)

            with patch.object(app.state.runtime.scorer, "score", return_value={"孤独": 0.5, "狂欢": 0.5}):
                tied = client.post("/api/v1/emotion/recognize", json={"title": "同分"})
            self.assertEqual(tied.status_code, 200)
            self.assertEqual(tied.json()["emotion_label"], "狂欢")

    def test_invalid_request_uses_standard_validation_errors(self) -> None:
        with TemporaryDirectory() as directory:
            client = TestClient(create_app(_write_bundle(Path(directory) / "bundle")))
            self.assertEqual(client.post("/api/v1/emotion/recognize", json={"title": "  "}).status_code, 422)
            self.assertEqual(client.post("/api/v1/emotion/recognize", json={"title": 7}).status_code, 422)
            self.assertEqual(client.post("/api/v1/emotion/recognize", json={"title": "ok", "extra": "no"}).status_code, 422)
            self.assertEqual(client.post("/api/v1/emotion/recognize", json={"title": "x" * 301}).status_code, 422)

    def test_corrupted_or_malicious_pointer_is_refused_at_initialization(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            root.mkdir()
            (root / "current.json").write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid bundle pointer"):
                create_app(root)

            (root / "current.json").write_text(
                json.dumps({"pointer_schema_version": 1, "active_bundle": "versions/../escape"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid active_bundle"):
                create_app(root)

    def test_mismatched_label_configuration_is_refused_at_initialization(self) -> None:
        with TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle")
            report_path = root / "versions" / "test-version" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["labels"] = ["孤独", "狂欢"]
            report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "labels do not match"):
                create_app(root)


if __name__ == "__main__":
    unittest.main()
