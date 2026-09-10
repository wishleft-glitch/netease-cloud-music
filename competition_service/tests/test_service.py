from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from competition_emotion.models import TextScorer, save_text_scorer
from competition_emotion.service import MAX_REQUEST_BODY_BYTES, create_app
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

    def test_body_limit_rejects_declared_and_misreported_oversized_payloads(self) -> None:
        with TemporaryDirectory() as directory:
            client = TestClient(create_app(_write_bundle(Path(directory) / "bundle")))
            oversized = b" " * (MAX_REQUEST_BODY_BYTES + 1)
            self.assertEqual(
                client.post("/api/v1/emotion/recognize", content=oversized).status_code,
                413,
            )
            # The counting receive wrapper also applies when there is no
            # Content-Length header at all, as in a chunked request.
            sent: list[dict[str, object]] = []

            async def receive() -> dict[str, object]:
                return {"type": "http.request", "body": oversized, "more_body": False}

            async def send(message: dict[str, object]) -> None:
                sent.append(message)

            scope: dict[str, object] = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/emotion/recognize",
                "raw_path": b"/api/v1/emotion/recognize",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            }
            asyncio.run(client.app(scope, receive, send))
            self.assertEqual(sent[0]["status"], 413)

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

    def test_raw_symlink_bundle_root_is_refused_at_initialization(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            target = _write_bundle(base / "actual-bundle")
            symlink = base / "symlink-bundle"
            try:
                symlink.symlink_to(target, target_is_directory=True)
            except OSError as error:  # pragma: no cover - host policy can deny symlinks
                # Windows hosts without Developer Mode cannot create a test
                # symlink.  Simulate the raw lstat result instead: rejection
                # occurs before resolve, which is the security property here.
                parts = list(target.lstat())
                parts[0] = stat.S_IFLNK | 0o777
                fake_symlink_stat = os.stat_result(parts)
                with patch("competition_emotion.service.Path.lstat", return_value=fake_symlink_stat):
                    with self.assertRaisesRegex(ValueError, "bundle root must be a regular directory"):
                        create_app(target)
            else:
                with self.assertRaisesRegex(ValueError, "bundle root must be a regular directory"):
                    create_app(symlink)

    def test_model_replacement_between_precheck_and_open_is_refused_without_loading(self) -> None:
        with TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle")
            version_dir = root / "versions" / "test-version"
            model_path = version_dir / "model.joblib"
            replacement_path = version_dir / "replacement.joblib"
            save_text_scorer(
                TextScorer().fit(
                    [
                        Song("a", frozenset({"狂欢"}), "a", "", "", "热闹", ""),
                        Song("b", frozenset({"孤独"}), "b", "", "", "孤单", ""),
                        Song("c", frozenset({"狂欢"}), "c", "", "", "欢呼", ""),
                        Song("d", frozenset({"孤独"}), "d", "", "", "寂寞", ""),
                    ],
                    LABELS,
                ),
                replacement_path,
            )
            real_open = os.open
            replaced = False

            def replace_before_open(path: str | bytes | os.PathLike[str], flags: int, *args: object) -> int:
                nonlocal replaced
                if Path(path) == model_path and not replaced:
                    replaced = True
                    os.replace(replacement_path, model_path)
                return real_open(path, flags, *args)

            with patch("competition_emotion.service.os.open", side_effect=replace_before_open), patch(
                "competition_emotion.service.load_text_scorer"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "changed while opening"):
                    create_app(root)
            self.assertTrue(replaced)
            loader.assert_not_called()

    def test_invalid_nonwinning_scores_are_rejected_before_ranking(self) -> None:
        with TemporaryDirectory() as directory:
            app = create_app(_write_bundle(Path(directory) / "bundle"))
            client = TestClient(app, raise_server_exceptions=False)
            for invalid in (float("nan"), True, np.bool_(True)):
                with self.subTest(invalid=repr(invalid)), patch.object(
                    app.state.runtime.scorer,
                    "score",
                    return_value={"狂欢": 0.8, "孤独": invalid},
                ):
                    response = client.post("/api/v1/emotion/recognize", json={"title": "验证"})
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("invalid scores", response.text)


if __name__ == "__main__":
    unittest.main()
