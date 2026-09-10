from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from shutil import copyfile
import stat
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import numpy as np

from competition_emotion.constants import LABELS
from competition_emotion.models import TextScorer, save_text_scorer
from competition_emotion.request_audio import RequestAudioResult
from competition_emotion.service import MAX_REQUEST_BODY_BYTES, ServiceAudioConfig, create_app
from competition_emotion.types import Song


def _write_bundle(root: Path) -> Path:
    version_dir = root / "versions" / "test-version"
    version_dir.mkdir(parents=True)
    songs: list[Song] = []
    for index, label in enumerate(LABELS):
        songs.extend(
            (
                Song(
                    str(index * 2 + 1),
                    frozenset({label}),
                    f"{label}歌曲",
                    "测试艺人",
                    "测试专辑",
                    f"{label} 情绪文本 {index}",
                    "",
                ),
                Song(
                    str(index * 2 + 2),
                    frozenset(),
                    f"中性歌曲{index}",
                    "测试艺人",
                    "测试专辑",
                    f"中性文本 {index}",
                    "",
                ),
            )
        )
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
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.app = create_app(_write_bundle(Path(self.directory.name) / "bundle"), audio_config=None)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _request(self, **overrides: object) -> dict[str, object]:
        request: dict[str, object] = {
            "audio_url": "https://example.test/audio.mp3",
            "song_name": "新歌",
            "song_id": "abc",
            "artists": "歌手",
        }
        request.update(overrides)
        return request

    def _scores(self, **overrides: float) -> dict[str, float]:
        scores = {label: 0.01 for label in LABELS}
        scores.update(overrides)
        return scores

    def test_health_and_official_recognition_envelope_with_rounding_and_tie_order(self) -> None:
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            health.json(),
            {"ready": True, "model_type": "lyrics_tfidf_logreg", "model_version": "test-v1", "label_count": 15},
        )

        with patch.object(
            self.app.state.runtime.scorer,
            "score",
            return_value=self._scores(狂欢=0.123456, 孤独=0.123455),
        ):
            started = perf_counter()
            response = self.client.post(
                "/api/v1/emotion/recognize",
                json=self._request(text_lyric="任意歌词"),
            )
            elapsed_ms = (perf_counter() - started) * 1000

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "code": 200,
                "data": {
                    "top_emotion": "狂欢",
                    "top_confidence": 0.1235,
                    "second_emotion": "孤独",
                    "second_confidence": 0.1235,
                    "evidence": "基于可用歌词文本进行情绪判定。",
                    "cost_ms": response.json()["data"]["cost_ms"],
                },
                "error": "",
            },
        )
        self.assertIsInstance(response.json()["data"]["cost_ms"], int)
        self.assertGreaterEqual(response.json()["data"]["cost_ms"], 0)
        self.assertLessEqual(response.json()["data"]["cost_ms"], elapsed_ms + 100)

    def test_audio_state_changes_only_evidence_and_never_the_ranked_output(self) -> None:
        audio_app = create_app(_write_bundle(Path(self.directory.name) / "audio-bundle"))
        audio_client = TestClient(audio_app)
        scored = self._scores(狂欢=0.8, 孤独=0.2)

        def recognize_with(state: str) -> dict[str, object]:
            result = RequestAudioResult(
                "measured", {"rms_db": -3.0}
            ) if state == "measured" else RequestAudioResult("unavailable")
            with patch.object(audio_app.state.runtime.scorer, "score", return_value=scored), patch(
                "competition_emotion.service.acquire_request_audio", return_value=result
            ):
                response = audio_client.post("/api/v1/emotion/recognize", json=self._request(text_lyric="任意歌词"))
            self.assertEqual(response.status_code, 200)
            return response.json()["data"]

        measured = recognize_with("measured")
        unavailable = recognize_with("unavailable")
        self.assertEqual(
            (measured["top_emotion"], measured["top_confidence"], measured["second_emotion"], measured["second_confidence"]),
            (unavailable["top_emotion"], unavailable["top_confidence"], unavailable["second_emotion"], unavailable["second_confidence"]),
        )
        self.assertIn("音频已完成测量", measured["evidence"])
        self.assertIn("未用于当前标签", measured["evidence"])
        self.assertIn("音频未参与", unavailable["evidence"])

    def test_audio_deadline_is_measured_from_request_ingress(self) -> None:
        audio_app = create_app(_write_bundle(Path(self.directory.name) / "deadline-bundle"))
        audio_client = TestClient(audio_app)
        with patch.object(audio_app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)), patch(
            "competition_emotion.service.acquire_request_audio", return_value=RequestAudioResult("unavailable")
        ) as acquire, patch("competition_emotion.service.monotonic", return_value=100.0):
            response = audio_client.post("/api/v1/emotion/recognize", json=self._request())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(acquire.call_args.kwargs["deadline_monotonic"], 125.0)

    def test_http_audio_error_keeps_a_valid_text_response_and_hides_url(self) -> None:
        audio_app = create_app(_write_bundle(Path(self.directory.name) / "http-error-bundle"))
        audio_client = TestClient(audio_app)
        request = httpx.Request("GET", "https://audio.example.test/signed?token=secret")
        error = httpx.HTTPStatusError("secret signed URL", request=request, response=httpx.Response(503, request=request))
        with patch.object(audio_app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)), patch(
            "competition_emotion.service.acquire_request_audio", side_effect=error
        ):
            response = audio_client.post("/api/v1/emotion/recognize", json=self._request())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["top_emotion"], "狂欢")
        self.assertIn("音频未参与", response.json()["data"]["evidence"])
        self.assertNotIn("secret", response.text)

    def test_saturated_audio_capacity_skips_queue_and_keeps_health_responsive(self) -> None:
        audio_app = create_app(
            _write_bundle(Path(self.directory.name) / "saturation-bundle"),
            audio_config=ServiceAudioConfig(max_concurrent_audio=1),
        )
        entered = Event()
        release = Event()

        def blocking_audio(*_: object, **__: object) -> RequestAudioResult:
            entered.set()
            release.wait(timeout=2)
            return RequestAudioResult("measured", {"rms_db": -3.0})

        async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response, int]:
            transport = httpx.ASGITransport(app=audio_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                with patch.object(audio_app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)), patch(
                    "competition_emotion.service.acquire_request_audio", side_effect=blocking_audio
                ) as acquire:
                    first = asyncio.create_task(client.post("/api/v1/emotion/recognize", json=self._request()))
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    health = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
                    second = await asyncio.wait_for(
                        client.post("/api/v1/emotion/recognize", json=self._request(song_id="second")), timeout=0.5
                    )
                    calls = acquire.call_count
                    release.set()
                    first_response = await asyncio.wait_for(first, timeout=1)
            return first_response, second, health, calls

        first, second, health, calls = asyncio.run(exercise())
        self.assertEqual(calls, 1)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("音频未参与", second.json()["data"]["evidence"])

    def test_audio_capacity_is_shared_by_all_event_loops_for_one_app(self) -> None:
        audio_app = create_app(
            _write_bundle(Path(self.directory.name) / "shared-capacity-bundle"),
            audio_config=ServiceAudioConfig(max_concurrent_audio=1),
        )

        async def acquire_from_this_loop() -> object:
            return audio_app.state.audio_capacity.try_acquire()

        first = asyncio.run(acquire_from_this_loop())
        second = asyncio.run(acquire_from_this_loop())
        self.assertIs(first, True)
        self.assertIs(second, False)
        audio_app.state.audio_capacity.release()

    def test_cancelled_request_keeps_audio_capacity_until_worker_finishes(self) -> None:
        audio_app = create_app(
            _write_bundle(Path(self.directory.name) / "cancel-capacity-bundle"),
            audio_config=ServiceAudioConfig(max_concurrent_audio=1),
        )
        entered = Event()
        release = Event()
        capacity = audio_app.state.audio_capacity

        def blocking_audio(*_: object, **__: object) -> RequestAudioResult:
            entered.set()
            release.wait(timeout=2)
            return RequestAudioResult("measured", {"rms_db": -3.0})

        async def exercise() -> tuple[httpx.Response, httpx.Response, int]:
            transport = httpx.ASGITransport(app=audio_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                with patch.object(audio_app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)), patch(
                    "competition_emotion.service.acquire_request_audio", side_effect=blocking_audio
                ) as acquire, patch.object(capacity, "release", wraps=capacity.release) as returned:
                    first = asyncio.create_task(client.post("/api/v1/emotion/recognize", json=self._request()))
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    first.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await first
                    second = await asyncio.wait_for(
                        client.post("/api/v1/emotion/recognize", json=self._request(song_id="second")), timeout=0.5
                    )
                    self.assertEqual(acquire.call_count, 1)
                    self.assertIn("音频未参与", second.json()["data"]["evidence"])
                    release.set()
                    for _ in range(50):
                        if returned.call_count == 1:
                            break
                        await asyncio.sleep(0.01)
                    self.assertEqual(returned.call_count, 1)
                    third = await asyncio.wait_for(
                        client.post("/api/v1/emotion/recognize", json=self._request(song_id="third")), timeout=1
                    )
            return second, third, acquire.call_count

        second, third, calls = asyncio.run(exercise())
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 200)
        self.assertEqual(calls, 2)

    def test_router_404_and_405_use_safe_protocol_envelopes(self) -> None:
        method_not_allowed = self.client.get("/api/v1/emotion/recognize")
        self.assertEqual(method_not_allowed.status_code, 405)
        self.assertEqual(
            method_not_allowed.json(),
            {"code": 405, "message": "method not allowed"},
        )
        self.assertNotIn("detail", method_not_allowed.json())

        not_found = self.client.get("/missing")
        self.assertEqual(not_found.status_code, 404)
        self.assertEqual(not_found.json(), {"code": 404, "message": "not found"})
        self.assertNotIn("detail", not_found.json())

    def test_cost_ms_starts_at_body_middleware_ingress(self) -> None:
        payload = json.dumps(self._request(), ensure_ascii=False).encode("utf-8")
        sent: list[dict[str, object]] = []
        clock = {"value": 100.0}

        async def receive() -> dict[str, object]:
            clock["value"] = 100.25
            return {"type": "http.request", "body": payload, "more_body": False}

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
        with patch.object(self.app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)), patch(
            "competition_emotion.service.perf_counter", side_effect=lambda: clock["value"]
        ):
            asyncio.run(self.client.app(scope, receive, send))

        body = json.loads(sent[1]["body"])
        self.assertEqual(body["data"]["cost_ms"], 250)

    def test_top_two_ties_follow_configured_label_order(self) -> None:
        with patch.object(
            self.app.state.runtime.scorer,
            "score",
            return_value=self._scores(孤独=0.5, 狂欢=0.5),
        ):
            response = self.client.post("/api/v1/emotion/recognize", json=self._request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["top_emotion"], "狂欢")
        self.assertEqual(response.json()["data"]["second_emotion"], "孤独")
        self.assertEqual(response.json()["data"]["top_confidence"], 0.5)
        self.assertEqual(response.json()["data"]["second_confidence"], 0.5)

    def test_required_only_request_reports_song_name_as_the_only_evidence(self) -> None:
        with patch.object(self.app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)) as score:
            response = self.client.post(
                "/api/v1/emotion/recognize",
                json={
                    "audio_url": "https://example.test/audio.mp3",
                    "song_name": "新歌",
                    "song_id": "abc",
                },
            )
        self.assertEqual(response.status_code, 200)
        score.assert_called_once_with("新歌")
        evidence = response.json()["data"]["evidence"]
        self.assertEqual(evidence, "仅基于歌曲名称进行情绪判定。")
        self.assertNotIn("艺人", evidence)

    def test_lyrical_request_evidence_excludes_song_name_and_artist(self) -> None:
        with patch.object(self.app.state.runtime.scorer, "score", return_value=self._scores(狂欢=0.8, 孤独=0.2)) as score:
            response = self.client.post(
                "/api/v1/emotion/recognize",
                json=self._request(text_lyric="任意歌词"),
            )
        self.assertEqual(response.status_code, 200)
        score.assert_called_once_with("任意歌词")
        evidence = response.json()["data"]["evidence"]
        self.assertEqual(evidence, "基于可用歌词文本进行情绪判定。")
        self.assertNotIn("歌曲名称", evidence)
        self.assertNotIn("艺人", evidence)

    def test_request_validation_has_official_400_envelope(self) -> None:
        malformed = (
            {},
            self._request(audio_url="ftp://example.test/a.mp3"),
            self._request(song_name="  "),
            self._request(song_id=""),
            self._request(artists=7),
            self._request(extra="no"),
            {"song_id": "abc", "title": "old", "lyrics": "old", "audio_url": "https://example.test/a.mp3"},
        )
        for payload in malformed:
            with self.subTest(payload=payload):
                response = self.client.post("/api/v1/emotion/recognize", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"code": 400, "message": "invalid request"})

        malformed_json = self.client.post(
            "/api/v1/emotion/recognize",
            content=b'{"audio_url":',
            headers={"content-type": "application/json"},
        )
        self.assertEqual(malformed_json.status_code, 400)
        self.assertEqual(malformed_json.json(), {"code": 400, "message": "invalid request"})

    def test_rejects_invalid_scores_before_selecting_a_winner(self) -> None:
        failing_client = TestClient(self.app, raise_server_exceptions=False)
        invalid_results = (
            {label: float("nan") for label in LABELS},
            self._scores(狂欢=0.8, 孤独=True),
            self._scores(狂欢=0.8, 孤独=np.bool_(True)),
            self._scores(狂欢=0.8, 孤独=1.1),
            self._scores(狂欢=0.8, 孤独=-0.1),
        )
        for scores in invalid_results:
            with self.subTest(scores=repr(scores)), patch.object(
                self.app.state.runtime.scorer,
                "score",
                return_value=scores,
            ):
                response = failing_client.post("/api/v1/emotion/recognize", json=self._request())
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.headers["content-type"], "application/json")
            self.assertEqual(response.json(), {"code": 500, "message": "internal service error"})

    def test_body_limit_has_official_413_envelope_for_declared_and_chunked_requests(self) -> None:
        oversized = b" " * (MAX_REQUEST_BODY_BYTES + 1)
        response = self.client.post("/api/v1/emotion/recognize", content=oversized)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"code": 413, "message": "request body too large"})

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
        asyncio.run(self.client.app(scope, receive, send))
        self.assertEqual(sent[0]["status"], 413)
        self.assertEqual(json.loads(sent[1]["body"]), {"code": 413, "message": "request body too large"})

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
            with self.assertRaisesRegex(ValueError, "official labels"):
                create_app(root)

    def test_nonofficial_label_bundle_is_refused_at_initialization(self) -> None:
        with TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle")
            version_dir = root / "versions" / "test-version"
            nonofficial_labels = ("狂欢", "孤独")
            save_text_scorer(
                TextScorer().fit(
                    [
                        Song("a", frozenset({"狂欢"}), "a", "", "", "欢呼", ""),
                        Song("b", frozenset(), "b", "", "", "中性", ""),
                        Song("c", frozenset({"孤独"}), "c", "", "", "孤单", ""),
                        Song("d", frozenset(), "d", "", "", "平静", ""),
                    ],
                    nonofficial_labels,
                ),
                version_dir / "model.joblib",
            )
            report_path = version_dir / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["labels"] = list(nonofficial_labels)
            report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official labels"):
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
            copyfile(model_path, replacement_path)
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


if __name__ == "__main__":
    unittest.main()
