from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import DEFAULT, patch

import numpy as np

from competition_emotion.audio import FEATURE_NAMES
from competition_emotion.audio_cache import build_audio_feature_cache, main
from competition_emotion.types import Song


FEATURES = {
    "rms_db": -12.0,
    "zero_crossing_rate": 0.1,
    "spectral_centroid_hz": 1234.0,
    "dynamic_range_db": 3.0,
}


def song(song_id: str, url: str) -> Song:
    return Song(song_id, frozenset({"孤独"}), "name", "artist", "genre", "text", url)


class AudioFeatureCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.songs = [song("2", "https://cdn.example.test/2?token=secret-two"), song("1", "https://cdn.example.test/1?token=secret-one")]

    @staticmethod
    def fake_download(url: str, destination: Path, **_: object) -> Path:
        Path(destination).write_bytes(url.encode("utf-8"))
        return Path(destination)

    @staticmethod
    def fake_decode(_: Path) -> tuple[np.ndarray, int]:
        return np.ones(512, dtype=np.float32), 22050

    @staticmethod
    def fake_features(_: np.ndarray, __: int) -> dict[str, float]:
        return dict(FEATURES)

    def _patch_extraction(self) -> object:
        return patch.multiple(
            "competition_emotion.audio_cache",
            download_audio=DEFAULT,
            decode_audio=self.fake_decode,
            measured_features=self.fake_features,
        )

    def test_successful_records_are_reused_without_re_downloading(self) -> None:
        with TemporaryDirectory() as directory, self._patch_extraction() as mocked:
            mocked["download_audio"].side_effect = self.fake_download
            cache_dir = Path(directory) / "cache"
            first = build_audio_feature_cache(self.songs, cache_dir)
            second = build_audio_feature_cache(list(reversed(self.songs)), cache_dir)

            self.assertEqual(first["attempted"], 2)
            self.assertEqual(first["succeeded"], 2)
            self.assertEqual(first["failed"], 0)
            self.assertEqual(second["attempted"], 2)
            self.assertEqual(second["skipped"], 2)
            self.assertEqual(mocked["download_audio"].call_count, 2)
            records = list((cache_dir / "records").glob("*.json"))
            self.assertEqual(len(records), 2)
            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "success")
            self.assertEqual(tuple(record["features"]), FEATURE_NAMES)
            self.assertNotIn("token=secret", records[0].read_text(encoding="utf-8"))
            self.assertFalse(list((cache_dir / "temporary").rglob("source.audio")))

    def test_changed_audio_url_invalidates_its_record(self) -> None:
        with TemporaryDirectory() as directory, self._patch_extraction() as mocked:
            mocked["download_audio"].side_effect = self.fake_download
            cache_dir = Path(directory) / "cache"
            original = [song("42", "https://cdn.example.test/original?signature=private")]
            changed = [song("42", "https://cdn.example.test/replaced?signature=private")]
            build_audio_feature_cache(original, cache_dir)
            summary = build_audio_feature_cache(changed, cache_dir)

            self.assertEqual(mocked["download_audio"].call_count, 2)
            self.assertEqual(summary["succeeded"], 1)
            record = json.loads(next((cache_dir / "records").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "success")
            self.assertNotIn("replaced?signature=private", json.dumps(record, ensure_ascii=False))

    def test_failed_record_is_saved_without_error_text_and_retried(self) -> None:
        calls = 0

        def fail_once(url: str, destination: Path, **_: object) -> Path:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError(f"signed source URL must stay private: {url}")
            return self.fake_download(url, destination)

        with TemporaryDirectory() as directory, self._patch_extraction():
            cache_dir = Path(directory) / "cache"
            with patch("competition_emotion.audio_cache.download_audio", side_effect=fail_once):
                first = build_audio_feature_cache([self.songs[0]], cache_dir)
                failed_record = next((cache_dir / "records").glob("*.json")).read_text(encoding="utf-8")
                second = build_audio_feature_cache([self.songs[0]], cache_dir)

            self.assertEqual(first["failed"], 1)
            self.assertEqual(second["succeeded"], 1)
            self.assertEqual(calls, 2)
            self.assertIn('"error_type":"ValueError"', failed_record)
            self.assertNotIn("secret-two", failed_record)
            self.assertNotIn("signed source", failed_record)

    def test_malformed_song_url_is_recorded_as_a_failure_without_stopping_worker_batch(self) -> None:
        malformed = Song("bad-url", frozenset({"孤独"}), "name", "artist", "genre", "text", 42)  # type: ignore[arg-type]
        valid = song("good-url", "https://cdn.example.test/good?token=private")
        with TemporaryDirectory() as directory, self._patch_extraction() as mocked:
            mocked["download_audio"].side_effect = self.fake_download
            cache_dir = Path(directory) / "cache"
            summary = build_audio_feature_cache([valid, malformed], cache_dir, workers=2)

            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(mocked["download_audio"].call_count, 1)
            records = [json.loads(path.read_text(encoding="utf-8")) for path in (cache_dir / "records").glob("*.json")]
            failed = next(record for record in records if record["song_id"] == "bad-url")
            self.assertEqual(failed["status"], "error")
            self.assertIsNone(failed["audio_url_sha256"])
            self.assertNotIn("42", json.dumps(failed, ensure_ascii=False))

    def test_boolean_cached_features_are_not_accepted_or_reused(self) -> None:
        source = [song("42", "https://cdn.example.test/42?signature=private")]
        with TemporaryDirectory() as directory, self._patch_extraction() as mocked:
            mocked["download_audio"].side_effect = self.fake_download
            root = Path(directory)
            for index, boolean in enumerate((True, False)):
                cache_dir = root / str(index)
                build_audio_feature_cache(source, cache_dir)
                record_path = next((cache_dir / "records").glob("*.json"))
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record["features"]["rms_db"] = boolean
                record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

                summary = build_audio_feature_cache(source, cache_dir)
                self.assertEqual(summary["skipped"], 0)
                self.assertEqual(summary["succeeded"], 1)

            self.assertEqual(mocked["download_audio"].call_count, 4)

    def test_batch_continues_after_one_song_failure_in_sorted_order(self) -> None:
        requested: list[str] = []

        def selectively_fail(url: str, destination: Path, **_: object) -> Path:
            requested.append(url)
            if url.endswith("/1?token=secret-one"):
                raise RuntimeError("download failed")
            return self.fake_download(url, destination)

        with TemporaryDirectory() as directory, self._patch_extraction():
            with patch("competition_emotion.audio_cache.download_audio", side_effect=selectively_fail):
                summary = build_audio_feature_cache(self.songs, Path(directory) / "cache", workers=1)

            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["error_counts"], {"RuntimeError": 1})
            self.assertEqual(requested, [self.songs[1].audio_url, self.songs[0].audio_url])

    def test_worker_mode_submits_at_most_the_configured_in_flight_limit(self) -> None:
        class CompletedFuture:
            def __init__(self, value: tuple[str, str | None]) -> None:
                self.value = value

            def result(self) -> tuple[str, str | None]:
                return self.value

        class RecordingExecutor:
            submissions: list[str] = []

            def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
                self.max_workers = max_workers
                self.thread_name_prefix = thread_name_prefix

            def __enter__(self) -> "RecordingExecutor":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def submit(self, function: object, item: Song) -> CompletedFuture:
                self.submissions.append(item.song_id)
                return CompletedFuture(function(item))  # type: ignore[operator]

        observed_in_flight: list[int] = []

        def complete_one(in_flight: object, **_: object) -> tuple[set[object], set[object]]:
            futures = set(in_flight)  # type: ignore[arg-type]
            observed_in_flight.append(len(futures))
            first = next(iter(futures))
            return {first}, futures - {first}

        many_songs = [song(str(index), f"https://cdn.example.test/{index}") for index in range(8)]
        with TemporaryDirectory() as directory, self._patch_extraction() as mocked:
            mocked["download_audio"].side_effect = self.fake_download
            with patch("competition_emotion.audio_cache.ThreadPoolExecutor", RecordingExecutor), patch(
                "competition_emotion.audio_cache.wait", side_effect=complete_one
            ):
                summary = build_audio_feature_cache(many_songs, Path(directory) / "cache", workers=2)

        self.assertEqual(summary["succeeded"], 8)
        self.assertEqual(RecordingExecutor.submissions, [str(index) for index in range(8)])
        self.assertTrue(observed_in_flight)
        self.assertLessEqual(max(observed_in_flight), 2)

    def test_rejects_invalid_batch_configuration_before_downloading(self) -> None:
        duplicate = [self.songs[0], song(self.songs[0].song_id, "https://cdn.example.test/new")]
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_audio_feature_cache(duplicate, Path(directory) / "cache")
            with self.assertRaisesRegex(ValueError, "limit"):
                build_audio_feature_cache(self.songs, Path(directory) / "cache", limit=0)
            with self.assertRaisesRegex(ValueError, "workers"):
                build_audio_feature_cache(self.songs, Path(directory) / "cache", workers=17)
            with self.assertRaisesRegex(ValueError, "proxy_url"):
                build_audio_feature_cache(self.songs, Path(directory) / "cache", proxy_url="http://proxy")
            with self.assertRaisesRegex(ValueError, "proxy"):
                build_audio_feature_cache(
                    self.songs, Path(directory) / "cache", proxy_url="file:///proxy", allowed_hosts=("music.163.com",)
                )


class AudioCacheCliTests(unittest.TestCase):
    def test_proxy_cli_requires_explicit_allowed_hosts(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main([
                "--workbook", "official.xlsx", "--cache-dir", "cache", "--proxy-url", "http://127.0.0.1:7897",
            ])
        self.assertEqual(raised.exception.code, 2)

    def test_allowed_host_cli_requires_explicit_proxy(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main([
                "--workbook", "official.xlsx", "--cache-dir", "cache", "--allowed-host", "music.163.com",
            ])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
