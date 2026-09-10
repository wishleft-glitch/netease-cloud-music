from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from competition_emotion.audio import (
    FEATURE_NAMES,
    MAX_PCM_BYTES,
    _PinnedAsyncNetworkBackend,
    _resolve_public_host,
    decode_audio,
    download_audio,
    feature_vector,
    measured_features,
)
from competition_emotion.models import AudioScorer
from competition_emotion.types import Song


LABELS = ("热血励志", "孤独")


def song(song_id: str, labels: set[str]) -> Song:
    return Song(
        song_id=song_id,
        labels=frozenset(labels),
        name=song_id,
        artists="",
        genre="",
        text="",
        audio_url="",
    )


class DownloadAudioTests(unittest.TestCase):
    @staticmethod
    def public_resolution(*_: object, **__: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    @staticmethod
    def async_response(
        chunks: tuple[bytes, ...] = (b"audio",), *, status_code: int = 200, location: str | None = None,
        chunk_delay_seconds: float = 0.0,
    ) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        response.headers = {} if location is None else {"Location": location}

        async def aiter_bytes() -> object:
            for chunk in chunks:
                if chunk_delay_seconds:
                    await asyncio.sleep(chunk_delay_seconds)
                yield chunk

        response.aiter_bytes.side_effect = aiter_bytes
        return response

    @staticmethod
    def async_client(*responses: MagicMock, open_delay_seconds: float = 0.0) -> MagicMock:
        client = MagicMock()
        client.stream_calls = []

        async def enter() -> MagicMock:
            return client

        async def exit(*_: object) -> None:
            return None

        client.__aenter__.side_effect = enter
        client.__aexit__.side_effect = exit
        queue = list(responses)

        def stream(method: str, url: str) -> MagicMock:
            client.stream_calls.append((method, url))
            response = queue.pop(0)
            context = MagicMock()

            async def enter_response() -> MagicMock:
                if open_delay_seconds:
                    await asyncio.sleep(open_delay_seconds)
                return response

            async def exit_response(*_: object) -> None:
                return None

            context.__aenter__.side_effect = enter_response
            context.__aexit__.side_effect = exit_response
            return context

        client.stream.side_effect = stream
        return client

    def test_download_rejects_oversized_stream_and_keeps_existing_destination(self) -> None:
        client = self.async_client(self.async_response((b"abc", b"def")))
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "track.bin"
            destination.write_bytes(b"old")
            with patch("competition_emotion.audio.httpx.AsyncClient", return_value=client):
                with self.assertRaisesRegex(ValueError, "maximum"):
                    download_audio("https://example.test/music", destination, max_bytes=5)

            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_download_rejects_non_http_url(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "http"):
                download_audio("file:///secret", Path(directory) / "track.bin")

    def test_resolver_rejects_every_non_global_address_including_cgnat(self) -> None:
        addresses = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("100.64.0.1", 0)),
        ]
        with patch("competition_emotion.audio.socket.getaddrinfo", return_value=addresses):
            with self.assertRaisesRegex(ValueError, "global"):
                _resolve_public_host("cdn.example.test", 443)

    def test_pinned_connector_never_passes_a_hostname_to_the_socket_backend(self) -> None:
        backend = _PinnedAsyncNetworkBackend()
        with patch("competition_emotion.audio.socket.getaddrinfo", side_effect=self.public_resolution), patch.object(
            backend._backend, "connect_tcp", new_callable=AsyncMock
        ) as connect_tcp:
            asyncio.run(backend.connect_tcp("cdn.example.test", 443, timeout=1.0))

        self.assertEqual(connect_tcp.await_args.args[:2], ("93.184.216.34", 443))

    def test_pinned_connector_rejects_loopback_before_socket_connection(self) -> None:
        backend = _PinnedAsyncNetworkBackend()
        with patch(
            "competition_emotion.audio.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 0))]
        ), patch.object(backend._backend, "connect_tcp", new_callable=AsyncMock) as connect_tcp:
            with self.assertRaisesRegex(ValueError, "global"):
                asyncio.run(backend.connect_tcp("redirect.example.test", 443, timeout=1.0))

        connect_tcp.assert_not_awaited()

    def test_redirect_without_location_is_rejected_even_at_redirect_limit(self) -> None:
        client = self.async_client(
            *(self.async_response(status_code=302, location=location) for location in ("/one?signature=1", "/two?signature=2", "/three?signature=3", None))
        )

        with TemporaryDirectory() as directory:
            with patch("competition_emotion.audio.httpx.AsyncClient", return_value=client):
                with self.assertRaisesRegex(ValueError, "missing Location"):
                    download_audio("https://example.test/music?signature=origin", Path(directory) / "track.bin")

        self.assertEqual(client.stream_calls[1][1], "https://example.test/one?signature=1")

    def test_download_creates_nested_destination_parent(self) -> None:
        client = self.async_client(self.async_response())
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new" / "nested" / "track.bin"
            with patch("competition_emotion.audio.httpx.AsyncClient", return_value=client):
                result = download_audio("https://example.test/music", destination)
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), b"audio")

    def test_download_deadline_cancels_a_blocking_stream_open(self) -> None:
        client = self.async_client(self.async_response(), open_delay_seconds=0.05)
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "track.bin"
            destination.write_bytes(b"old")
            with patch("competition_emotion.audio.httpx.AsyncClient", return_value=client):
                with self.assertRaisesRegex(TimeoutError, "total timeout"):
                    download_audio("https://example.test/music", destination, timeout_seconds=0.01)

            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_download_deadline_cancels_a_blocking_body_read(self) -> None:
        client = self.async_client(self.async_response((b"audio",), chunk_delay_seconds=0.05))
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "track.bin"
            destination.write_bytes(b"old")
            with patch("competition_emotion.audio.httpx.AsyncClient", return_value=client):
                with self.assertRaisesRegex(TimeoutError, "total timeout"):
                    download_audio("https://example.test/music", destination, timeout_seconds=0.01)

            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class DecodeAudioTests(unittest.TestCase):
    def test_decode_uses_bounded_mono_ffmpeg_invocation(self) -> None:
        result = MagicMock(returncode=0, stdout=np.array([0.1, -0.2], dtype=np.float32).tobytes(), stderr=b"")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.mp3"
            path.write_bytes(b"input")
            with patch("competition_emotion.audio.subprocess.run", return_value=result) as mocked_run:
                waveform, sample_rate = decode_audio(path, max_seconds=3, sample_rate=8000)

        self.assertEqual(sample_rate, 8000)
        np.testing.assert_allclose(waveform, [0.1, -0.2])
        arguments, keyword_arguments = mocked_run.call_args
        self.assertIn("-t", arguments[0])
        self.assertEqual(arguments[0][arguments[0].index("-t") + 1], "3")
        self.assertEqual(keyword_arguments["timeout"], 13)
        self.assertNotIn("shell", keyword_arguments)

    def test_decode_rejects_failed_or_missing_ffmpeg(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.mp3"
            path.write_bytes(b"input")
            failed = MagicMock(returncode=1, stdout=b"", stderr=b"x" * 500)
            with patch("competition_emotion.audio.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(ValueError, "ffmpeg"):
                    decode_audio(path)
            with patch("competition_emotion.audio.subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg.*prerequisite"):
                    decode_audio(path)

    def test_decode_rejects_misaligned_and_overlong_output(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.mp3"
            path.write_bytes(b"input")
            misaligned = MagicMock(returncode=0, stdout=b"abc", stderr=b"")
            with patch("competition_emotion.audio.subprocess.run", return_value=misaligned):
                with self.assertRaisesRegex(ValueError, "misaligned"):
                    decode_audio(path)
            overlong = MagicMock(
                returncode=0,
                stdout=np.zeros(8001, dtype=np.float32).tobytes(),
                stderr=b"",
            )
            with patch("competition_emotion.audio.subprocess.run", return_value=overlong):
                with self.assertRaisesRegex(ValueError, "duration"):
                    decode_audio(path, max_seconds=1, sample_rate=8000)

    def test_decode_rejects_pcm_estimate_before_starting_ffmpeg(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.mp3"
            path.write_bytes(b"input")
            with patch("competition_emotion.audio.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "PCM"):
                    decode_audio(path, max_seconds=600, sample_rate=48000)

        self.assertLess(MAX_PCM_BYTES, 600 * 48000 * 4)
        run.assert_not_called()


class FeatureTests(unittest.TestCase):
    def test_sine_features_are_finite_and_vector_uses_fixed_order(self) -> None:
        sample_rate = 22050
        waveform = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(np.float32)
        features = measured_features(waveform, sample_rate)

        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        vector = feature_vector(features)
        self.assertEqual(vector.dtype, np.float64)
        self.assertEqual(vector.shape, (4,))
        np.testing.assert_allclose(vector, [features[name] for name in FEATURE_NAMES])

    def test_silence_has_zero_dynamic_range(self) -> None:
        features = measured_features(np.zeros(512, dtype=np.float32), 22050)

        self.assertEqual(features["dynamic_range_db"], 0.0)

    def test_leading_silence_and_constant_tone_has_stable_dynamic_range(self) -> None:
        sample_rate = 8000
        waveform = np.concatenate(
            (np.zeros(sample_rate // 2, dtype=np.float32), np.full(sample_rate * 3 // 2, 0.5, dtype=np.float32))
        )

        features = measured_features(waveform, sample_rate)

        self.assertEqual(features["dynamic_range_db"], 0.0)

    def test_extra_long_waveform_uses_a_bounded_feature_window(self) -> None:
        sample_rate = 8000
        waveform = np.sin(2 * np.pi * 200 * np.arange(sample_rate * 46) / sample_rate).astype(np.float32)
        with patch("competition_emotion.audio.np.fft.rfft", wraps=np.fft.rfft) as rfft:
            features = measured_features(waveform, sample_rate)

        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertEqual(rfft.call_args.args[0].size, sample_rate * 45)

    def test_feature_functions_reject_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            measured_features(np.zeros((512, 1), dtype=np.float32), 22050)
        with self.assertRaises(ValueError):
            measured_features(np.full(512, np.nan, dtype=np.float32), 22050)
        with self.assertRaises(ValueError):
            feature_vector({name: 0.0 for name in FEATURE_NAMES[:-1]})
        with self.assertRaises(ValueError):
            feature_vector({name: 0.0 for name in FEATURE_NAMES} | {"extra": 0.0})


class AudioScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            (song("1", {"热血励志"}), {"rms_db": -5.0, "zero_crossing_rate": 0.30, "spectral_centroid_hz": 7000.0, "dynamic_range_db": 20.0}),
            (song("2", {"热血励志"}), {"rms_db": -6.0, "zero_crossing_rate": 0.28, "spectral_centroid_hz": 6800.0, "dynamic_range_db": 18.0}),
            (song("3", {"孤独"}), {"rms_db": -35.0, "zero_crossing_rate": 0.03, "spectral_centroid_hz": 500.0, "dynamic_range_db": 3.0}),
            (song("4", {"孤独"}), {"rms_db": -32.0, "zero_crossing_rate": 0.04, "spectral_centroid_hz": 600.0, "dynamic_range_db": 4.0}),
        ]
        self.scorer = AudioScorer().fit(self.samples, LABELS)

    def test_scores_bilingual_labels_and_preserves_label_order(self) -> None:
        scores = self.scorer.score({"rms_db": -4.0, "zero_crossing_rate": 0.31, "spectral_centroid_hz": 7200.0, "dynamic_range_db": 21.0})

        self.assertEqual(tuple(scores), LABELS)
        self.assertGreater(scores["热血励志"], scores["孤独"])
        self.assertTrue(all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in scores.values()))
        self.assertEqual(self.scorer.score_many([]).shape, (0, len(LABELS)))

    def test_rejects_unknown_labels_and_invalid_feature_records(self) -> None:
        unknown = [(song("unknown", {"未知"}), self.samples[0][1]), *self.samples[1:]]
        with self.assertRaisesRegex(ValueError, "未知"):
            AudioScorer().fit(unknown, LABELS)
        duplicate = [self.samples[0], (song("1", {"孤独"}), self.samples[2][1]), *self.samples[2:]]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            AudioScorer().fit(duplicate, LABELS)
        invalid = [(self.samples[0][0], {"rms_db": 0.0}), *self.samples[1:]]
        with self.assertRaises(ValueError):
            AudioScorer().fit(invalid, LABELS)


if __name__ == "__main__":
    unittest.main()
