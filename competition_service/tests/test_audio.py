from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from competition_emotion.audio import (
    FEATURE_NAMES,
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
    def test_download_rejects_oversized_stream_and_keeps_existing_destination(self) -> None:
        response = MagicMock()
        response.iter_bytes.return_value = iter((b"abc", b"def"))
        context = MagicMock()
        context.__enter__.return_value = response
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "track.bin"
            destination.write_bytes(b"old")
            with patch("competition_emotion.audio.httpx.stream", return_value=context):
                with self.assertRaisesRegex(ValueError, "maximum"):
                    download_audio("https://example.test/music", destination, max_bytes=5)

            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_download_rejects_non_http_url(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "http"):
                download_audio("file:///secret", Path(directory) / "track.bin")


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
