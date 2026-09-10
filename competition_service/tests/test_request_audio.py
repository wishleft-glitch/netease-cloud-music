from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from competition_emotion.request_audio import RequestAudioConfig, acquire_request_audio


class RequestAudioTests(unittest.TestCase):
    def test_success_returns_immutable_measurements_and_removes_raw_audio(self) -> None:
        with TemporaryDirectory() as directory:
            temporary_root = Path(directory) / "raw"
            waveform = np.ones(512, dtype=np.float32)
            with patch("competition_emotion.request_audio.download_audio") as download, patch(
                "competition_emotion.request_audio.decode_audio", return_value=(waveform, 22050)
            ), patch(
                "competition_emotion.request_audio.measured_features", return_value={"rms_db": -3.0}
            ):
                result = acquire_request_audio(
                    "https://audio.example.test/signed?token=secret",
                    RequestAudioConfig(temp_root=temporary_root),
                )

            self.assertEqual(result.state, "measured")
            self.assertEqual(dict(result.features or {}), {"rms_db": -3.0})
            with self.assertRaises(TypeError):
                (result.features or {})["rms_db"] = 0.0
            self.assertEqual(list(temporary_root.iterdir()), [])
            download.assert_called_once()

    def test_download_failure_is_unavailable_and_cleans_the_temporary_directory(self) -> None:
        with TemporaryDirectory() as directory:
            temporary_root = Path(directory) / "raw"
            with patch("competition_emotion.request_audio.download_audio", side_effect=OSError("signed?secret")), patch(
                "competition_emotion.request_audio.decode_audio"
            ) as decode:
                result = acquire_request_audio(
                    "https://audio.example.test/signed?token=secret",
                    RequestAudioConfig(temp_root=temporary_root),
                )

            self.assertEqual(result.state, "unavailable")
            self.assertIsNone(result.features)
            self.assertEqual(list(temporary_root.iterdir()), [])
            decode.assert_not_called()

    def test_decoder_failure_is_unavailable_and_cleans_the_temporary_directory(self) -> None:
        with TemporaryDirectory() as directory:
            temporary_root = Path(directory) / "raw"
            with patch("competition_emotion.request_audio.download_audio"), patch(
                "competition_emotion.request_audio.decode_audio", side_effect=RuntimeError("decoder failed")
            ):
                result = acquire_request_audio(
                    "https://audio.example.test/signed?token=secret",
                    RequestAudioConfig(temp_root=temporary_root),
                )

            self.assertEqual(result.state, "unavailable")
            self.assertIsNone(result.features)
            self.assertEqual(list(temporary_root.iterdir()), [])

    def test_expired_deadline_skips_all_audio_calls(self) -> None:
        with patch("competition_emotion.request_audio.monotonic", return_value=10.0), patch(
            "competition_emotion.request_audio.download_audio"
        ) as download, patch("competition_emotion.request_audio.decode_audio") as decode:
            result = acquire_request_audio(
                "https://audio.example.test/signed?token=secret",
                RequestAudioConfig(),
                deadline_monotonic=10.0,
            )

        self.assertEqual(result.state, "unavailable")
        self.assertIsNone(result.features)
        download.assert_not_called()
        decode.assert_not_called()

    def test_proxy_configuration_is_forwarded_to_the_downloader(self) -> None:
        with patch("competition_emotion.request_audio.download_audio") as download, patch(
            "competition_emotion.request_audio.decode_audio", return_value=(np.ones(512, dtype=np.float32), 22050)
        ), patch("competition_emotion.request_audio.measured_features", return_value={"rms_db": -3.0}):
            acquire_request_audio(
                "https://audio.example.test/signed?token=secret",
                RequestAudioConfig(
                    proxy_url="http://proxy.example.test",
                    allowed_hosts=("audio.example.test",),
                ),
            )

        self.assertEqual(download.call_args.kwargs["proxy_url"], "http://proxy.example.test")
        self.assertEqual(download.call_args.kwargs["allowed_hosts"], ("audio.example.test",))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RequestAudioConfig(max_bytes=0)
        with self.assertRaises(ValueError):
            RequestAudioConfig(max_seconds=0)
        with self.assertRaises(ValueError):
            RequestAudioConfig(total_seconds=0)
        with self.assertRaises(ValueError):
            RequestAudioConfig(proxy_url="http://proxy.example.test")
        with self.assertRaises(ValueError):
            RequestAudioConfig(allowed_hosts=("audio.example.test",))

    def test_http_download_error_becomes_sanitized_unavailable_result(self) -> None:
        request = httpx.Request("GET", "https://audio.example.test/signed?token=secret")
        response = httpx.Response(503, request=request)
        error = httpx.HTTPStatusError("secret signed URL", request=request, response=response)
        with patch("competition_emotion.request_audio.download_audio", side_effect=error):
            result = acquire_request_audio(
                "https://audio.example.test/signed?token=secret",
                RequestAudioConfig(),
            )

        self.assertEqual(result.state, "unavailable")
        self.assertIsNone(result.features)
        self.assertNotIn("secret", repr(result))


if __name__ == "__main__":
    unittest.main()
