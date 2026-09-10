from __future__ import annotations

import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Mapping
from urllib.parse import urlsplit

import httpx
import numpy as np


FEATURE_NAMES = (
    "rms_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "dynamic_range_db",
)


def download_audio(
    url: str,
    destination: str | Path,
    max_bytes: int = 25_000_000,
    timeout_seconds: float = 12.0,
) -> Path:
    """Download an HTTP(S) audio payload without replacing a prior file on failure."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("audio URL must be a nonblank http or https URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("audio URL must be a nonblank http or https URL")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    target = Path(destination)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            total = 0
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=timeout_seconds
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    if total + len(chunk) > max_bytes:
                        raise ValueError("audio download exceeds maximum size")
                    temporary_file.write(chunk)
                    total += len(chunk)
            if total == 0:
                raise ValueError("audio download is empty")
        os.replace(temporary_path, target)
        return target
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def decode_audio(
    path: str | Path,
    ffmpeg_path: str = "ffmpeg",
    max_seconds: int = 45,
    sample_rate: int = 22050,
) -> tuple[np.ndarray, int]:
    """Decode a bounded mono float32 waveform through ffmpeg."""
    audio_path = Path(path)
    if not audio_path.exists() or not audio_path.is_file():
        raise ValueError("audio path must exist and be a file")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not 1 <= max_seconds <= 600:
        raise ValueError("max_seconds must be between 1 and 600")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 48000:
        raise ValueError("sample_rate must be between 8000 and 48000")
    if not isinstance(ffmpeg_path, str) or not ffmpeg_path.strip():
        raise ValueError("ffmpeg_path must be nonblank")

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(audio_path),
        "-t",
        str(max_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max_seconds + 10,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg prerequisite is unavailable") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("ffmpeg decoding timed out") from error

    stderr = _bounded_stderr(result.stderr)
    if result.returncode != 0:
        message = "ffmpeg decoding failed"
        if stderr:
            message += f": {stderr}"
        raise ValueError(message)
    if not result.stdout:
        raise ValueError("ffmpeg decoding produced an empty waveform")
    if len(result.stdout) % np.dtype(np.float32).itemsize:
        raise ValueError("ffmpeg decoding produced a misaligned waveform")

    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if waveform.size > max_seconds * sample_rate:
        raise ValueError("ffmpeg decoding exceeded the requested duration")
    if not np.isfinite(waveform).all():
        raise ValueError("ffmpeg decoding produced nonfinite samples")
    return waveform, sample_rate


def measured_features(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute a fixed, finite feature set from a real decoded waveform."""
    samples = np.asarray(waveform)
    if samples.ndim != 1 or samples.size < 512 or samples.dtype.kind != "f":
        raise ValueError("waveform must be a one-dimensional float array with at least 512 samples")
    if not np.isfinite(samples).all():
        raise ValueError("waveform samples must be finite")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    values = samples.astype(np.float64, copy=False)
    amplitude = np.abs(values)
    rms = float(np.sqrt(np.mean(np.square(values))))
    rms_db = -120.0 if rms == 0.0 else float(20.0 * np.log10(rms))
    zero_crossing_rate = float(np.mean(values[1:] * values[:-1] < 0.0))

    spectrum = np.abs(np.fft.rfft(values)) ** 2
    total_spectrum = float(np.sum(spectrum))
    if total_spectrum == 0.0:
        spectral_centroid_hz = 0.0
    else:
        frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
        spectral_centroid_hz = float(np.dot(frequencies, spectrum) / total_spectrum)

    peak = float(np.max(amplitude))
    if peak == 0.0:
        dynamic_range_db = 0.0
    else:
        floor, ceiling = np.percentile(amplitude, (5.0, 95.0))
        if floor <= 0.0:
            floor = float(np.min(amplitude[amplitude > 0.0]))
        dynamic_range_db = max(0.0, float(20.0 * np.log10(ceiling / floor)))

    features = {
        "rms_db": rms_db,
        "zero_crossing_rate": max(0.0, min(1.0, zero_crossing_rate)),
        "spectral_centroid_hz": max(0.0, spectral_centroid_hz),
        "dynamic_range_db": dynamic_range_db,
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("waveform measurements must be finite")
    return {name: float(features[name]) for name in FEATURE_NAMES}


def feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Validate and order measured features for model input."""
    if not isinstance(features, Mapping) or set(features) != set(FEATURE_NAMES):
        raise ValueError("features must contain exactly the supported feature names")
    try:
        vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("feature values must be finite numbers") from error
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("feature values must be finite numbers")
    return vector


def _bounded_stderr(stderr: bytes | str | None) -> str:
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace")[:300]
    return str(stderr)[:300]
