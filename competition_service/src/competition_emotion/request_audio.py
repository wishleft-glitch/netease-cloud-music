"""Request-scoped, disposable audio measurement acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from types import MappingProxyType
from typing import Mapping

from .audio import _validate_trusted_proxy_configuration, decode_audio, download_audio, measured_features


_MAX_REQUEST_AUDIO_BYTES = 25_000_000
_MAX_REQUEST_AUDIO_SECONDS = 45
_MAX_REQUEST_TOTAL_SECONDS = 25.0


@dataclass(frozen=True)
class RequestAudioConfig:
    """Bounded configuration for one request's disposable audio work."""

    temp_root: Path | None = None
    max_bytes: int = _MAX_REQUEST_AUDIO_BYTES
    max_seconds: int = _MAX_REQUEST_AUDIO_SECONDS
    total_seconds: float = 20.0
    proxy_url: str | None = None
    allowed_hosts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.temp_root is not None:
            object.__setattr__(self, "temp_root", Path(self.temp_root))
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or not 1 <= self.max_bytes <= _MAX_REQUEST_AUDIO_BYTES:
            raise ValueError("max_bytes must be between 1 and 25000000")
        if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, int) or not 1 <= self.max_seconds <= _MAX_REQUEST_AUDIO_SECONDS:
            raise ValueError("max_seconds must be between 1 and 45")
        if (
            isinstance(self.total_seconds, bool)
            or not isinstance(self.total_seconds, (int, float))
            or not math.isfinite(float(self.total_seconds))
            or not 0.0 < float(self.total_seconds) <= _MAX_REQUEST_TOTAL_SECONDS
        ):
            raise ValueError("total_seconds must be positive and no greater than 25")
        if self.allowed_hosts is not None:
            if isinstance(self.allowed_hosts, (str, bytes)):
                raise ValueError("allowed_hosts must be a tuple of hosts")
            object.__setattr__(self, "allowed_hosts", tuple(self.allowed_hosts))
        _validate_trusted_proxy_configuration(self.proxy_url, self.allowed_hosts)


@dataclass(frozen=True)
class RequestAudioResult:
    """The only audio fact permitted to leave request-local storage."""

    state: str
    features: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.state not in {"measured", "unavailable"}:
            raise ValueError("audio state must be measured or unavailable")
        if self.state == "measured":
            if not isinstance(self.features, Mapping) or not self.features:
                raise ValueError("measured audio result requires features")
            object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        elif self.features is not None:
            raise ValueError("unavailable audio result must not expose features")


def acquire_request_audio(
    url: str,
    config: RequestAudioConfig,
    deadline_monotonic: float | None = None,
) -> RequestAudioResult:
    """Measure one URL within a shared deadline, leaving no raw audio behind."""
    if not isinstance(config, RequestAudioConfig):
        raise ValueError("config must be a RequestAudioConfig")
    started_at = monotonic()
    deadline = started_at + float(config.total_seconds)
    if deadline_monotonic is not None:
        if not isinstance(deadline_monotonic, (int, float)) or not math.isfinite(float(deadline_monotonic)):
            raise ValueError("deadline_monotonic must be finite")
        deadline = min(deadline, float(deadline_monotonic))
    if _remaining_seconds(deadline) <= 0.0:
        return RequestAudioResult("unavailable")

    try:
        if config.temp_root is not None:
            config.temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="competition-request-audio-", dir=config.temp_root) as temporary_dir:
            audio_path = Path(temporary_dir) / "source.audio"
            remaining = _remaining_seconds(deadline)
            if remaining <= 0.0:
                return RequestAudioResult("unavailable")
            download_audio(
                url,
                audio_path,
                max_bytes=config.max_bytes,
                timeout_seconds=remaining,
                proxy_url=config.proxy_url,
                allowed_hosts=config.allowed_hosts,
            )

            remaining = _remaining_seconds(deadline)
            decode_seconds = min(config.max_seconds, int(remaining))
            if remaining <= 0.0 or decode_seconds < 1:
                return RequestAudioResult("unavailable")
            waveform, sample_rate = decode_audio(
                audio_path,
                max_seconds=decode_seconds,
                timeout_seconds=remaining,
            )

            if _remaining_seconds(deadline) <= 0.0:
                return RequestAudioResult("unavailable")
            features = measured_features(waveform, sample_rate)
            if _remaining_seconds(deadline) <= 0.0:
                return RequestAudioResult("unavailable")
            return RequestAudioResult("measured", features)
    except (ValueError, RuntimeError, OSError, TimeoutError):
        return RequestAudioResult("unavailable")


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - monotonic())
