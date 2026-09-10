from __future__ import annotations

import asyncio
import ipaddress
import os
from pathlib import Path
import socket
import subprocess
from tempfile import NamedTemporaryFile
from typing import Collection, Mapping
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx
import numpy as np
from httpcore._backends.anyio import AnyIOBackend
from httpcore._backends.base import AsyncNetworkBackend


FEATURE_NAMES = (
    "rms_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "dynamic_range_db",
)
MAX_REDIRECTS = 3
MAX_PCM_BYTES = 16_000_000
MAX_FEATURE_SECONDS = 45


class _PinnedAsyncNetworkBackend(AsyncNetworkBackend):
    """Resolve each hostname once and connect only to that checked numeric address."""

    def __init__(self) -> None:
        self._backend = AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> object:
        address = await asyncio.to_thread(_resolve_public_host, host, port)
        return await self._backend.connect_tcp(
            address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("Unix socket connections are not permitted for audio downloads")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _PinnedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, stream: object) -> None:
        self._stream = stream

    async def __aiter__(self) -> object:
        async for chunk in self._stream:  # type: ignore[union-attr]
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()  # type: ignore[union-attr]


class _PinnedAsyncTransport(httpx.AsyncBaseTransport):
    """HTTPX transport whose network backend pins DNS results for each connection."""

    def __init__(self) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            max_keepalive_connections=0,
            network_backend=_PinnedAsyncNetworkBackend(),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        core_response = await self._pool.handle_async_request(core_request)
        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_PinnedAsyncByteStream(core_response.stream),
            extensions=core_response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


def download_audio(
    url: str,
    destination: str | Path,
    max_bytes: int = 25_000_000,
    timeout_seconds: float = 12.0,
    *,
    proxy_url: str | None = None,
    allowed_hosts: Collection[str] | None = None,
) -> Path:
    """Download audio without replacing a prior file on failure.

    Direct downloads pin a globally routable DNS result to the actual socket
    connection. A proxy is an explicit deployment trust boundary: it requires
    both an operator-configured proxy URL and an exact allowlist of HTTPS audio
    hosts. Environment proxy settings are never used.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("audio URL must be a nonblank http or https URL")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    trusted_proxy = _validate_trusted_proxy_configuration(proxy_url, allowed_hosts)

    target = Path(destination)
    temporary_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            _run_download_with_deadline(
                url,
                temporary_file,
                max_bytes=max_bytes,
                timeout_seconds=float(timeout_seconds),
                trusted_proxy=trusted_proxy,
            )
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
    *,
    timeout_seconds: float | None = None,
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
    if timeout_seconds is None:
        process_timeout = float(max_seconds + 10)
    elif (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not np.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("timeout_seconds must be positive")
    else:
        process_timeout = float(timeout_seconds)
    estimated_pcm_bytes = max_seconds * sample_rate * np.dtype(np.float32).itemsize
    if estimated_pcm_bytes > MAX_PCM_BYTES:
        raise ValueError("requested PCM output exceeds the maximum PCM byte limit")

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
            timeout=process_timeout,
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
    """Compute features from at most the first 45 seconds of a valid waveform."""
    samples = np.asarray(waveform)
    if samples.ndim != 1 or samples.size < 512 or samples.dtype.kind != "f":
        raise ValueError("waveform must be a one-dimensional float array with at least 512 samples")
    if not np.isfinite(samples).all():
        raise ValueError("waveform samples must be finite")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    window_samples = min(samples.size, int(sample_rate) * MAX_FEATURE_SECONDS)
    values = samples[:window_samples].astype(np.float64, copy=False)
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

    frame_size = max(1, int(sample_rate) // 2)
    frame_rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(values[start : start + frame_size])))
            for start in range(0, values.size, frame_size)
        ],
        dtype=np.float64,
    )
    lower, upper = np.percentile(frame_rms, (5.0, 95.0), method="nearest")
    if lower <= 0.0 or upper <= 0.0 or np.isclose(lower, upper, rtol=1e-7, atol=np.finfo(np.float64).eps):
        dynamic_range_db = 0.0
    else:
        dynamic_range_db = max(0.0, float(20.0 * np.log10(upper / lower)))

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


def _run_download_with_deadline(
    url: str,
    temporary_file: object,
    *,
    max_bytes: int,
    timeout_seconds: float,
    trusted_proxy: tuple[str, frozenset[str]] | None,
) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(
            _download_to_temporary_file(
                url,
                temporary_file,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
                trusted_proxy=trusted_proxy,
            )
        )
        return
    raise RuntimeError("download_audio cannot run inside an active event loop")


async def _download_to_temporary_file(
    url: str,
    temporary_file: object,
    *,
    max_bytes: int,
    timeout_seconds: float,
    trusted_proxy: tuple[str, frozenset[str]] | None,
) -> None:
    total = 0
    current_url = url
    try:
        async with asyncio.timeout(timeout_seconds):
            if trusted_proxy is None:
                _validate_http_url(current_url)
            else:
                _validate_trusted_proxy_target(current_url, trusted_proxy[1])
            client_options: dict[str, object] = {
                "follow_redirects": False,
                "timeout": None,
                "trust_env": False,
            }
            if trusted_proxy is None:
                client_options["transport"] = _PinnedAsyncTransport()
            else:
                client_options["proxy"] = trusted_proxy[0]
            async with httpx.AsyncClient(**client_options) as client:
                for redirects in range(MAX_REDIRECTS + 1):
                    async with client.stream("GET", current_url) as response:
                        if 300 <= response.status_code < 400:
                            location = response.headers.get("Location")
                            if not location:
                                raise ValueError("audio redirect is missing Location")
                            if redirects == MAX_REDIRECTS:
                                raise ValueError("audio download exceeded redirect limit")
                            current_url = urljoin(current_url, location)
                            if trusted_proxy is None:
                                _validate_http_url(current_url)
                            else:
                                current_url = _upgrade_allowlisted_http_redirect(current_url, trusted_proxy[1])
                                _validate_trusted_proxy_target(current_url, trusted_proxy[1])
                            continue
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            if total + len(chunk) > max_bytes:
                                raise ValueError("audio download exceeds maximum size")
                            temporary_file.write(chunk)  # type: ignore[union-attr]
                            total += len(chunk)
                        break
                else:
                    raise ValueError("audio download exceeded redirect limit")
    except TimeoutError as error:
        raise TimeoutError("audio download exceeded total timeout") from error
    if total == 0:
        raise ValueError("audio download is empty")


def _validate_http_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("audio URL must be a valid public http or https URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("audio URL must be a valid public http or https URL")


def _validate_trusted_proxy_configuration(
    proxy_url: str | None,
    allowed_hosts: Collection[str] | None,
) -> tuple[str, frozenset[str]] | None:
    if proxy_url is None and allowed_hosts is None:
        return None
    if not isinstance(proxy_url, str) or not proxy_url.strip() or allowed_hosts is None:
        raise ValueError("trusted proxy mode requires both proxy_url and allowed_hosts")
    try:
        parsed_proxy = urlsplit(proxy_url)
        proxy_port = parsed_proxy.port
    except ValueError as error:
        raise ValueError("trusted proxy URL must be a valid http or https URL") from error
    if (
        parsed_proxy.scheme not in {"http", "https"}
        or not parsed_proxy.hostname
        or parsed_proxy.username is not None
        or parsed_proxy.password is not None
        or parsed_proxy.query
        or parsed_proxy.fragment
        or parsed_proxy.path not in {"", "/"}
        or proxy_port is not None and not 1 <= proxy_port <= 65535
    ):
        raise ValueError("trusted proxy URL must be a valid credential-free http or https URL")
    if isinstance(allowed_hosts, (str, bytes)):
        raise ValueError("trusted proxy allowed_hosts must be a nonempty host collection")
    try:
        normalized_hosts = frozenset(_normalize_allowlisted_host(host) for host in allowed_hosts)
    except TypeError as error:
        raise ValueError("trusted proxy allowed_hosts must be a nonempty host collection") from error
    if not normalized_hosts:
        raise ValueError("trusted proxy allowed_hosts must be nonempty")
    return proxy_url, normalized_hosts


def _normalize_allowlisted_host(host: str) -> str:
    if not isinstance(host, str) or not host or host != host.strip():
        raise ValueError("trusted proxy allowlist contains an invalid host")
    candidate = host.rstrip(".").lower()
    if not candidate:
        raise ValueError("trusted proxy allowlist contains an invalid host")
    try:
        parsed = urlsplit(f"//{candidate}")
        parsed_port = parsed.port
    except ValueError as error:
        raise ValueError("trusted proxy allowlist contains an invalid host") from error
    if (
        parsed.hostname != candidate
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed_port is not None
    ):
        raise ValueError("trusted proxy allowlist contains an invalid host")
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return candidate
    raise ValueError("trusted proxy allowlist must not contain an IP address")


def _validate_trusted_proxy_target(url: str, allowed_hosts: frozenset[str]) -> None:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("trusted proxy audio URL must be a valid HTTPS URL") from error
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise ValueError("trusted proxy audio URL must be an allowlisted HTTPS origin")
    normalized_host = host.rstrip(".").lower()
    try:
        ipaddress.ip_address(normalized_host)
    except ValueError:
        pass
    else:
        raise ValueError("trusted proxy audio URL must not use an IP address")
    if normalized_host not in allowed_hosts:
        raise ValueError("trusted proxy audio URL host is not in the exact allowlist")


def _upgrade_allowlisted_http_redirect(url: str, allowed_hosts: frozenset[str]) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("trusted proxy audio redirect must be a valid URL") from error
    if parsed.scheme != "http":
        return url
    if (
        not host
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 80}
    ):
        raise ValueError("trusted proxy audio redirect must be an allowlisted HTTPS origin")
    normalized_host = host.rstrip(".").lower()
    try:
        ipaddress.ip_address(normalized_host)
    except ValueError:
        pass
    else:
        raise ValueError("trusted proxy audio redirect must not use an IP address")
    if normalized_host not in allowed_hosts:
        raise ValueError("trusted proxy audio redirect host is not in the exact allowlist")
    return parsed._replace(scheme="https", netloc=normalized_host).geturl()


def _resolve_public_host(host: str, port: int | None) -> str:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise ValueError("audio URL host could not be resolved") from error
    if not addresses:
        raise ValueError("audio URL host could not be resolved")
    for address in addresses:
        try:
            resolved = ipaddress.ip_address(address[4][0])
        except (IndexError, ValueError) as error:
            raise ValueError("audio URL host resolved to an invalid address") from error
        if not resolved.is_global:
            raise ValueError("audio URL host must resolve only to global addresses")
    return str(ipaddress.ip_address(addresses[0][4][0]))
