"""Reproducible, resumable cache construction for measured audio features.

The cache deliberately stores measurements only.  Audio is downloaded into a
per-song temporary directory and removed before the next record is published.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from urllib.parse import quote

from .audio import (
    FEATURE_NAMES,
    _validate_trusted_proxy_configuration,
    decode_audio,
    download_audio,
    feature_vector,
    measured_features,
)
from .data import load_official_songs
from .types import Song


CACHE_SCHEMA_VERSION = 1
EXTRACTION_CONFIG_VERSION = "audio-measurements-v1"
MAX_CACHE_WORKERS = 16
MAX_ERROR_TYPES = 20


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


EXTRACTION_CONFIG: dict[str, object] = {
    "version": EXTRACTION_CONFIG_VERSION,
    "decoder": "ffmpeg",
    "max_seconds": 45,
    "sample_rate": 22050,
    "feature_names": list(FEATURE_NAMES),
}
EXTRACTION_CONFIG_SHA256 = hashlib.sha256(_canonical_json(EXTRACTION_CONFIG).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":")) + "\n"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _safe_song_id(song_id: str) -> str:
    """Return a bounded Windows-safe filename derived from a song ID."""
    readable = quote(song_id, safe="-_.")[:80].strip(".") or "song"
    digest = hashlib.sha256(song_id.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}"


def _url_hash(audio_url: str) -> str:
    return hashlib.sha256(audio_url.strip().encode("utf-8")).hexdigest()


def _record_path(cache_dir: Path, song_id: str) -> Path:
    return cache_dir / "records" / f"{_safe_song_id(song_id)}.json"


def _is_reusable_success(record: object, song: Song) -> bool:
    if not isinstance(record, dict):
        return False
    if (
        record.get("schema_version") != CACHE_SCHEMA_VERSION
        or record.get("song_id") != song.song_id
        or record.get("audio_url_sha256") != _url_hash(song.audio_url)
        or record.get("extraction_config_sha256") != EXTRACTION_CONFIG_SHA256
        or record.get("status") != "success"
    ):
        return False
    features = record.get("features")
    try:
        return tuple(feature_vector(features).shape) == (len(FEATURE_NAMES),)
    except ValueError:
        return False


def _cached_success(record_path: Path, song: Song) -> bool:
    try:
        with record_path.open("r", encoding="utf-8") as source:
            record = json.load(source)
    except (OSError, json.JSONDecodeError):
        return False
    return _is_reusable_success(record, song)


def _error_type(error: Exception) -> str:
    # Exception text can contain a signed download URL.  Persist only its
    # stable class name, which remains useful for aggregate troubleshooting.
    name = type(error).__name__
    return name if name.isidentifier() else "AudioExtractionError"


def _extract_one(
    song: Song,
    cache_dir: Path,
    *,
    proxy_url: str | None,
    allowed_hosts: Sequence[str] | None,
    timeout_seconds: float,
) -> tuple[str, str | None]:
    """Build one record and return (status, error type when failed)."""
    record_path = _record_path(cache_dir, song.song_id)
    if _cached_success(record_path, song):
        return "skipped", None

    base_record: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "song_id": song.song_id,
        "audio_url_sha256": _url_hash(song.audio_url),
        "extraction_config": EXTRACTION_CONFIG,
        "extraction_config_sha256": EXTRACTION_CONFIG_SHA256,
    }
    temporary_root = cache_dir / "temporary"
    try:
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="audio-", dir=temporary_root) as temporary_dir:
            audio_path = Path(temporary_dir) / "source.audio"
            download_audio(
                song.audio_url,
                audio_path,
                timeout_seconds=timeout_seconds,
                proxy_url=proxy_url,
                allowed_hosts=allowed_hosts,
            )
            waveform, sample_rate = decode_audio(audio_path)
            features = measured_features(waveform, sample_rate)
            # feature_vector enforces exact field names and finite values before
            # a cache success becomes reusable.
            feature_vector(features)
        _atomic_json(
            record_path, base_record | {"status": "success", "features": features}, sort_keys=False
        )
        return "succeeded", None
    except Exception as error:
        error_name = _error_type(error)
        _atomic_json(
            record_path, base_record | {"status": "error", "error_type": error_name}, sort_keys=False
        )
        return "failed", error_name


def _validate_inputs(
    songs: Sequence[Song],
    *,
    proxy_url: str | None,
    allowed_hosts: Sequence[str] | None,
    limit: int | None,
    workers: int,
    timeout_seconds: float,
) -> None:
    song_ids = [song.song_id for song in songs]
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("songs contain duplicate song IDs")
    if any(not isinstance(song_id, str) or not song_id for song_id in song_ids):
        raise ValueError("song IDs must be nonblank strings")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer when specified")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_CACHE_WORKERS:
        raise ValueError(f"workers must be an integer between 1 and {MAX_CACHE_WORKERS}")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    if (proxy_url is None) != (allowed_hosts is None):
        raise ValueError("proxy_url and allowed_hosts must be supplied together")
    if allowed_hosts is not None:
        if isinstance(allowed_hosts, str) or not allowed_hosts or any(not isinstance(host, str) or not host.strip() for host in allowed_hosts):
            raise ValueError("allowed_hosts must contain one or more nonblank hosts")
        # Reuse the downloader's pure validation so malformed proxy URLs and
        # non-exact host entries fail before any song is attempted.
        _validate_trusted_proxy_configuration(proxy_url, allowed_hosts)


def _bounded_error_counts(errors: Sequence[str]) -> dict[str, int]:
    counts = Counter(errors)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    visible = ordered[:MAX_ERROR_TYPES]
    result = {name: count for name, count in visible}
    remainder = sum(count for _, count in ordered[MAX_ERROR_TYPES:])
    if remainder:
        result["other"] = remainder
    return result


def build_audio_feature_cache(
    songs: Sequence[Song],
    cache_dir: str | Path,
    *,
    proxy_url: str | None = None,
    allowed_hosts: Sequence[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Build a deterministic feature cache, retrying only non-reusable records."""
    materialized = list(songs)
    _validate_inputs(
        materialized,
        proxy_url=proxy_url,
        allowed_hosts=allowed_hosts,
        limit=limit,
        workers=workers,
        timeout_seconds=timeout_seconds,
    )
    selected = sorted(materialized, key=lambda song: song.song_id)
    if limit is not None:
        selected = selected[:limit]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    def run(song: Song) -> tuple[str, str | None]:
        return _extract_one(
            song, root, proxy_url=proxy_url, allowed_hosts=allowed_hosts,
            timeout_seconds=float(timeout_seconds),
        )

    if workers == 1:
        outcomes = [run(song) for song in selected]
    else:
        # executor.map preserves selected ordering while bounding in-flight work
        # to the explicit operator-selected worker limit.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audio-cache") as executor:
            outcomes = list(executor.map(run, selected))

    statuses = Counter(status for status, _ in outcomes)
    errors = [error for _, error in outcomes if error is not None]
    summary: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "extraction_config_sha256": EXTRACTION_CONFIG_SHA256,
        "attempted": len(selected),
        "succeeded": statuses["succeeded"],
        "skipped": statuses["skipped"],
        "failed": statuses["failed"],
        "error_counts": _bounded_error_counts(errors),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a resumable audio feature cache")
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--proxy-url")
    parser.add_argument("--allowed-host", action="append", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--timeout-seconds", default=30.0, type=float)
    arguments = parser.parse_args(argv)
    if arguments.proxy_url is not None and not arguments.allowed_host:
        parser.error("--proxy-url requires at least one --allowed-host")
    if arguments.proxy_url is None and arguments.allowed_host:
        parser.error("--allowed-host requires --proxy-url")

    songs = load_official_songs(arguments.workbook)
    summary = build_audio_feature_cache(
        songs,
        arguments.cache_dir,
        proxy_url=arguments.proxy_url,
        allowed_hosts=arguments.allowed_host,
        limit=arguments.limit,
        workers=arguments.workers,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - covered via main()
    raise SystemExit(main())
