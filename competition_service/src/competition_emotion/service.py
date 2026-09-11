"""Production HTTP interface for a trusted lyrics-model bundle."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
from numbers import Real
import stat
from threading import BoundedSemaphore
from time import monotonic, perf_counter
from typing import Any, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
import uvicorn
import httpx
import numpy as np

from .models import TextScorer, load_text_scorer
from .constants import LABELS
from .evidence import build_evidence
from .iteration import TraceRecord, append_trace
from .lyrics import compose_lyrics
from .rubric import DEFAULT_RUBRIC_PATH, load_rubric
from .request_audio import RequestAudioConfig, RequestAudioResult, acquire_request_audio
from .semantic import SemanticReviewConfig, review_candidates
from .train import BUNDLE_POINTER_SCHEMA_VERSION, MODEL_TYPE, REPORT_SCHEMA_VERSION
from .types import Song


_MAX_POINTER_BYTES = 64 * 1024
_MAX_REPORT_BYTES = 1024 * 1024
MAX_REQUEST_BODY_BYTES = 110_000
MAX_INTERNAL_REQUEST_SECONDS = 25.0
_REQUEST_STARTED_AT_SCOPE_KEY = "competition_emotion.request_started_at"
_REQUEST_MONOTONIC_STARTED_AT_SCOPE_KEY = "competition_emotion.request_monotonic_started_at"


def _append_trace_safely(path: Path, record: TraceRecord) -> None:
    """Tracing is best effort and must never turn a successful prediction into a 500."""
    try:
        append_trace(path, record)
    except (OSError, ValueError):
        return


class RecognizeRequest(BaseModel):
    """The bounded official synchronous-recognition payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    audio_url: StrictStr = Field(max_length=1024)
    song_name: StrictStr = Field(max_length=200)
    song_id: StrictStr = Field(max_length=200)
    album_name: StrictStr | None = Field(default=None, max_length=200)
    artists: StrictStr | None = Field(default=None, max_length=500)
    text_lyric: StrictStr | None = Field(default=None, max_length=5000)
    lrc_lyric: StrictStr | None = Field(default=None, max_length=5000)
    lrc_translation: StrictStr | None = Field(default=None, max_length=5000)

    @field_validator("audio_url", "song_name", "song_id")
    @classmethod
    def required_string_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required field must not be blank")
        return value

    @field_validator("audio_url")
    @classmethod
    def audio_url_must_be_http_or_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("audio_url must be an http or https URL")
        return value


class RecognizeData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    top_emotion: str
    top_confidence: float
    second_emotion: str
    second_confidence: float
    evidence: str = Field(max_length=500)
    cost_ms: int = Field(ge=0)


class RecognizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    code: int
    data: RecognizeData
    error: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ready: bool
    model_type: str
    model_version: str
    label_count: int


@dataclass(frozen=True)
class _Runtime:
    scorer: TextScorer
    model_type: str
    model_version: str


@dataclass(frozen=True)
class ServiceAudioConfig:
    """Audio behavior for the synchronous service; ``None`` disables it in tests."""

    request_config: RequestAudioConfig = field(default_factory=RequestAudioConfig)
    request_deadline_seconds: float = MAX_INTERNAL_REQUEST_SECONDS
    max_concurrent_audio: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.request_config, RequestAudioConfig):
            raise ValueError("request_config must be a RequestAudioConfig")
        if (
            isinstance(self.request_deadline_seconds, bool)
            or not isinstance(self.request_deadline_seconds, (int, float))
            or not math.isfinite(float(self.request_deadline_seconds))
            or not 0.0 < float(self.request_deadline_seconds) <= MAX_INTERNAL_REQUEST_SECONDS
        ):
            raise ValueError("request_deadline_seconds must be positive and no greater than 25")
        if (
            isinstance(self.max_concurrent_audio, bool)
            or not isinstance(self.max_concurrent_audio, int)
            or self.max_concurrent_audio < 1
        ):
            raise ValueError("max_concurrent_audio must be a positive integer")


_DEFAULT_SERVICE_AUDIO_CONFIG = ServiceAudioConfig()


class _AudioCapacityController:
    """A non-queuing audio permit pool shared by every loop in one app."""

    def __init__(self, maximum: int) -> None:
        self._semaphore = BoundedSemaphore(maximum)

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()


def _release_audio_capacity(capacity: _AudioCapacityController):
    """Create the one completion callback that returns a held audio permit."""

    def release_when_done(task: asyncio.Task[RequestAudioResult]) -> None:
        try:
            task.exception()
        except BaseException:
            pass
        finally:
            capacity.release()

    return release_when_done


class _RequestBodyLimitMiddleware:
    """Reject oversized request bodies before and while FastAPI parses JSON."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        scope[_REQUEST_STARTED_AT_SCOPE_KEY] = perf_counter()
        scope[_REQUEST_MONOTONIC_STARTED_AT_SCOPE_KEY] = monotonic()
        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > self.max_bytes:
                await self._too_large(scope, receive, send)
                return

        # Buffer no more than the configured cap, then replay the bounded
        # sequence.  Pre-reading is intentional: if a later chunk crosses the
        # cap, FastAPI never parses it and cannot convert our rejection to 400.
        received_size = 0
        buffered_messages: list[dict[str, Any]] = []
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > self.max_bytes:
                    await self._too_large(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        async def bounded_receive() -> dict[str, Any]:
            if buffered_messages:
                return buffered_messages.pop(0)
            return {"type": "http.disconnect"}

        await self.app(scope, bounded_receive, send)

    @staticmethod
    async def _too_large(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await JSONResponse(
            status_code=413, content={"code": 413, "message": "request body too large"}
        )(scope, receive, send)


def _same_file_identity(before: os.stat_result, opened: os.stat_result) -> bool:
    return before.st_dev == opened.st_dev and before.st_ino == opened.st_ino


def _open_regular_file(path: Path, *, description: str):
    """Open one stable regular file without following a final symlink.

    The lstat/fstat identity check makes replacement between the check and the
    open fail.  After that, consumers use the open handle rather than looking
    the path up again, so a later rename cannot switch the artifact being read.
    """
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{description} must be a regular file") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{description} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} must be a regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(before, opened):
            raise ValueError(f"{description} changed while opening")
        return os.fdopen(descriptor, "rb", closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _read_json_file(path: Path, *, max_bytes: int, description: str) -> dict[str, Any]:
    with _open_regular_file(path, description=description) as raw_file:
        try:
            opened = os.fstat(raw_file.fileno())
            if opened.st_size > max_bytes:
                raise ValueError(f"{description} is too large")
            raw_value = raw_file.read(max_bytes + 1)
            if len(raw_value) > max_bytes:
                raise ValueError(f"{description} is too large")
            value = json.loads(raw_value.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {description}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description}")
    return value


def _safe_bundle_dir(bundle_root: Path) -> Path:
    """Resolve the operator-supplied pointer without accepting path escapes."""
    raw_root = Path(bundle_root)
    try:
        raw_root_stat = raw_root.lstat()
    except OSError as error:
        raise ValueError("bundle root does not exist") from error
    if not stat.S_ISDIR(raw_root_stat.st_mode):
        raise ValueError("bundle root must be a regular directory")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("bundle root does not exist") from error

    pointer = _read_json_file(
        root / "current.json", max_bytes=_MAX_POINTER_BYTES, description="bundle pointer"
    )
    if set(pointer) != {"pointer_schema_version", "active_bundle"}:
        raise ValueError("invalid bundle pointer schema")
    if pointer.get("pointer_schema_version") != BUNDLE_POINTER_SCHEMA_VERSION:
        raise ValueError("unsupported bundle pointer schema")
    active_bundle = pointer.get("active_bundle")
    if not isinstance(active_bundle, str):
        raise ValueError("bundle pointer has invalid active_bundle")
    # Use a portable JSON path form and require exactly versions/<opaque-id>.
    if "\\" in active_bundle:
        raise ValueError("bundle pointer has invalid active_bundle")
    relative = PurePosixPath(active_bundle)
    if (
        relative.is_absolute()
        or relative.parts[0:1] != ("versions",)
        or len(relative.parts) != 2
        or relative.parts[1] in {"", ".", ".."}
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise ValueError("bundle pointer has invalid active_bundle")

    versions_dir = root / "versions"
    candidate = versions_dir / relative.parts[1]
    try:
        versions_stat = versions_dir.lstat()
        candidate_stat = candidate.lstat()
    except OSError as error:
        raise ValueError("active bundle must be a regular directory") from error
    if not stat.S_ISDIR(versions_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
        raise ValueError("active bundle must be a regular directory")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError("active bundle escapes the bundle root") from error
    return resolved


def _load_runtime(bundle_root: Path) -> _Runtime:
    bundle_dir = _safe_bundle_dir(bundle_root)
    model_path = bundle_dir / "model.joblib"
    report = _read_json_file(
        bundle_dir / "report.json", max_bytes=_MAX_REPORT_BYTES, description="bundle report"
    )
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported bundle report schema")
    source = report.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {"file_name", "sha256", "rows"}
        or not isinstance(source["file_name"], str)
        or not source["file_name"].strip()
        or not isinstance(source["sha256"], str)
        or len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
        or not isinstance(source["rows"], int)
        or isinstance(source["rows"], bool)
        or source["rows"] < 0
    ):
        raise ValueError("invalid bundle report source provenance")
    if report.get("model_type") != MODEL_TYPE:
        raise ValueError("bundle model type is not supported by this service")
    model_version = report.get("model_version")
    report_labels = report.get("labels")
    if (
        not isinstance(model_version, str)
        or not model_version.strip()
        or not isinstance(report_labels, list)
        or any(not isinstance(label, str) or not label.strip() for label in report_labels)
        or len(set(report_labels)) != len(report_labels)
    ):
        raise ValueError("invalid bundle report configuration")
    if tuple(report_labels) != LABELS:
        raise ValueError("bundle report labels do not match official labels")

    # The operator selects this bundle directory; request bodies never control
    # a model path.  Joblib remains intentionally opt-in at this boundary.
    with _open_regular_file(model_path, description="active bundle model artifact") as model_file:
        try:
            scorer = load_text_scorer(model_file, trusted=True)
        except ValueError as error:
            raise ValueError("invalid active bundle model artifact") from error
    if scorer.labels != LABELS:
        raise ValueError("bundle model labels do not match official labels")
    if tuple(report_labels) != scorer.labels:
        raise ValueError("bundle report labels do not match the model")
    report_input_mode = report.get("model_input_mode")
    if report_input_mode is not None:
        if report_input_mode not in {"legacy", "metadata_v1"} or report_input_mode != scorer.input_mode:
            raise ValueError("bundle report input mode does not match the model")
    report_score_mode = report.get("score_mode")
    if report_score_mode is not None:
        if report_score_mode not in {"probability", "softmax"} or report_score_mode != scorer.score_mode:
            raise ValueError("bundle report score mode does not match the model")
    return _Runtime(scorer=scorer, model_type=MODEL_TYPE, model_version=model_version)


def create_app(
    bundle_root: Path,
    *,
    audio_config: ServiceAudioConfig | None = _DEFAULT_SERVICE_AUDIO_CONFIG,
    semantic_config: SemanticReviewConfig | None = None,
    rubric_path: Path | None = None,
    trace_path: Path | None = None,
) -> FastAPI:
    """Create a ready FastAPI service from one fully validated model bundle."""
    if audio_config is not None and not isinstance(audio_config, ServiceAudioConfig):
        raise ValueError("audio_config must be a ServiceAudioConfig or None")
    if semantic_config is not None and not isinstance(semantic_config, SemanticReviewConfig):
        raise ValueError("semantic_config must be a SemanticReviewConfig or None")
    if rubric_path is not None and not isinstance(rubric_path, Path):
        raise ValueError("rubric_path must be a Path or None")
    if trace_path is not None and not isinstance(trace_path, Path):
        raise ValueError("trace_path must be a Path or None")
    rubric = load_rubric(rubric_path or DEFAULT_RUBRIC_PATH)
    if trace_path is not None and trace_path.exists() and not trace_path.is_file():
        raise ValueError("trace_path must be a regular file path")
    runtime = _load_runtime(Path(bundle_root))
    app = FastAPI(title="Competition Emotion Service", version=runtime.model_version)
    app.add_middleware(_RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
    app.state.runtime = runtime
    app.state.audio_capacity = (
        _AudioCapacityController(audio_config.max_concurrent_audio) if audio_config is not None else None
    )
    app.state.semantic_config = semantic_config
    app.state.rubric = rubric
    app.state.trace_path = trace_path

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"code": 400, "message": "invalid request"})

    @app.exception_handler(StarletteHTTPException)
    async def protocol_http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        message = {
            404: "not found",
            405: "method not allowed",
        }.get(error.status_code, "request failed")
        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.status_code, "message": message},
        )

    @app.exception_handler(Exception)
    async def internal_service_error(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "internal service error"},
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(
            ready=True,
            model_type=runtime.model_type,
            model_version=runtime.model_version,
            label_count=len(runtime.scorer.labels),
        )

    @app.post("/api/v1/emotion/recognize", response_model=RecognizeResponse)
    async def recognize(request: RecognizeRequest, http_request: Request) -> RecognizeResponse:
        started_at = http_request.scope[_REQUEST_STARTED_AT_SCOPE_KEY]
        lyric_text = compose_lyrics(
            request.text_lyric, request.lrc_lyric, request.lrc_translation
        )
        lyric_used = bool(lyric_text)
        song = Song(
            song_id=request.song_id,
            labels=frozenset(),
            name=request.song_name,
            artists=request.artists or "",
            # The official request has no separate genre field.  Keep the
            # established auxiliary-metadata channel populated with the
            # optional album value so the trained artifact remains usable.
            genre=request.album_name or "",
            text=lyric_text,
            audio_url=request.audio_url,
            album_name=request.album_name or "",
        )
        scores = await asyncio.to_thread(
            runtime.scorer.score, runtime.scorer.compose_text(song)
        )
        scores = runtime.scorer.apply_song_overrides(song, scores)
        if not isinstance(scores, dict) or set(scores) != set(runtime.scorer.labels):
            raise RuntimeError("model returned invalid scores")
        ranked: list[tuple[str, float]] = []
        for label in runtime.scorer.labels:
            value = scores.get(label)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise RuntimeError("model returned invalid scores")
            confidence = float(value)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise RuntimeError("model returned invalid scores")
            ranked.append((label, confidence))
        if len(ranked) < 2:
            raise RuntimeError("model returned insufficient labels")
        ranked.sort(key=lambda item: item[1], reverse=True)
        candidate_labels: tuple[str, ...] = tuple(label for label, _ in ranked[:2])
        reviewed = None
        review_evidence: str | None = None
        if semantic_config is not None and ranked[0][1] - ranked[1][1] < semantic_config.min_score_gap:
            candidates = [label for label, _ in ranked[: semantic_config.candidate_count]]
            candidate_labels = tuple(candidates)
            try:
                reviewed = await asyncio.wait_for(
                    asyncio.to_thread(
                        review_candidates,
                        song,
                        lyric_text,
                        candidates,
                        semantic_config,
                        rubric,
                    ),
                    timeout=float(semantic_config.timeout_seconds) + 0.25,
                )
            except (ValueError, RuntimeError, OSError, TimeoutError, httpx.HTTPError, asyncio.TimeoutError):
                reviewed = None
            if reviewed is not None:
                other = [item for item in ranked if item[0] != reviewed.label]
                if other:
                    other.sort(key=lambda item: item[1], reverse=True)
                    selected_confidence = float(reviewed.confidence)
                    ranked = [
                        (reviewed.label, selected_confidence),
                        (other[0][0], min(other[0][1], selected_confidence)),
                    ]
                review_evidence = reviewed.evidence
        (top_emotion, top_confidence), (second_emotion, second_confidence) = ranked[:2]
        audio_result = RequestAudioResult("unavailable")
        if audio_config is not None:
            request_started_at = http_request.scope[_REQUEST_MONOTONIC_STARTED_AT_SCOPE_KEY]
            request_deadline = request_started_at + float(audio_config.request_deadline_seconds)
            capacity = app.state.audio_capacity
            if capacity.try_acquire():
                try:
                    audio_task = asyncio.create_task(
                        asyncio.to_thread(
                            acquire_request_audio,
                            request.audio_url,
                            audio_config.request_config,
                            deadline_monotonic=request_deadline,
                        )
                    )
                    audio_task.add_done_callback(_release_audio_capacity(capacity))
                except BaseException:
                    capacity.release()
                    raise
                try:
                    audio_result = await asyncio.shield(audio_task)
                except (ValueError, RuntimeError, OSError, TimeoutError, httpx.HTTPError):
                    audio_result = RequestAudioResult("unavailable")
        metadata_fields: tuple[str, ...] = ()
        metadata_used = runtime.scorer.input_mode == "metadata_v1"
        if metadata_used:
            fields = ["歌曲名称"]
            if request.artists and request.artists.strip():
                fields.append("艺人")
            if request.album_name and request.album_name.strip():
                fields.append("专辑/风格元数据")
            metadata_fields = tuple(fields)
        evidence = build_evidence(
            lyric_used=lyric_used,
            title_used=not lyric_used,
            audio_state=audio_result.state if audio_config is not None else None,
            metadata_used=metadata_used,
            metadata_fields=metadata_fields,
        )
        if review_evidence is not None:
            review_parts = [review_evidence]
            if reviewed is not None and reviewed.quotes:
                review_parts.append("歌词引文：“" + "；".join(reviewed.quotes) + "”")
            if reviewed is not None and reviewed.rule_ids:
                review_parts.append("Rubric规则：" + "、".join(reviewed.rule_ids))
            evidence = f"{evidence} 语义复核选择候选“{top_emotion}”：" + " ".join(review_parts)
            evidence = evidence[:500]
        if trace_path is not None:
            trace_record = TraceRecord(
                trace_id=uuid4().hex,
                song_id=request.song_id,
                model_version=runtime.model_version,
                rubric_version=rubric.version,
                top_label=top_emotion,
                top_confidence=float(top_confidence),
                second_label=second_emotion,
                second_confidence=float(second_confidence),
                candidates=candidate_labels,
                evidence=evidence,
                reviewer_used=reviewed is not None,
                reviewer_label=reviewed.label if reviewed is not None else None,
                reviewer_confidence=reviewed.confidence if reviewed is not None else None,
                reviewer_evidence=reviewed.evidence if reviewed is not None else "",
                reviewer_quotes=reviewed.quotes if reviewed is not None else (),
                reviewer_rule_ids=reviewed.rule_ids if reviewed is not None else (),
                latency_ms=max(0, int((perf_counter() - started_at) * 1000)),
            )
            asyncio.create_task(asyncio.to_thread(_append_trace_safely, trace_path, trace_record))
        return RecognizeResponse(
            code=200,
            data=RecognizeData(
                top_emotion=top_emotion,
                top_confidence=round(top_confidence, 4),
                second_emotion=second_emotion,
                second_confidence=round(second_confidence, 4),
                evidence=evidence,
                cost_ms=max(0, int((perf_counter() - started_at) * 1000)),
            ),
            error="",
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a trusted emotion-model bundle")
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--audio-proxy-url")
    parser.add_argument("--audio-allowed-host", action="append", default=None)
    parser.add_argument("--audio-temp-root", type=Path)
    parser.add_argument("--audio-budget-seconds", default=20.0, type=float)
    parser.add_argument("--max-concurrent-audio", default=4, type=int)
    parser.add_argument("--semantic-reranker-url")
    parser.add_argument("--semantic-reranker-timeout-seconds", type=float)
    parser.add_argument("--semantic-reranker-min-gap", type=float)
    parser.add_argument("--semantic-reranker-candidate-count", type=int)
    parser.add_argument("--rubric-path", type=Path)
    parser.add_argument("--trace-path", type=Path)
    arguments = parser.parse_args(argv)
    try:
        host_address = ipaddress.ip_address(arguments.host)
    except ValueError:
        parser.error("--host must be a non-loopback IP literal")
    if host_address.is_loopback:
        parser.error("--host must be a non-loopback IP literal")
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if arguments.audio_proxy_url is not None and not arguments.audio_allowed_host:
        parser.error("--audio-proxy-url requires at least one --audio-allowed-host")
    if arguments.audio_proxy_url is None and arguments.audio_allowed_host:
        parser.error("--audio-allowed-host requires --audio-proxy-url")
    try:
        request_audio_config = RequestAudioConfig(
            temp_root=arguments.audio_temp_root,
            total_seconds=arguments.audio_budget_seconds,
            proxy_url=arguments.audio_proxy_url,
            allowed_hosts=arguments.audio_allowed_host,
        )
    except ValueError as error:
        parser.error(str(error))
    semantic_url = arguments.semantic_reranker_url or os.environ.get(
        "EMOTION_SEMANTIC_RERANKER_URL", ""
    ).strip()
    try:
        semantic_timeout = (
            arguments.semantic_reranker_timeout_seconds
            if arguments.semantic_reranker_timeout_seconds is not None
            else float(os.environ.get("SEMANTIC_RERANKER_TIMEOUT_SECONDS", "3"))
        )
        semantic_gap = (
            arguments.semantic_reranker_min_gap
            if arguments.semantic_reranker_min_gap is not None
            else float(os.environ.get("SEMANTIC_RERANKER_MIN_GAP", "0.10"))
        )
        semantic_candidate_count = (
            arguments.semantic_reranker_candidate_count
            if arguments.semantic_reranker_candidate_count is not None
            else int(os.environ.get("SEMANTIC_RERANKER_CANDIDATE_COUNT", "10"))
        )
    except ValueError:
        parser.error("semantic reviewer environment values must be numeric")
    semantic_config = None
    if semantic_url:
        try:
            semantic_config = SemanticReviewConfig(
                semantic_url,
                timeout_seconds=semantic_timeout,
                min_score_gap=semantic_gap,
                candidate_count=semantic_candidate_count,
            )
        except ValueError as error:
            parser.error(str(error))
    rubric_path = arguments.rubric_path
    if rubric_path is None:
        configured_rubric_path = os.environ.get("EMOTION_RUBRIC_PATH", "").strip()
        rubric_path = Path(configured_rubric_path) if configured_rubric_path else None
    trace_path = arguments.trace_path
    if trace_path is None:
        configured_trace_path = os.environ.get("EMOTION_TRACE_PATH", "").strip()
        trace_path = Path(configured_trace_path) if configured_trace_path else None
    uvicorn.run(
        create_app(
            arguments.bundle_root,
            audio_config=ServiceAudioConfig(
                request_config=request_audio_config,
                max_concurrent_audio=arguments.max_concurrent_audio,
            ),
            semantic_config=semantic_config,
            rubric_path=rubric_path,
            trace_path=trace_path,
        ),
        host=arguments.host,
        port=arguments.port,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
