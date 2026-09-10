"""Production HTTP interface for a trusted lyrics-model bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
import uvicorn

from .models import TextScorer, load_text_scorer
from .train import BUNDLE_POINTER_SCHEMA_VERSION, MODEL_TYPE, REPORT_SCHEMA_VERSION
from .types import Song


_MAX_POINTER_BYTES = 64 * 1024
_MAX_REPORT_BYTES = 1024 * 1024


class RecognizeRequest(BaseModel):
    """The bounded, text-only inference payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    song_id: StrictStr | None = Field(default=None, max_length=300)
    title: StrictStr = Field(max_length=300)
    artists: StrictStr | None = Field(default=None, max_length=500)
    genre: StrictStr | None = Field(default=None, max_length=200)
    lyrics: StrictStr | None = Field(default=None, max_length=100_000)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be blank")
        return value


class RecognizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    song_id: str | None
    emotion_label: str
    confidence: float
    model_type: str
    model_version: str


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


def _read_json_file(path: Path, *, max_bytes: int, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file")
    try:
        if path.stat().st_size > max_bytes:
            raise ValueError(f"{description} is too large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description}")
    return value


def _safe_bundle_dir(bundle_root: Path) -> Path:
    """Resolve the operator-supplied pointer without accepting path escapes."""
    try:
        root = bundle_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("bundle root does not exist") from error
    if not root.is_dir() or root.is_symlink():
        raise ValueError("bundle root must be a regular directory")

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
    if versions_dir.is_symlink() or candidate.is_symlink() or not candidate.is_dir():
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
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("active bundle has no regular model artifact")
    report = _read_json_file(
        bundle_dir / "report.json", max_bytes=_MAX_REPORT_BYTES, description="bundle report"
    )
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported bundle report schema")
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

    # The operator selects this bundle directory; request bodies never control
    # a model path.  Joblib remains intentionally opt-in at this boundary.
    scorer = load_text_scorer(model_path, trusted=True)
    if tuple(report_labels) != scorer.labels:
        raise ValueError("bundle report labels do not match the model")
    return _Runtime(scorer=scorer, model_type=MODEL_TYPE, model_version=model_version)


def create_app(bundle_root: Path) -> FastAPI:
    """Create a ready FastAPI service from one fully validated model bundle."""
    runtime = _load_runtime(Path(bundle_root))
    app = FastAPI(title="Competition Emotion Service", version=runtime.model_version)
    app.state.runtime = runtime

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            ready=True,
            model_type=runtime.model_type,
            model_version=runtime.model_version,
            label_count=len(runtime.scorer.labels),
        )

    @app.post("/api/v1/emotion/recognize", response_model=RecognizeResponse)
    def recognize(request: RecognizeRequest) -> RecognizeResponse:
        song = Song(
            song_id=request.song_id or "",
            labels=frozenset(),
            name=request.title,
            artists=request.artists or "",
            genre=request.genre or "",
            text=request.lyrics or "",
            audio_url="",
        )
        scores = runtime.scorer.score(song.text.strip() or song.name)
        ranked = [(label, scores.get(label)) for label in runtime.scorer.labels]
        if any(not isinstance(value, (int, float)) for _, value in ranked):
            raise RuntimeError("model returned incomplete scores")
        emotion_label, confidence = max(ranked, key=lambda item: item[1])
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError("model returned an invalid confidence")
        return RecognizeResponse(
            song_id=request.song_id,
            emotion_label=emotion_label,
            confidence=confidence,
            model_type=runtime.model_type,
            model_version=runtime.model_version,
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a trusted emotion-model bundle")
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    uvicorn.run(create_app(arguments.bundle_root), host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
