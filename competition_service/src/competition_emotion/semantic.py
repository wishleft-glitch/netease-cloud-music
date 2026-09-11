"""Optional, candidate-limited semantic review for ambiguous requests."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx

from .evidence_validator import validate_evidence
from .rubric import DEFAULT_RUBRIC_PATH, Rubric, load_rubric
from .types import Song


@dataclass(frozen=True)
class SemanticReviewConfig:
    """Configuration for an operator-managed internal review endpoint."""

    url: str
    timeout_seconds: float = 3.0
    min_score_gap: float = 0.10
    # Ten candidates retain 97.67% strict-singleton coverage on the fixed
    # official holdout while keeping the reviewer prompt bounded.
    candidate_count: int = 10
    require_verified_evidence: bool = True

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("semantic review URL must be an http or https URL")
        if parsed.username or parsed.password:
            raise ValueError("semantic review URL must not contain credentials")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            raise ValueError("semantic review URL must include a host")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None:
            if address.is_global:
                raise ValueError("semantic review URL must target an internal host")
        elif "." in hostname and not hostname.endswith((".internal", ".local", ".corp")):
            raise ValueError("semantic review URL must target an internal host")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or not 0.1 <= float(self.timeout_seconds) <= 10.0
        ):
            raise ValueError("semantic review timeout must be between 0.1 and 10 seconds")
        if (
            isinstance(self.min_score_gap, bool)
            or not isinstance(self.min_score_gap, (int, float))
            or not math.isfinite(float(self.min_score_gap))
            or not 0.0 <= float(self.min_score_gap) <= 1.0
        ):
            raise ValueError("semantic review min_score_gap must be between 0 and 1")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or not 2 <= self.candidate_count <= 15
        ):
            raise ValueError("semantic review candidate_count must be between 2 and 15")
        if not isinstance(self.require_verified_evidence, bool):
            raise ValueError("semantic review require_verified_evidence must be a boolean")


@dataclass(frozen=True)
class SemanticReviewResult:
    label: str
    confidence: float
    evidence: str
    quotes: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()


def review_candidates(
    song: Song,
    lyric_text: str,
    candidates: Sequence[str],
    config: SemanticReviewConfig,
    rubric: Rubric | None = None,
) -> SemanticReviewResult:
    """Ask the internal endpoint to choose only from the supplied candidates."""
    rubric = rubric or load_rubric(DEFAULT_RUBRIC_PATH)
    candidate_list = tuple(dict.fromkeys(str(candidate).strip() for candidate in candidates))
    if not candidate_list:
        raise ValueError("semantic review requires candidates")
    payload = {
        "song_id": song.song_id,
        "song_name": song.name,
        "artists": song.artists,
        "album_name": song.album_name,
        "lyrics": lyric_text,
        "candidates": list(candidate_list),
        "rubric": rubric.context(candidate_list) if rubric is not None else [],
        "instruction": "选择一个最符合歌曲整体情绪的候选标签，只返回候选标签、置信度、可核验歌词引文和Rubric规则ID。",
    }
    with httpx.Client(timeout=float(config.timeout_seconds), follow_redirects=False) as client:
        response = client.post(config.url, json=payload)
        response.raise_for_status()
        result: Any = response.json()
    if not isinstance(result, dict):
        raise ValueError("semantic review response must be an object")
    label = result.get("label")
    confidence = result.get("confidence")
    evidence = result.get("evidence")
    if (
        not isinstance(label, str)
        or label not in candidate_list
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
        or not isinstance(evidence, str)
        or not evidence.strip()
        or len(evidence) > 450
    ):
        raise ValueError("semantic review response is invalid")
    quotes = result.get("quotes", [])
    rule_ids = result.get("rule_ids", [])
    if not isinstance(quotes, list) or not isinstance(rule_ids, list):
        raise ValueError("semantic review response is invalid")
    try:
        validated = validate_evidence(
            lyric_text,
            evidence,
            quotes,
            rule_ids,
            rubric=rubric,
            label=label,
            require_quote=config.require_verified_evidence,
        )
    except ValueError as error:
        raise ValueError("semantic review response is invalid") from error
    return SemanticReviewResult(
        label,
        float(confidence),
        validated.evidence,
        validated.quotes,
        validated.rule_ids,
    )
