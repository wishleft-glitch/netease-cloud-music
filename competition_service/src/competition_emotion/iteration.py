"""Auditable trace, hard-case mining, and guarded rubric iteration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
import math
import os
import shutil
from tempfile import NamedTemporaryFile
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .rubric import load_rubric


TRACE_SCHEMA_VERSION = 1
PATCH_SCHEMA_VERSION = 1
_TRACE_APPEND_LOCK = Lock()


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    song_id: str
    model_version: str
    rubric_version: str
    top_label: str
    top_confidence: float
    second_label: str
    second_confidence: float
    candidates: tuple[str, ...]
    evidence: str
    reviewer_used: bool = False
    reviewer_label: str | None = None
    reviewer_confidence: float | None = None
    reviewer_evidence: str = ""
    reviewer_quotes: tuple[str, ...] = ()
    reviewer_rule_ids: tuple[str, ...] = ()
    latency_ms: int | None = None
    created_at_utc: str = ""

    @property
    def margin(self) -> float:
        return float(self.top_confidence - self.second_confidence)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TRACE_SCHEMA_VERSION
        payload["candidates"] = list(self.candidates)
        payload["reviewer_quotes"] = list(self.reviewer_quotes)
        payload["reviewer_rule_ids"] = list(self.reviewer_rule_ids)
        if not payload["created_at_utc"]:
            payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceRecord":
        if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema")
        string_fields = ("trace_id", "song_id", "model_version", "rubric_version", "top_label", "second_label", "evidence")
        if any(not isinstance(payload.get(key), str) or not str(payload.get(key)).strip() for key in string_fields):
            raise ValueError("invalid trace record")
        candidates = payload.get("candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates or any(not isinstance(item, str) or not item.strip() for item in candidates):
            raise ValueError("invalid trace record")
        if not isinstance(payload.get("reviewer_used", False), bool):
            raise ValueError("invalid trace record")
        try:
            top_confidence = float(payload["top_confidence"])
            second_confidence = float(payload["second_confidence"])
            if not math.isfinite(top_confidence) or not math.isfinite(second_confidence):
                raise ValueError("non-finite confidence")
            reviewer_confidence = float(payload["reviewer_confidence"]) if payload.get("reviewer_confidence") is not None else None
            if reviewer_confidence is not None and not math.isfinite(reviewer_confidence):
                raise ValueError("non-finite reviewer confidence")
            return cls(
                trace_id=str(payload["trace_id"]),
                song_id=str(payload["song_id"]),
                model_version=str(payload["model_version"]),
                rubric_version=str(payload["rubric_version"]),
                top_label=str(payload["top_label"]),
                top_confidence=top_confidence,
                second_label=str(payload["second_label"]),
                second_confidence=second_confidence,
                candidates=tuple(str(item).strip() for item in candidates),
                evidence=str(payload["evidence"]),
                reviewer_used=bool(payload.get("reviewer_used", False)),
                reviewer_label=(str(payload["reviewer_label"]) if payload.get("reviewer_label") is not None else None),
                reviewer_confidence=reviewer_confidence,
                reviewer_evidence=str(payload.get("reviewer_evidence", "")),
                reviewer_quotes=tuple(str(item) for item in payload.get("reviewer_quotes", [])),
                reviewer_rule_ids=tuple(str(item) for item in payload.get("reviewer_rule_ids", [])),
                latency_ms=(int(payload["latency_ms"]) if payload.get("latency_ms") is not None else None),
                created_at_utc=str(payload.get("created_at_utc", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid trace record") from error


def append_trace(path: Path, record: TraceRecord) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    with _TRACE_APPEND_LOCK:
        with destination.open("a", encoding="utf-8", newline="") as output:
            output.write(line)
            output.flush()


def read_traces(path: Path) -> list[TraceRecord]:
    source = Path(path)
    if not source.is_file():
        return []
    records: list[TraceRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("record must be an object")
            records.append(TraceRecord.from_dict(payload))
        except (json.JSONDecodeError, UnicodeError, ValueError) as error:
            raise ValueError(f"invalid trace at line {line_number}") from error
    return records


def mine_hard_cases(
    records: Iterable[TraceRecord], *, margin_threshold: float = 0.10, limit: int = 100
) -> list[dict[str, Any]]:
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError("margin_threshold must be between 0 and 1")
    if not 1 <= limit <= 10_000:
        raise ValueError("limit must be between 1 and 10000")
    mined: list[dict[str, Any]] = []
    for record in records:
        reasons: list[str] = []
        if record.margin < margin_threshold:
            reasons.append("low_margin")
        if record.reviewer_used and record.reviewer_label and record.reviewer_label != record.top_label:
            reasons.append("reviewer_disagreement")
        if not reasons:
            continue
        mined.append({
            "trace_id": record.trace_id,
            "song_id": record.song_id,
            "top_label": record.top_label,
            "second_label": record.second_label,
            "margin": round(record.margin, 6),
            "candidates": list(record.candidates),
            "reasons": reasons,
            "reviewer_label": record.reviewer_label,
            "reviewer_rule_ids": list(record.reviewer_rule_ids),
        })
    mined.sort(key=lambda item: (float(item["margin"]), str(item["song_id"])))
    return mined[:limit]


def build_rubric_patch_proposal(
    hard_cases: Sequence[dict[str, Any]], *, base_rubric_version: str, max_changes: int = 50
) -> dict[str, Any]:
    if not isinstance(base_rubric_version, str) or not base_rubric_version.strip():
        raise ValueError("base_rubric_version must be non-empty")
    changes: list[dict[str, Any]] = []
    for case in hard_cases[:max_changes]:
        label = str(case.get("reviewer_label") or case.get("top_label") or "").strip()
        if not label:
            continue
        changes.append({
            "label": label,
            "trigger_song_ids": [str(case.get("song_id", ""))],
            "reason": list(case.get("reasons", [])),
            "candidate_pair": [str(case.get("top_label", "")), str(case.get("second_label", ""))],
            "action": "人工复核后补充正例、反例或成对判别规则",
        })
    return {
        "schema_version": PATCH_SCHEMA_VERSION,
        "patch_id": uuid4().hex,
        "base_rubric_version": base_rubric_version.strip(),
        "status": "proposed",
        "requires_human_review": True,
        "changes": changes,
    }


def evaluate_patch_gate(
    baseline: dict[str, Any], candidate: dict[str, Any], *, max_top1_drop: float = 0.005, max_macro_drop: float = 0.005, max_label_recall_drop: float = 0.03
) -> dict[str, Any]:
    """Return a CI-style gate result; publishing remains a separate operator action."""
    reasons: list[str] = []
    for key, allowance, title in (
        ("strict_top1_accuracy", max_top1_drop, "strict_top1_accuracy"),
        ("macro_recall", max_macro_drop, "macro_recall"),
    ):
        try:
            delta = float(candidate[key]) - float(baseline[key])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"missing metric: {key}")
            continue
        if delta < -allowance:
            reasons.append(f"{title} regression {delta:.6f} exceeds {-allowance:.6f}")
    base_recall = baseline.get("per_label_recall", {})
    candidate_recall = candidate.get("per_label_recall", {})
    if not isinstance(base_recall, dict) or not isinstance(candidate_recall, dict) or set(base_recall) != set(candidate_recall):
        reasons.append("per_label_recall label set changed")
    else:
        for label in sorted(base_recall):
            try:
                delta = float(candidate_recall[label]) - float(base_recall[label])
            except (TypeError, ValueError):
                reasons.append(f"invalid per_label_recall: {label}")
                continue
            if delta < -max_label_recall_drop:
                reasons.append(f"{label} recall regression {delta:.6f} exceeds {-max_label_recall_drop:.6f}")
    return {"passed": not reasons, "reasons": reasons}


def publish_rubric_candidate(
    candidate_path: Path,
    destination_path: Path,
    *,
    gate_result: dict[str, Any],
    approved: bool,
) -> dict[str, str]:
    """Atomically publish a validated Rubric after an explicit human approval."""
    if not approved:
        raise ValueError("human approval is required before publishing a rubric")
    if not isinstance(gate_result, dict) or gate_result.get("passed") is not True:
        raise ValueError("rubric regression gate must pass before publishing")
    candidate = load_rubric(Path(candidate_path))
    destination = Path(destination_path)
    if destination.exists() and not destination.is_file():
        raise ValueError("rubric destination must be a regular file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with Path(candidate_path).open("rb") as source:
            shutil.copyfileobj(source, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"rubric_version": candidate.version, "destination": str(destination)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine hard cases and create an unpublished Rubric patch proposal")
    parser.add_argument("--trace-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-rubric-version", default="emotion-rubric-v2")
    parser.add_argument("--margin-threshold", default=0.10, type=float)
    parser.add_argument("--limit", default=100, type=int)
    arguments = parser.parse_args(argv)
    cases = mine_hard_cases(
        read_traces(arguments.trace_path),
        margin_threshold=arguments.margin_threshold,
        limit=arguments.limit,
    )
    proposal = build_rubric_patch_proposal(cases, base_rubric_version=arguments.base_rubric_version)
    output = {"hard_cases": cases, "proposal": proposal}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
