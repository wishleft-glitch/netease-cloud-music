"""Versioned, validated rubric context for candidate-limited semantic review."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .constants import LABELS


_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUBRIC_PATH = _SERVICE_ROOT / "rubric" / "emotion_rubric.json"


@dataclass(frozen=True)
class RubricEntry:
    label: str
    definition: str
    positive_cues: tuple[str, ...]
    counterexamples: tuple[str, ...]
    rule_ids: tuple[str, ...]

    def as_context(self) -> dict[str, object]:
        return {
            "label": self.label,
            "definition": self.definition,
            "positive_cues": list(self.positive_cues),
            "counterexamples": list(self.counterexamples),
            "rule_ids": list(self.rule_ids),
        }


@dataclass(frozen=True)
class Rubric:
    version: str
    entries: tuple[RubricEntry, ...]
    mode: str = "soft_context"

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in self.entries)

    @property
    def rule_ids(self) -> frozenset[str]:
        return frozenset(rule_id for entry in self.entries for rule_id in entry.rule_ids)

    def context(self, candidates: Sequence[str] | None = None) -> list[dict[str, object]]:
        requested = self.labels if candidates is None else tuple(dict.fromkeys(str(item).strip() for item in candidates))
        by_label = {entry.label: entry for entry in self.entries}
        unknown = [label for label in requested if label not in by_label]
        if unknown:
            raise ValueError("unknown rubric label: " + ", ".join(unknown))
        return [by_label[label].as_context() for label in requested]

    def rule_ids_for(self, label: str) -> frozenset[str]:
        for entry in self.entries:
            if entry.label == label:
                return frozenset(entry.rule_ids)
        raise ValueError("unknown rubric label: " + str(label))


def _string_tuple(value: Any, field: str, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"rubric {field} for {label} must be a non-empty string list")
    return tuple(item.strip() for item in value)


def load_rubric(path: Path = DEFAULT_RUBRIC_PATH) -> Rubric:
    rubric_path = Path(path)
    if not rubric_path.is_file():
        raise ValueError("rubric path must be a regular file")
    try:
        payload = json.loads(rubric_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid rubric JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported rubric schema")
    labels = payload.get("labels")
    if labels != list(LABELS):
        raise ValueError("rubric labels do not match official labels")
    version = payload.get("rubric_version")
    mode = payload.get("mode")
    entries_payload = payload.get("entries")
    if not isinstance(version, str) or not version.strip() or mode != "soft_context" or not isinstance(entries_payload, list):
        raise ValueError("invalid rubric configuration")
    entries: list[RubricEntry] = []
    seen_labels: set[str] = set()
    seen_rules: set[str] = set()
    for raw in entries_payload:
        if not isinstance(raw, dict):
            raise ValueError("invalid rubric entry")
        label = raw.get("label")
        definition = raw.get("definition")
        if not isinstance(label, str) or label not in LABELS or label in seen_labels:
            raise ValueError("rubric entries must cover each official label once")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"rubric definition for {label} must be non-empty")
        positive_cues = _string_tuple(raw.get("positive_cues"), "positive_cues", label)
        counterexamples = _string_tuple(raw.get("counterexamples"), "counterexamples", label)
        rule_ids = _string_tuple(raw.get("rule_ids"), "rule IDs", label)
        if seen_rules & set(rule_ids):
            raise ValueError("rubric rule IDs must be globally unique")
        seen_labels.add(label)
        seen_rules.update(rule_ids)
        entries.append(RubricEntry(label, definition.strip(), positive_cues, counterexamples, rule_ids))
    if tuple(entry.label for entry in entries) != LABELS:
        raise ValueError("rubric entries do not match official labels")
    return Rubric(version.strip(), tuple(entries), mode)
