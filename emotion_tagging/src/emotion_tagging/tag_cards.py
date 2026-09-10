"""Load configurable emotion-tag rules for the pilot."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .models import TagCard


_REQUIRED_CARD_FIELDS = frozenset({"name", "audio", "lyrics", "requires_human_review"})


def load_tag_cards(config_path: str | Path) -> list[TagCard]:
    """Read ordered tag cards from a JSON pilot configuration file."""
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Tag card configuration must be a JSON list")

    return [_parse_tag_card(item, index) for index, item in enumerate(payload)]


def _parse_tag_card(item: object, index: int) -> TagCard:
    if not isinstance(item, Mapping):
        raise ValueError(f"Tag card {index} must be an object")

    missing = _REQUIRED_CARD_FIELDS.difference(item)
    if missing:
        raise ValueError(f"Tag card {index} is missing required fields: {', '.join(sorted(missing))}")

    name = item["name"]
    audio = item["audio"]
    lyrics = item["lyrics"]
    requires_human_review = item["requires_human_review"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"Tag card {index} name must be a non-empty string")
    if not isinstance(audio, Mapping) or not isinstance(lyrics, Mapping):
        raise ValueError(f"Tag card {index} audio and lyrics must be objects")
    if not isinstance(requires_human_review, bool):
        raise ValueError(f"Tag card {index} requires_human_review must be a boolean")

    return TagCard(
        name=name,
        min_valence=_optional_number(audio, "min_valence", index),
        max_valence=_optional_number(audio, "max_valence", index),
        min_arousal=_optional_number(audio, "min_arousal", index),
        max_arousal=_optional_number(audio, "max_arousal", index),
        lyric_includes=_lyric_terms(lyrics, "includes", index),
        lyric_excludes=_lyric_terms(lyrics, "excludes", index),
        requires_human_review=requires_human_review,
    )


def _optional_number(audio: Mapping[str, object], field: str, index: int) -> float | None:
    value = audio.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Tag card {index} audio.{field} must be a finite number between 0.0 and 1.0")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"Tag card {index} audio.{field} must be finite")
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"Tag card {index} audio.{field} must be between 0.0 and 1.0")
    return numeric_value


def _lyric_terms(lyrics: Mapping[str, object], field: str, index: int) -> tuple[str, ...]:
    terms = lyrics.get(field, [])
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise ValueError(f"Tag card {index} lyrics.{field} must be a list of strings")
    if not all(isinstance(term, str) for term in terms):
        raise ValueError(f"Tag card {index} lyrics.{field} must be a list of strings")
    return tuple(term.casefold() for term in terms)
