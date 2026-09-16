"""Validation for model supplied evidence and lyric quotations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .rubric import Rubric


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ValidatedEvidence:
    evidence: str
    quotes: tuple[str, ...]
    rule_ids: tuple[str, ...]


def _normalize(value: str) -> str:
    return _WHITESPACE.sub("", value).strip()


def validate_evidence(
    lyric_text: str,
    evidence: str,
    quotes: Sequence[str] | None = None,
    rule_ids: Sequence[str] | None = None,
    *,
    rubric: Rubric | None = None,
    label: str | None = None,
    require_quote: bool = True,
) -> ValidatedEvidence:
    """Validate bounded evidence without trusting an endpoint's citations.

    Quotes are accepted only when their normalized text occurs in the supplied
    lyrics.  Rule IDs are checked against the versioned rubric when present.
    """
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence.strip()) > 450:
        raise ValueError("evidence must be a non-empty string of at most 450 characters")
    if not isinstance(lyric_text, str):
        raise ValueError("lyric_text must be a string")
    raw_quotes = () if quotes is None else tuple(quotes)
    if len(raw_quotes) > 3:
        raise ValueError("evidence may contain at most three quotes")
    cleaned_quotes: list[str] = []
    normalized_lyrics = _normalize(lyric_text)
    for quote in raw_quotes:
        if not isinstance(quote, str) or not quote.strip() or len(quote.strip()) > 120:
            raise ValueError("each evidence quote must be a non-empty string of at most 120 characters")
        cleaned = quote.strip()
        if _normalize(cleaned) not in normalized_lyrics:
            raise ValueError("evidence quote does not occur in supplied lyrics")
        cleaned_quotes.append(cleaned)
    if normalized_lyrics and require_quote and not cleaned_quotes:
        raise ValueError("lyric evidence requires at least one verified quote")
    raw_rule_ids = () if rule_ids is None else tuple(rule_ids)
    if len(raw_rule_ids) > 8 or any(not isinstance(rule_id, str) or not rule_id.strip() for rule_id in raw_rule_ids):
        raise ValueError("rule_ids must be a bounded string list")
    cleaned_rule_ids = tuple(dict.fromkeys(rule_id.strip() for rule_id in raw_rule_ids))
    if rubric is not None:
        unknown = [rule_id for rule_id in cleaned_rule_ids if rule_id not in rubric.rule_ids]
        if unknown:
            raise ValueError("unknown rubric rule ID: " + ", ".join(unknown))
        if label is not None:
            wrong_label = [rule_id for rule_id in cleaned_rule_ids if rule_id not in rubric.rule_ids_for(label)]
            if wrong_label:
                raise ValueError("rubric rule ID does not belong to selected label: " + ", ".join(wrong_label))
    return ValidatedEvidence(evidence.strip(), tuple(cleaned_quotes), cleaned_rule_ids)
