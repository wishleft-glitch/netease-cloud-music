from __future__ import annotations

import re

import pandas as pd


_TIMESTAMP = re.compile(r"\[\d{1,3}:\d{2}(?:\.\d+)?\]")
_WHITESPACE = re.compile(r"\s+")


def _clean_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_lyric(value: object) -> str:
    """Remove LRC timestamps and normalize lyric whitespace."""
    return _WHITESPACE.sub(" ", _TIMESTAMP.sub("", _clean_string(value))).strip()


def compose_lyrics(
    text_lyric: object, lrc_lyric: object, lrc_translation: object
) -> str:
    """Combine the official lyric sources in their documented order."""
    parts: list[str] = []
    for value in (text_lyric, lrc_lyric, lrc_translation):
        cleaned = clean_lyric(value)
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return "\n".join(parts)
