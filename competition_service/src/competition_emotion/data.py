from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

import pandas as pd

from .constants import LABEL_SET
from .types import Song


REQUIRED_COLUMNS = (
    "歌曲id",
    "情绪类型",
    "歌曲名称",
    "一级曲风标签",
    "演唱艺人",
    "文本歌词",
    "音频下载地址",
    "lrc歌词（滚词）",
    "翻译歌词",
)
_TIMESTAMP = re.compile(r"\[\d{1,3}:\d{2}(?:\.\d+)?\]")
_WHITESPACE = re.compile(r"\s+")


def _clean_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_lyric(value: object) -> str:
    """Remove LRC timestamps and normalize lyric whitespace."""
    return _WHITESPACE.sub(" ", _TIMESTAMP.sub("", _clean_string(value))).strip()


def _song_sort_key(song_id: str) -> tuple[int, Decimal | str, str]:
    try:
        return (0, Decimal(song_id), song_id)
    except InvalidOperation:
        return (1, song_id, song_id)


def _song_text(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def load_official_songs(path: Path) -> list[Song]:
    """Load official rows, combining labels that belong to the same song."""
    frame = pd.read_excel(path, dtype={"歌曲id": "string"})
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")

    songs_by_id: dict[str, Song] = {}
    for _, row in frame.iterrows():
        song_id = _clean_string(row["歌曲id"])
        label = _clean_string(row["情绪类型"])
        if not song_id or label not in LABEL_SET:
            continue

        existing = songs_by_id.get(song_id)
        if existing is not None:
            songs_by_id[song_id] = Song(
                song_id=existing.song_id,
                labels=existing.labels | frozenset({label}),
                name=existing.name,
                artists=existing.artists,
                genre=existing.genre,
                text=existing.text,
                audio_url=existing.audio_url,
            )
            continue

        name = _clean_string(row["歌曲名称"])
        artists = _clean_string(row["演唱艺人"])
        lyric = clean_lyric(row["文本歌词"])
        lrc = clean_lyric(row["lrc歌词（滚词）"])
        translation = clean_lyric(row["翻译歌词"])
        songs_by_id[song_id] = Song(
            song_id=song_id,
            labels=frozenset({label}),
            name=name,
            artists=artists,
            genre=_clean_string(row["一级曲风标签"]),
            text=_song_text((name, artists, lyric, lrc, translation)),
            audio_url=_clean_string(row["音频下载地址"]),
        )

    return sorted(songs_by_id.values(), key=lambda song: _song_sort_key(song.song_id))
