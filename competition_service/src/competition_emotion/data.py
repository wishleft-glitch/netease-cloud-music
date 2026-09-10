from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path
import stat
from tempfile import NamedTemporaryFile

import pandas as pd

from .constants import LABEL_SET
from .lyrics import clean_lyric, compose_lyrics
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
def _clean_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _song_sort_key(song_id: str) -> tuple[int, Decimal | str, str]:
    try:
        numeric_id = Decimal(song_id)
    except (InvalidOperation, ValueError):
        return (1, song_id, song_id)
    if not numeric_id.is_finite() or numeric_id != numeric_id.to_integral_value():
        return (1, song_id, song_id)
    return (0, numeric_id, song_id)


def _song_text(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


@contextmanager
def workbook_snapshot(path: Path) -> Iterator[tuple[Path, dict[str, object]]]:
    """Yield one immutable workbook snapshot with its reproducible provenance."""
    source_path = Path(path)
    digest = hashlib.sha256()
    snapshot_path: Path | None = None
    try:
        with source_path.open("rb") as workbook:
            if not stat.S_ISREG(os.fstat(workbook.fileno()).st_mode):
                raise ValueError("Official workbook must be a regular file")
            with NamedTemporaryFile(
                mode="wb", suffix=source_path.suffix, delete=False
            ) as snapshot:
                snapshot_path = Path(snapshot.name)
                for chunk in iter(lambda: workbook.read(1024 * 1024), b""):
                    digest.update(chunk)
                    snapshot.write(chunk)

        provenance = {
            "file_name": source_path.name,
            "sha256": digest.hexdigest(),
            "rows": len(pd.read_excel(snapshot_path)),
        }
        yield snapshot_path, provenance
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)


def workbook_provenance(path: Path) -> dict[str, object]:
    """Return the minimum reproducible provenance for an official workbook."""
    with workbook_snapshot(path) as (_, provenance):
        return provenance


def load_official_songs(path: Path) -> list[Song]:
    """Load official rows, combining labels that belong to the same song."""
    frame = pd.read_excel(
        path, dtype={"歌曲id": "string"}, keep_default_na=False
    )
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
        lyrics = compose_lyrics(
            row["文本歌词"], row["lrc歌词（滚词）"], row["翻译歌词"]
        )
        songs_by_id[song_id] = Song(
            song_id=song_id,
            labels=frozenset({label}),
            name=name,
            artists=artists,
            genre=_clean_string(row["一级曲风标签"]),
            text=_song_text((name, artists, lyrics)),
            audio_url=_clean_string(row["音频下载地址"]),
        )

    return sorted(songs_by_id.values(), key=lambda song: _song_sort_key(song.song_id))
