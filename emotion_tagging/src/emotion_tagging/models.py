from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TagCard:
    name: str
    min_valence: float | None
    max_valence: float | None
    min_arousal: float | None
    max_arousal: float | None
    lyric_includes: tuple[str, ...]
    lyric_excludes: tuple[str, ...]
    requires_human_review: bool


@dataclass(frozen=True)
class SongRecord:
    music_id: str
    title: str
    artist: str
    duration_seconds: float
    valence: float | None
    arousal: float | None
    lyric_path: Path | None
    chorus_path: Path | None
    has_netease_comments: bool

