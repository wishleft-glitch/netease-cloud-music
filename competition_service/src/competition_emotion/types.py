from dataclasses import dataclass


@dataclass(frozen=True)
class Song:
    song_id: str
    labels: frozenset[str]
    name: str
    artists: str
    genre: str
    text: str
    audio_url: str


@dataclass(frozen=True)
class SplitAssignment:
    train_ids: frozenset[str]
    test_ids: frozenset[str]
