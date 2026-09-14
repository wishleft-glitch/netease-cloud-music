"""Validated public-playlist features used by the optional prior model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MAX_PLAYLIST_CATEGORIES = 64
MAX_PLAYLIST_SONGS = 200_000


def load_playlist_features(
    path: str | Path,
) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    """Load a category-to-playlist JSON snapshot as binary song features."""
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("playlist prior must be a UTF-8 JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError("playlist prior must be a JSON object")
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, dict) or not raw_categories:
        raise ValueError("playlist prior must contain categories")
    categories = tuple(str(category).strip() for category in raw_categories)
    if (
        len(categories) > MAX_PLAYLIST_CATEGORIES
        or any(not category for category in categories)
        or len(set(categories)) != len(categories)
    ):
        raise ValueError("playlist prior has invalid categories")

    features: dict[str, list[float]] = {}
    for index, category in enumerate(categories):
        playlists = raw_categories[category]
        if not isinstance(playlists, dict):
            raise ValueError("playlist prior category must contain playlist objects")
        for track_ids in playlists.values():
            if not isinstance(track_ids, list):
                raise ValueError("playlist prior track lists must be arrays")
            for track_id in track_ids:
                song_id = str(track_id).strip()
                if not song_id:
                    continue
                row = features.setdefault(song_id, [0.0] * len(categories))
                row[index] = 1.0
                if len(features) > MAX_PLAYLIST_SONGS:
                    raise ValueError("playlist prior contains too many songs")

    materialized = {song_id: tuple(values) for song_id, values in features.items()}
    provenance = {
        "file_name": source.name,
        "sha256": digest,
        "categories": list(categories),
        "song_count": len(materialized),
    }
    return materialized, provenance
