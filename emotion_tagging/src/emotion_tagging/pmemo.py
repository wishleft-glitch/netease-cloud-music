"""Load PMEmo metadata, static annotations, and local asset availability."""

import csv
import math
from pathlib import Path

from .models import SongRecord


_ANNOTATION_COLUMNS = frozenset({"musicId", "Valence(mean)", "Arousal(mean)"})


def load_pmemo_records(root: Path) -> list[SongRecord]:
    """Return annotated PMEmo songs with paths for assets that exist locally."""
    annotations = _load_annotations(root / "annotations" / "static_annotations.csv")
    records: list[tuple[int, SongRecord]] = []

    for metadata in _read_csv(root / "metadata.csv"):
        music_id = int(metadata["musicId"])
        annotation = annotations.get(music_id)
        if annotation is None:
            continue

        valence, arousal = annotation
        music_id_text = str(music_id)
        lyric_path = _existing_file(root / "lyrics" / f"{music_id_text}.lrc")
        chorus_path = _existing_file(root / "chorus" / metadata["fileName"])
        records.append((music_id, SongRecord(
            music_id=music_id_text,
            title=metadata["title"],
            artist=metadata["artist"],
            duration_seconds=float(metadata["duration"]),
            valence=valence,
            arousal=arousal,
            lyric_path=lyric_path,
            chorus_path=chorus_path,
            has_netease_comments=(root / "comments" / "netease" / f"{music_id_text}.txt").exists(),
        )))

    records.sort(key=lambda item: item[0])
    return [record for _, record in records]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _load_annotations(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        missing_columns = _ANNOTATION_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Annotation CSV is missing required columns: {missing}")

        annotations: dict[int, tuple[float, float]] = {}
        for row in reader:
            parsed_annotation = _parse_annotation(row)
            if parsed_annotation is not None:
                music_id, valence, arousal = parsed_annotation
                annotations[music_id] = (valence, arousal)
        return annotations


def _parse_annotation(row: dict[str, str | None]) -> tuple[int, float, float] | None:
    raw_music_id = row.get("musicId")
    valence = _finite_float(row.get("Valence(mean)"))
    arousal = _finite_float(row.get("Arousal(mean)"))
    if raw_music_id is None or not raw_music_id.strip() or valence is None or arousal is None:
        return None

    try:
        music_id = int(raw_music_id)
    except ValueError:
        return None
    return music_id, valence, arousal


def _finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        numeric_value = float(value)
    except ValueError:
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def _existing_file(path: Path) -> Path | None:
    return path if path.is_file() else None
