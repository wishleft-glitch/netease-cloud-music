"""Deterministic Dev/calibration split kept strictly inside official Train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .data import load_official_songs, workbook_snapshot
from .splits import make_holdout
from .types import Song


CALIBRATION_SCHEMA_VERSION = 1


class CalibrationAssignment:
    """Fit/Dev IDs derived only from the already fixed official Train IDs."""

    __slots__ = ("fit_ids", "dev_ids")

    def __init__(self, fit_ids: frozenset[str], dev_ids: frozenset[str]) -> None:
        self.fit_ids = frozenset(fit_ids)
        self.dev_ids = frozenset(dev_ids)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CalibrationAssignment) and self.fit_ids == other.fit_ids and self.dev_ids == other.dev_ids

    def __repr__(self) -> str:
        return f"CalibrationAssignment(fit_ids={self.fit_ids!r}, dev_ids={self.dev_ids!r})"


def make_calibration_split(
    train_songs: list[Song], *, calibration_ratio: float = 0.15, seed: int = 20260911
) -> CalibrationAssignment:
    """Reserve a deterministic Dev slice from the official Train songs."""
    if not train_songs:
        raise ValueError("train_songs cannot be empty")
    assignment = make_holdout(train_songs, test_ratio=calibration_ratio, seed=seed)
    return CalibrationAssignment(assignment.train_ids, assignment.test_ids)


def build_calibration_manifest(
    *,
    source: dict[str, object],
    total_ids: set[str] | frozenset[str],
    official_test_ids: set[str] | frozenset[str],
    calibration: CalibrationAssignment,
    calibration_ratio: float,
    official_seed: int,
    calibration_seed: int,
    official_test_ratio: float = 0.2,
) -> dict[str, Any]:
    total = frozenset(total_ids)
    official_test = frozenset(official_test_ids)
    fit_ids = calibration.fit_ids
    dev_ids = calibration.dev_ids
    if not official_test <= total or fit_ids & official_test or dev_ids & official_test:
        raise ValueError("calibration split overlaps official test or leaves source IDs")
    if fit_ids & dev_ids or fit_ids | dev_ids != total - official_test:
        raise ValueError("calibration fit/dev IDs do not reconcile with official train")
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source": dict(source),
        "official_split": {"seed": official_seed, "test_ratio": official_test_ratio},
        "calibration_split": {"seed": calibration_seed, "dev_ratio": calibration_ratio},
        "counts": {"total": len(total), "official_test": len(official_test), "fit": len(fit_ids), "dev": len(dev_ids)},
        "overlap": {"fit_test": len(fit_ids & official_test), "dev_test": len(dev_ids & official_test), "fit_dev": len(fit_ids & dev_ids)},
        "ids": {"official_test": sorted(official_test), "fit": sorted(fit_ids), "dev": sorted(dev_ids)},
    }


def create_calibration_manifest(
    workbook: Path,
    output: Path,
    *,
    official_seed: int = 20260910,
    test_ratio: float = 0.2,
    calibration_seed: int = 20260911,
    calibration_ratio: float = 0.15,
) -> dict[str, Any]:
    with workbook_snapshot(Path(workbook)) as (snapshot_path, source):
        songs = load_official_songs(snapshot_path)
    official = make_holdout(songs, test_ratio=test_ratio, seed=official_seed)
    train_songs = [song for song in songs if song.song_id in official.train_ids]
    calibration = make_calibration_split(train_songs, calibration_ratio=calibration_ratio, seed=calibration_seed)
    manifest = build_calibration_manifest(
        source=source,
        total_ids={song.song_id for song in songs},
        official_test_ids=official.test_ids,
        calibration=calibration,
        calibration_ratio=calibration_ratio,
        official_seed=official_seed,
        official_test_ratio=test_ratio,
        calibration_seed=calibration_seed,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a Dev/calibration manifest nested inside official Train")
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-seed", default=20260910, type=int)
    parser.add_argument("--test-ratio", default=0.2, type=float)
    parser.add_argument("--calibration-seed", default=20260911, type=int)
    parser.add_argument("--calibration-ratio", default=0.15, type=float)
    arguments = parser.parse_args(argv)
    manifest = create_calibration_manifest(
        arguments.workbook,
        arguments.output,
        official_seed=arguments.official_seed,
        test_ratio=arguments.test_ratio,
        calibration_seed=arguments.calibration_seed,
        calibration_ratio=arguments.calibration_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
