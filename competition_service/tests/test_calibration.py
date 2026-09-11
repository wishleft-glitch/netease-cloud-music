from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from competition_emotion.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    build_calibration_manifest,
    make_calibration_split,
)
from competition_emotion.splits import make_holdout
from competition_emotion.types import Song


def song(song_id: str, *labels: str) -> Song:
    return Song(song_id, frozenset(labels), song_id, "artist", "genre", "text", "")


class CalibrationSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.songs = [song(str(index), "狂欢" if index % 2 else "孤独") for index in range(20)]

    def test_calibration_is_nested_inside_fixed_train_and_disjoint_from_test(self) -> None:
        official = make_holdout(self.songs, test_ratio=0.2, seed=20260910)
        calibration = make_calibration_split(
            [item for item in self.songs if item.song_id in official.train_ids],
            calibration_ratio=0.15,
            seed=20260911,
        )

        self.assertFalse(calibration.fit_ids & calibration.dev_ids)
        self.assertFalse(calibration.dev_ids & official.test_ids)
        self.assertEqual(calibration.fit_ids | calibration.dev_ids, official.train_ids)

    def test_manifest_contains_reproducible_counts_and_ids(self) -> None:
        official = make_holdout(self.songs, test_ratio=0.2, seed=20260910)
        calibration = make_calibration_split(
            [item for item in self.songs if item.song_id in official.train_ids],
            calibration_ratio=0.15,
            seed=20260911,
        )
        manifest = build_calibration_manifest(
            source={"file_name": "official.xlsx", "sha256": "abc", "rows": 20},
            total_ids={item.song_id for item in self.songs},
            official_test_ids=official.test_ids,
            calibration=calibration,
            calibration_ratio=0.15,
            official_seed=20260910,
            calibration_seed=20260911,
        )

        self.assertEqual(manifest["schema_version"], CALIBRATION_SCHEMA_VERSION)
        self.assertEqual(manifest["counts"], {"total": 20, "official_test": 4, "fit": 14, "dev": 2})
        self.assertEqual(manifest["overlap"], {"fit_test": 0, "dev_test": 0, "fit_dev": 0})
        self.assertEqual(manifest["ids"]["official_test"], sorted(official.test_ids))
        json.dumps(manifest, ensure_ascii=False)

    def test_rejects_empty_or_invalid_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "songs"):
            make_calibration_split([], calibration_ratio=0.15, seed=1)
        with self.assertRaises(ValueError):
            make_calibration_split(self.songs, calibration_ratio=0.5, seed=1)


if __name__ == "__main__":
    unittest.main()
