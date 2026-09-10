from __future__ import annotations

import unittest

import numpy as np

from competition_emotion.evaluate import metric_report
from competition_emotion.splits import make_holdout
from competition_emotion.types import Song


def song(song_id: str, *labels: str) -> Song:
    return Song(
        song_id=song_id,
        labels=frozenset(labels),
        name=song_id,
        artists="artist",
        genre="genre",
        text="text",
        audio_url="",
    )


class HoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.songs = [
            song("1", "狂欢"),
            song("2", "孤独"),
            song("3", "狂欢", "孤独"),
            song("4", "狂欢"),
            song("5", "孤独"),
            song("6", "狂欢"),
            song("7", "孤独"),
        ]

    def test_same_seed_makes_exact_disjoint_group_holdout(self) -> None:
        first = make_holdout(self.songs, test_ratio=0.29, seed=7)
        second = make_holdout(self.songs, test_ratio=0.29, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first.test_ids), round(len(self.songs) * 0.29))
        self.assertFalse(first.train_ids & first.test_ids)
        self.assertEqual(first.train_ids | first.test_ids, {item.song_id for item in self.songs})

    def test_different_seeds_still_make_valid_group_holdouts(self) -> None:
        first = make_holdout(self.songs, test_ratio=0.29, seed=7)
        second = make_holdout(self.songs, test_ratio=0.29, seed=8)

        for assignment in (first, second):
            self.assertEqual(len(assignment.test_ids), round(len(self.songs) * 0.29))
            self.assertFalse(assignment.train_ids & assignment.test_ids)
            self.assertEqual(
                assignment.train_ids | assignment.test_ids,
                {item.song_id for item in self.songs},
            )

    def test_rejects_out_of_range_test_ratios(self) -> None:
        for ratio in (0.0, 0.049, 0.5, 1.0):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    make_holdout(self.songs, test_ratio=ratio, seed=7)


class MetricReportTests(unittest.TestCase):
    def test_reports_top_one_metrics_and_singleton_confusion_matrix(self) -> None:
        report = metric_report(
            np.array([[1, 0], [0, 1], [0, 1]]),
            np.array([[0.8, 0.1], [0.1, 0.9], [0.3, 0.6]]),
            ("狂欢", "孤独"),
        )

        self.assertEqual(report["any_positive_top1_accuracy"], 1.0)
        self.assertEqual(report["strict_top1_accuracy"], 1.0)
        self.assertEqual(report["macro_recall"], 1.0)
        self.assertEqual(report["per_label_recall"], {"狂欢": 1.0, "孤独": 1.0})
        self.assertEqual(report["confusion_matrix"], [[1, 0], [0, 2]])

    def test_multilabel_row_counts_only_for_any_positive_accuracy(self) -> None:
        report = metric_report(
            np.array([[1, 1], [1, 0]]),
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            ("狂欢", "孤独"),
        )

        self.assertEqual(report["any_positive_top1_accuracy"], 1.0)
        self.assertEqual(report["strict_top1_accuracy"], 1.0)
        self.assertEqual(report["macro_recall"], 0.5)
        self.assertEqual(report["per_label_recall"], {"狂欢": 1.0, "孤独": 0.0})
        self.assertEqual(report["confusion_matrix"], [[1, 0], [0, 0]])

    def test_reports_empty_singleton_metrics_without_singleton_rows(self) -> None:
        report = metric_report(
            np.array([[1, 1], [0, 0]]),
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            ("狂欢", "孤独"),
        )

        self.assertEqual(report["any_positive_top1_accuracy"], 0.5)
        self.assertIsNone(report["strict_top1_accuracy"])
        self.assertEqual(report["macro_recall"], 0.0)
        self.assertEqual(report["per_label_recall"], {"狂欢": 0.0, "孤独": 0.0})
        self.assertEqual(report["confusion_matrix"], [[0, 0], [0, 0]])

    def test_rejects_incompatible_metric_shapes(self) -> None:
        with self.assertRaises(ValueError):
            metric_report(
                np.array([[1, 0]]),
                np.array([[0.8, 0.2], [0.1, 0.9]]),
                ("狂欢", "孤独"),
            )
        with self.assertRaises(ValueError):
            metric_report(
                np.array([[1, 0]]),
                np.array([[0.8, 0.2]]),
                ("狂欢",),
            )


if __name__ == "__main__":
    unittest.main()
