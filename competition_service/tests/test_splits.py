from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from competition_emotion.evaluate import metric_report
from competition_emotion.splits import make_holdout, make_stratified_holdout
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

    def test_rejects_empty_and_duplicate_song_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            make_holdout([], test_ratio=0.15, seed=7)
        with self.assertRaisesRegex(ValueError, "unique"):
            make_holdout(
                [song("1", "狂欢"), song("1", "孤独")],
                test_ratio=0.15,
                seed=7,
            )

    def test_uses_python_rounding_for_holdout_size(self) -> None:
        songs = [song(str(index), "狂欢") for index in range(10)]

        assignment = make_holdout(songs, test_ratio=0.15, seed=7)

        self.assertEqual(len(assignment.test_ids), 2)

    def test_balanced_fixture_covers_every_label_when_feasible(self) -> None:
        songs = [
            song("a1", "A"),
            song("a2", "A"),
            song("a3", "A"),
            song("a4", "A"),
            song("a5", "A"),
            song("b1", "B"),
            song("b2", "B"),
            song("b3", "B"),
            song("b4", "B"),
            song("b5", "B"),
        ]

        assignment = make_holdout(songs, test_ratio=0.2, seed=7)
        by_id = {item.song_id: item for item in songs}
        test_labels = set().union(*(by_id[song_id].labels for song_id in assignment.test_ids))

        self.assertEqual(test_labels, {"A", "B"})

    def test_stated_seeds_produce_distinct_assignments_on_crafted_fixture(self) -> None:
        songs = [song(str(index), "A") for index in range(10)]

        first = make_holdout(songs, test_ratio=0.2, seed=7)
        second = make_holdout(songs, test_ratio=0.2, seed=8)

        self.assertNotEqual(first, second)

    def test_stratified_holdout_preserves_singleton_and_multilabel_rates(self) -> None:
        songs = (
            [song(f"a{i}", "A") for i in range(60)]
            + [song(f"b{i}", "B") for i in range(40)]
            + [song(f"m{i}", "A", "B") for i in range(20)]
        )
        first = make_stratified_holdout(songs, 0.2, 42)
        second = make_stratified_holdout(list(reversed(songs)), 0.2, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first.test_ids), 24)
        self.assertEqual(len(first.train_ids), 96)
        by_id = {item.song_id: item for item in songs}
        counts = Counter(tuple(sorted(by_id[song_id].labels)) for song_id in first.test_ids)
        self.assertEqual(counts, {("A",): 12, ("B",): 8, ("A", "B"): 4})
        self.assertFalse(first.train_ids & first.test_ids)

    def test_stratified_holdout_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            make_stratified_holdout([song("x", "A"), song("x", "B")], 0.2, 42)

    def test_stratified_holdout_keeps_rare_singleton_labels_in_fit(self) -> None:
        songs = [song(str(index), "common") for index in range(5)]
        songs += [song(f"rare-{index}", f"rare-{index}") for index in range(5)]
        split = make_stratified_holdout(songs, 0.2, 42)
        self.assertEqual(len(split.test_ids), 2)
        self.assertTrue({f"rare-{index}" for index in range(5)} <= split.train_ids)


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

    def test_rejects_non_finite_metric_values_before_prediction(self) -> None:
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(matrix="y_true", value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    metric_report(
                        np.array([[1.0, bad_value]]),
                        np.array([[0.8, 0.2]]),
                        ("狂欢", "孤独"),
                    )
            with self.subTest(matrix="scores", value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    metric_report(
                        np.array([[1.0, 0.0]]),
                        np.array([[0.8, bad_value]]),
                        ("狂欢", "孤独"),
                    )

    def test_rejects_complex_metric_values_before_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "real"):
            metric_report(
                np.array([[1 + 0j, 0 + 0j]]),
                np.array([[0.8, 0.2]]),
                ("狂欢", "孤独"),
            )
        with self.assertRaisesRegex(ValueError, "real"):
            metric_report(
                np.array([[1, 0]]),
                np.array([[0.8 + 0j, 0.2 + 0j]]),
                ("狂欢", "孤独"),
            )

    def test_rejects_non_binary_or_non_numeric_truth_values(self) -> None:
        for bad_value in (-1, 0.5, 2):
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(ValueError, "binary"):
                    metric_report(
                        np.array([[1, bad_value]]),
                        np.array([[0.8, 0.2]]),
                        ("狂欢", "孤独"),
                    )
        with self.assertRaisesRegex(ValueError, "numeric"):
            metric_report(
                np.array([["1", "0"]]),
                np.array([[0.8, 0.2]]),
                ("狂欢", "孤独"),
            )

    def test_rejects_empty_duplicate_or_blank_labels(self) -> None:
        for labels in ((), ("狂欢", "狂欢"), ("狂欢", ""), ("狂欢", "  ")):
            with self.subTest(labels=labels):
                columns = len(labels)
                with self.assertRaisesRegex(ValueError, "labels"):
                    metric_report(
                        np.zeros((1, columns)),
                        np.zeros((1, columns)),
                        labels,
                    )
        with self.assertRaisesRegex(ValueError, "labels"):
            metric_report(
                np.array([[1, 0]]),
                np.array([[0.8, 0.2]]),
                ("狂欢", 1),
            )

    def test_empty_rows_have_zero_top_one_metrics(self) -> None:
        report = metric_report(
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            ("狂欢", "孤独"),
        )

        self.assertEqual(report["any_positive_top1_accuracy"], 0.0)
        self.assertIsNone(report["strict_top1_accuracy"])
        self.assertEqual(report["macro_recall"], 0.0)
        self.assertEqual(report["per_label_recall"], {"狂欢": 0.0, "孤独": 0.0})
        self.assertEqual(report["confusion_matrix"], [[0, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
