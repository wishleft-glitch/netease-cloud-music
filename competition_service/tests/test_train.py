from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from competition_emotion.models import load_text_scorer
from competition_emotion.train import train_text_baseline


LABELS = ("狂欢", "孤独")
REQUIRED_COLUMNS = (
    "歌曲id", "情绪类型", "歌曲名称", "一级曲风标签", "演唱艺人", "文本歌词",
    "音频下载地址", "lrc歌词（滚词）", "翻译歌词",
)


class TrainTextBaselineTests(unittest.TestCase):
    def test_uses_one_snapshot_when_source_is_replaced_before_loading(self) -> None:
        original_rows = []
        for index in range(12):
            label = LABELS[index % 2]
            original_rows.append(
                {
                    "歌曲id": str(index + 1),
                    "情绪类型": label,
                    "歌曲名称": f"original {index}",
                    "一级曲风标签": "流行",
                    "演唱艺人": "艺人",
                    "文本歌词": "派对 跳舞 欢呼" if label == "狂欢" else "一个人 夜晚 寂寞",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                }
            )
        original_rows.append({**original_rows[0], "情绪类型": "孤独"})
        replacement_rows = [
            {
                **row,
                "歌曲id": str(index + 101),
                "歌曲名称": f"replacement {index}",
            }
            for index, row in enumerate(original_rows[:8])
        ]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "official.xlsx"
            replacement = root / "replacement.xlsx"
            bundle_dir = root / "bundle"
            pd.DataFrame(original_rows, columns=REQUIRED_COLUMNS).to_excel(workbook, index=False)
            expected_sha256 = hashlib.sha256(workbook.read_bytes()).hexdigest()
            pd.DataFrame(replacement_rows, columns=REQUIRED_COLUMNS).to_excel(replacement, index=False)
            from competition_emotion.train import load_official_songs as real_loader

            observed_loader_path: Path | None = None

            def replace_source_before_loading(loader_path: Path) -> list[object]:
                nonlocal observed_loader_path
                observed_loader_path = loader_path
                replacement.replace(workbook)
                return real_loader(loader_path)

            with patch(
                "competition_emotion.train.load_official_songs",
                side_effect=replace_source_before_loading,
            ):
                report = train_text_baseline(
                    workbook, bundle_dir, labels=LABELS, seed=19, test_ratio=0.25
                )

            self.assertIsNotNone(observed_loader_path)
            self.assertNotEqual(observed_loader_path, workbook)
            self.assertEqual(
                report["source"],
                {
                    "file_name": "official.xlsx",
                    "sha256": expected_sha256,
                    "rows": 13,
                },
            )
            self.assertEqual(report["counts"]["total_songs"], 12)

    def test_end_to_end_writes_group_safe_evaluation_bundle(self) -> None:
        rows = []
        for index in range(12):
            label = LABELS[index % 2]
            rows.append(
                {
                    "歌曲id": str(index + 1),
                    "情绪类型": label,
                    "歌曲名称": f"{label}歌曲{index}",
                    "一级曲风标签": "流行",
                    "演唱艺人": f"艺人{index}",
                    "文本歌词": "派对 跳舞 欢呼" if label == "狂欢" else "一个人 夜晚 寂寞",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                }
            )
        rows.append({**rows[0], "情绪类型": "孤独"})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "official.xlsx"
            bundle_dir = root / "bundle"
            pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_excel(workbook, index=False)

            report = train_text_baseline(
                workbook, bundle_dir, labels=LABELS, seed=19, test_ratio=0.25
            )

            self.assertEqual(report["report_schema_version"], 2)
            self.assertEqual(report["model_type"], "lyrics_tfidf_logreg")
            self.assertEqual(report["labels"], list(LABELS))
            self.assertEqual(
                report["source"],
                {
                    "file_name": "official.xlsx",
                    "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                    "rows": 13,
                },
            )
            self.assertEqual(report["counts"]["total_songs"], 12)
            self.assertEqual(report["counts"]["train_songs"], 9)
            self.assertEqual(report["counts"]["test_songs"], 3)
            self.assertEqual(report["counts"]["split_group_overlap"], 0)
            self.assertEqual(report["evaluation_sample_counts"], {
                "any_positive_sample_count": 3,
                "strict_singleton_sample_count": 2,
                "multi_label_sample_count": 1,
            })
            self.assertIn("strict_top1_accuracy", report["evaluation_metrics"])
            self.assertEqual(set(report["label_support"]["train"]), set(LABELS))

            pointer = json.loads((bundle_dir / "current.json").read_text(encoding="utf-8"))
            version_dir = bundle_dir / pointer["active_bundle"]
            self.assertEqual(pointer["pointer_schema_version"], 1)
            self.assertFalse((bundle_dir / "model.joblib").exists())
            self.assertFalse((bundle_dir / "report.json").exists())
            self.assertFalse((bundle_dir / "predictions.jsonl").exists())
            self.assertTrue((version_dir / "model.joblib").is_file())
            self.assertTrue((version_dir / "report.json").is_file())
            prediction_lines = (version_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prediction_lines), 3)
            prediction = json.loads(prediction_lines[0])
            self.assertEqual(
                set(prediction),
                {"song_id", "title", "artist", "language", "predicted_label", "confidence", "top_labels", "top_scores"},
            )
            self.assertNotIn("labels", prediction)
            self.assertEqual(len(prediction["top_labels"]), 2)
            self.assertTrue(0.0 <= prediction["confidence"] <= 1.0)
            self.assertEqual(json.loads((version_dir / "report.json").read_text(encoding="utf-8")), report)

            scorer = load_text_scorer(version_dir / "model.joblib", trusted=True)
            self.assertEqual(scorer.labels, LABELS)
            self.assertEqual(set(scorer.score("一个人 寂寞")), set(LABELS))

            with patch("competition_emotion.train._atomic_write_text", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    train_text_baseline(workbook, bundle_dir, labels=LABELS, seed=20, test_ratio=0.25)

            # The failed candidate can exist as an unreachable immutable version,
            # but every reader still resolves the previous complete bundle.
            self.assertEqual(
                json.loads((bundle_dir / "current.json").read_text(encoding="utf-8")),
                pointer,
            )
            self.assertTrue((version_dir / "model.joblib").is_file())
            self.assertTrue((version_dir / "report.json").is_file())
            self.assertTrue((version_dir / "predictions.jsonl").is_file())
            self.assertEqual(list(bundle_dir.glob(".staging-*")), [])


if __name__ == "__main__":
    unittest.main()
