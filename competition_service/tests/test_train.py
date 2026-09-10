from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from competition_emotion.models import load_text_scorer
from competition_emotion.train import train_text_baseline


LABELS = ("狂欢", "孤独")
REQUIRED_COLUMNS = (
    "歌曲id", "情绪类型", "歌曲名称", "一级曲风标签", "演唱艺人", "文本歌词",
    "音频下载地址", "lrc歌词（滚词）", "翻译歌词",
)


class TrainTextBaselineTests(unittest.TestCase):
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "official.xlsx"
            bundle_dir = root / "bundle"
            pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_excel(workbook, index=False)

            report = train_text_baseline(
                workbook, bundle_dir, labels=LABELS, seed=19, test_ratio=0.25
            )

            self.assertEqual(report["report_schema_version"], 1)
            self.assertEqual(report["model_type"], "lyrics_tfidf_logreg")
            self.assertEqual(report["labels"], list(LABELS))
            self.assertEqual(report["counts"]["total_songs"], 12)
            self.assertEqual(report["counts"]["train_songs"], 9)
            self.assertEqual(report["counts"]["test_songs"], 3)
            self.assertEqual(report["counts"]["split_group_overlap"], 0)
            self.assertIn("strict_top1_accuracy", report["evaluation_metrics"])
            self.assertEqual(set(report["label_support"]["train"]), set(LABELS))

            self.assertTrue((bundle_dir / "model.joblib").is_file())
            self.assertTrue((bundle_dir / "report.json").is_file())
            prediction_lines = (bundle_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prediction_lines), 3)
            prediction = json.loads(prediction_lines[0])
            self.assertEqual(
                set(prediction),
                {"song_id", "title", "artist", "language", "predicted_label", "confidence", "top_labels", "top_scores"},
            )
            self.assertNotIn("labels", prediction)
            self.assertEqual(len(prediction["top_labels"]), 2)
            self.assertTrue(0.0 <= prediction["confidence"] <= 1.0)
            self.assertEqual(json.loads((bundle_dir / "report.json").read_text(encoding="utf-8")), report)

            scorer = load_text_scorer(bundle_dir / "model.joblib", trusted=True)
            self.assertEqual(scorer.labels, LABELS)
            self.assertEqual(set(scorer.score("一个人 寂寞")), set(LABELS))


if __name__ == "__main__":
    unittest.main()
