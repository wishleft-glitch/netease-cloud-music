from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from competition_emotion.constants import LABELS
from competition_emotion.rubric import DEFAULT_RUBRIC_PATH, load_rubric


class RubricTests(unittest.TestCase):
    def test_default_rubric_is_versioned_and_covers_exact_official_labels(self) -> None:
        rubric = load_rubric(DEFAULT_RUBRIC_PATH)

        self.assertEqual(rubric.version, "emotion-rubric-v1")
        self.assertEqual(rubric.labels, LABELS)
        context = rubric.context(("孤独", "思念"))
        self.assertEqual([item["label"] for item in context], ["孤独", "思念"])
        self.assertTrue(all(item["rule_ids"] for item in context))

    def test_loader_rejects_missing_or_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rubric.json"
            payload = json.loads(DEFAULT_RUBRIC_PATH.read_text(encoding="utf-8"))
            payload["labels"] = payload["labels"][:-1]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official labels"):
                load_rubric(path)

            payload = json.loads(DEFAULT_RUBRIC_PATH.read_text(encoding="utf-8"))
            payload["entries"][0]["rule_ids"] = payload["entries"][1]["rule_ids"]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rule IDs"):
                load_rubric(path)

    def test_context_rejects_unknown_candidate(self) -> None:
        rubric = load_rubric(DEFAULT_RUBRIC_PATH)
        with self.assertRaisesRegex(ValueError, "unknown rubric label"):
            rubric.context(("不存在",))


if __name__ == "__main__":
    unittest.main()
