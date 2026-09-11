from __future__ import annotations

import unittest

from competition_emotion.evidence_validator import validate_evidence
from competition_emotion.rubric import DEFAULT_RUBRIC_PATH, load_rubric


class EvidenceValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rubric = load_rubric(DEFAULT_RUBRIC_PATH)

    def test_normalizes_whitespace_for_quote_matching_and_deduplicates_rules(self) -> None:
        result = validate_evidence(
            "一个人 走在夜里",
            "表达独处感。",
            ["一个人走在夜里"],
            ["isolation.alone", "isolation.alone"],
            rubric=self.rubric,
        )
        self.assertEqual(result.quotes, ("一个人走在夜里",))
        self.assertEqual(result.rule_ids, ("isolation.alone",))

    def test_rejects_missing_quote_unknown_rule_and_fabricated_quote(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_evidence("一个人走在夜里", "表达独处感。", rubric=self.rubric)
        with self.assertRaisesRegex(ValueError, "does not occur"):
            validate_evidence("一个人走在夜里", "表达独处感。", ["不在歌词"], rubric=self.rubric)
        with self.assertRaisesRegex(ValueError, "unknown rubric rule"):
            validate_evidence("一个人走在夜里", "表达独处感。", ["一个人走在夜里"], ["fake.rule"], rubric=self.rubric)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            validate_evidence("一个人走在夜里", "表达独处感。", ["一个人走在夜里"], ["sadness.loss"], rubric=self.rubric, label="孤独")

    def test_allows_title_only_evidence_without_quote(self) -> None:
        result = validate_evidence("", "基于歌曲名称进行判定。", rubric=self.rubric)
        self.assertEqual(result.quotes, ())


if __name__ == "__main__":
    unittest.main()
