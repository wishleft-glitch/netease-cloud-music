from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
import unittest

from competition_emotion.iteration import (
    TraceRecord,
    append_trace,
    build_rubric_patch_proposal,
    evaluate_patch_gate,
    main,
    mine_hard_cases,
    publish_rubric_candidate,
    read_traces,
)


class IterationLoopTests(unittest.TestCase):
    def _record(self, song_id: str, margin: float, *, reviewer_label: str | None = None) -> TraceRecord:
        return TraceRecord(
            trace_id=f"trace-{song_id}",
            song_id=song_id,
            model_version="model-v1",
            rubric_version="emotion-rubric-v1",
            top_label="孤独",
            top_confidence=0.5 + margin / 2,
            second_label="思念",
            second_confidence=0.5 - margin / 2,
            candidates=("孤独", "思念", "悲伤"),
            evidence="基于歌词",
            reviewer_used=reviewer_label is not None,
            reviewer_label=reviewer_label,
            reviewer_confidence=0.9 if reviewer_label else None,
            reviewer_evidence="复核证据" if reviewer_label else "",
        )

    def test_trace_round_trip_and_hard_case_mining_is_deterministic(self) -> None:
        records = [self._record("2", 0.2), self._record("1", 0.01), self._record("3", 0.3, reviewer_label="悲伤")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            for record in records:
                append_trace(path, record)
            loaded = read_traces(path)

        self.assertEqual([item.trace_id for item in loaded], [item.trace_id for item in records])
        self.assertTrue(all(item.created_at_utc for item in loaded))
        cases = mine_hard_cases(loaded, margin_threshold=0.05)
        self.assertEqual([case["song_id"] for case in cases], ["1", "3"])
        self.assertIn("low_margin", cases[0]["reasons"])
        self.assertIn("reviewer_disagreement", cases[1]["reasons"])

    def test_patch_proposal_is_explicitly_unpublished(self) -> None:
        proposal = build_rubric_patch_proposal(
            mine_hard_cases([self._record("1", 0.01)], margin_threshold=0.05),
            base_rubric_version="emotion-rubric-v1",
        )
        self.assertEqual(proposal["status"], "proposed")
        self.assertTrue(proposal["requires_human_review"])
        self.assertEqual(proposal["changes"][0]["label"], "孤独")

    def test_patch_gate_blocks_metric_regression_and_accepts_safe_candidate(self) -> None:
        baseline = {
            "strict_top1_accuracy": 0.61,
            "macro_recall": 0.56,
            "per_label_recall": {"孤独": 0.60, "思念": 0.52},
        }
        self.assertFalse(
            evaluate_patch_gate(
                baseline,
                {"strict_top1_accuracy": 0.60, "macro_recall": 0.56, "per_label_recall": {"孤独": 0.60, "思念": 0.52}},
            )["passed"]
        )
        result = evaluate_patch_gate(
            baseline,
            {"strict_top1_accuracy": 0.612, "macro_recall": 0.561, "per_label_recall": {"孤独": 0.60, "思念": 0.53}},
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

    def test_cli_writes_hard_case_queue_and_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            output = root / "proposal.json"
            append_trace(trace_path, self._record("1", 0.01))
            self.assertEqual(
                main(["--trace-path", str(trace_path), "--output", str(output)]),
                0,
            )
            payload = __import__("json").loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["hard_cases"]), 1)
            self.assertEqual(payload["proposal"]["status"], "proposed")

    def test_publish_requires_human_approval_and_passed_gate(self) -> None:
        from competition_emotion.rubric import DEFAULT_RUBRIC_PATH

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.json"
            destination = root / "published.json"
            shutil.copyfile(DEFAULT_RUBRIC_PATH, candidate)
            with self.assertRaisesRegex(ValueError, "approval"):
                publish_rubric_candidate(candidate, destination, gate_result={"passed": True}, approved=False)
            with self.assertRaisesRegex(ValueError, "gate"):
                publish_rubric_candidate(candidate, destination, gate_result={"passed": False}, approved=True)
            result = publish_rubric_candidate(candidate, destination, gate_result={"passed": True}, approved=True)
            self.assertEqual(result["rubric_version"], "emotion-rubric-v2")
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["rubric_version"], "emotion-rubric-v2")


if __name__ == "__main__":
    unittest.main()
