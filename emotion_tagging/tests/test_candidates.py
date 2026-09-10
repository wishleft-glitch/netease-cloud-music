from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import unittest
from unittest.mock import patch

from emotion_tagging.candidates import _lyric_matches, _within, build_candidate
from emotion_tagging.cli import OUTPUT_FIELDS, main, write_outputs
from emotion_tagging.models import SongRecord, TagCard


class CandidateTests(unittest.TestCase):
    def test_within_accepts_inclusive_bounds_and_missing_bounds(self) -> None:
        self.assertTrue(_within(0.5, 0.5, 0.5))
        self.assertTrue(_within(0.5, None, None))
        self.assertFalse(_within(0.49, 0.5, None))
        self.assertFalse(_within(0.51, None, 0.5))

    def test_lyric_matches_returns_includes_in_order_and_honors_exclusions(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("RISE and fight", encoding="utf-8")

            self.assertEqual(_lyric_matches(lyric_path, ("fight", "rise"), ()), ["fight", "rise"])
            self.assertEqual(_lyric_matches(lyric_path, ("fight",), ("rise",)), [])

    def test_lyric_matches_returns_no_hits_when_path_is_missing(self) -> None:
        self.assertEqual(_lyric_matches(None, ("fight",), ()), [])

    def test_hot_blood_candidate_needs_passing_audio_and_fight_lyric(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("We will FIGHT tonight", encoding="utf-8")
            song = self._song(lyric_path=lyric_path, valence=0.7, arousal=0.8)
            card = self._card(
                name="热血",
                min_valence=0.55,
                min_arousal=0.7,
                lyric_includes=("fight",),
            )

            candidate = build_candidate(song, card)

        self.assertEqual(
            candidate,
            {
                "music_id": "7",
                "title": "Battle Song",
                "artist": "Artist",
                "tag": "热血",
                "valence": 0.7,
                "arousal": 0.8,
                "audio_evidence": "within_configured_range",
                "lyric_evidence": "fight",
                "evidence_state": "audio_and_lyrics",
                "review_status": "needs_human_review",
                "is_final_label": False,
            },
        )

    def test_returns_none_when_vitality_audio_rule_fails(self) -> None:
        song = self._song(valence=0.8, arousal=0.2)
        card = self._card(name="活力", min_arousal=0.6)

        self.assertIsNone(build_candidate(song, card))

    def test_returns_none_when_required_lyric_is_absent_or_does_not_match(self) -> None:
        card = self._card(name="热血", lyric_includes=("fight",))

        self.assertIsNone(build_candidate(self._song(lyric_path=None), card))
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("soft and quiet", encoding="utf-8")
            self.assertIsNone(build_candidate(self._song(lyric_path=lyric_path), card))

    def test_excluded_lyric_suppresses_candidate(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("fight no more", encoding="utf-8")
            card = self._card(name="热血", lyric_includes=("fight",), lyric_excludes=("no",))

            self.assertIsNone(build_candidate(self._song(lyric_path=lyric_path), card))

    def test_exclude_only_card_suppresses_candidate(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("no more", encoding="utf-8")
            card = self._card(name="活力", min_arousal=0.4, lyric_excludes=("no",))

            self.assertIsNone(build_candidate(self._song(lyric_path=lyric_path), card))

    def test_returns_none_when_card_has_only_lyric_exclusions(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("safe to continue", encoding="utf-8")
            card = self._card(name="未分类", lyric_excludes=("no",))

            self.assertIsNone(build_candidate(self._song(lyric_path=lyric_path), card))

    def test_returns_none_when_card_has_no_audio_or_lyric_rules(self) -> None:
        self.assertIsNone(build_candidate(self._song(), self._card(name="未分类")))

    def test_audio_only_card_ignores_unavailable_lyric_path(self) -> None:
        missing_lyric_path = Path("missing-lyrics.lrc")
        card = self._card(name="活力", min_arousal=0.4)

        candidate = build_candidate(self._song(lyric_path=missing_lyric_path), card)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["evidence_state"], "audio_only")
        self.assertEqual(candidate["lyric_evidence"], "")

    def test_lyric_rule_returns_none_when_lyric_path_is_unavailable(self) -> None:
        missing_lyric_path = Path("missing-lyrics.lrc")
        card = self._card(name="思念", lyric_includes=("miss",))

        self.assertIsNone(build_candidate(self._song(lyric_path=missing_lyric_path), card))

    def test_lyrics_only_missing_candidate_has_non_final_candidate_state(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            lyric_path = Path(temporary_directory) / "song.lrc"
            lyric_path.write_text("I miss you", encoding="utf-8")
            card = self._card(
                name="思念",
                lyric_includes=("miss",),
                requires_human_review=False,
            )

            candidate = build_candidate(self._song(lyric_path=lyric_path), card)

        self.assertEqual(candidate["audio_evidence"], "no_audio_rule")
        self.assertEqual(candidate["lyric_evidence"], "miss")
        self.assertEqual(candidate["evidence_state"], "lyrics_only")
        self.assertEqual(candidate["review_status"], "candidate_only")
        self.assertFalse(candidate["is_final_label"])

    def test_write_outputs_writes_bom_csv_and_review_only_summary(self) -> None:
        candidate = {
            "music_id": "7",
            "title": "Battle Song",
            "artist": "Artist",
            "tag": "活力",
            "valence": 0.7,
            "arousal": 0.8,
            "audio_evidence": "within_configured_range",
            "lyric_evidence": "",
            "evidence_state": "audio_only",
            "review_status": "needs_human_review",
            "is_final_label": False,
        }
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "nested" / "outputs"

            write_outputs([candidate], songs_total=3, songs_with_lyrics=2, output_dir=output_dir)

            csv_path = output_dir / "pmemo_pilot_candidates.csv"
            summary_path = output_dir / "pmemo_pilot_summary.json"
            self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with csv_path.open(encoding="utf-8-sig", newline="") as file_handle:
                rows = list(csv.DictReader(file_handle))
            self.assertEqual(rows[0]["tag"], "活力")
            self.assertEqual(tuple(rows[0]), OUTPUT_FIELDS)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["songs_with_lyrics"], 2)
        self.assertEqual(summary["final_labels_emitted"], 0)
        self.assertEqual(summary["candidates_by_tag"], {"活力": 1})

    def test_write_outputs_keeps_empty_csv_header_and_zero_candidate_counts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"

            write_outputs([], songs_total=2, songs_with_lyrics=0, output_dir=output_dir)

            with (output_dir / "pmemo_pilot_candidates.csv").open(encoding="utf-8-sig", newline="") as file_handle:
                self.assertEqual(next(csv.reader(file_handle)), list(OUTPUT_FIELDS))
                self.assertEqual(list(csv.reader(file_handle)), [])
            summary = json.loads((output_dir / "pmemo_pilot_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["songs_missing_lyrics"], 2)
        self.assertEqual(summary["candidate_rows"], 0)
        self.assertEqual(summary["candidates_by_tag"], {})
        self.assertEqual(summary["final_labels_emitted"], 0)

    def test_write_outputs_rejects_non_review_only_candidate_before_publishing(self) -> None:
        candidate = {
            "music_id": "7",
            "title": "Battle Song",
            "artist": "Artist",
            "tag": "活力",
            "valence": 0.7,
            "arousal": 0.8,
            "audio_evidence": "within_configured_range",
            "lyric_evidence": "",
            "evidence_state": "audio_only",
            "review_status": "needs_human_review",
            "is_final_label": True,
        }
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"

            with self.assertRaisesRegex(ValueError, "is_final_label"):
                write_outputs([candidate], songs_total=1, songs_with_lyrics=1, output_dir=output_dir)

            self.assertFalse((output_dir / "pmemo_pilot_candidates.csv").exists())
            self.assertFalse((output_dir / "pmemo_pilot_summary.json").exists())

    def test_write_outputs_preserves_existing_pair_when_summary_serialization_fails(self) -> None:
        candidate = {
            "music_id": "7",
            "title": "Battle Song",
            "artist": "Artist",
            "tag": "活力",
            "valence": 0.7,
            "arousal": 0.8,
            "audio_evidence": "within_configured_range",
            "lyric_evidence": "",
            "evidence_state": "audio_only",
            "review_status": "needs_human_review",
            "is_final_label": False,
        }
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            output_dir.mkdir()
            csv_path = output_dir / "pmemo_pilot_candidates.csv"
            summary_path = output_dir / "pmemo_pilot_summary.json"
            csv_path.write_bytes(b"old-csv")
            summary_path.write_bytes(b"old-json")

            with patch("emotion_tagging.cli.json.dumps", side_effect=TypeError("cannot serialize summary")):
                with self.assertRaisesRegex(TypeError, "cannot serialize summary"):
                    write_outputs([candidate], songs_total=1, songs_with_lyrics=1, output_dir=output_dir)

            self.assertEqual(csv_path.read_bytes(), b"old-csv")
            self.assertEqual(summary_path.read_bytes(), b"old-json")
            self.assertEqual(list(output_dir.glob(".pmemo_pilot_*")), [])

    def test_write_outputs_rolls_back_pair_when_summary_publish_fails(self) -> None:
        candidate = {
            "music_id": "7",
            "title": "Battle Song",
            "artist": "Artist",
            "tag": "活力",
            "valence": 0.7,
            "arousal": 0.8,
            "audio_evidence": "within_configured_range",
            "lyric_evidence": "",
            "evidence_state": "audio_only",
            "review_status": "needs_human_review",
            "is_final_label": False,
        }
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            output_dir.mkdir()
            csv_path = output_dir / "pmemo_pilot_candidates.csv"
            summary_path = output_dir / "pmemo_pilot_summary.json"
            csv_path.write_bytes(b"old-csv")
            summary_path.write_bytes(b"old-json")
            original_replace = Path.replace

            def fail_only_summary_publish(source: Path, destination: str | Path) -> Path:
                destination_path = Path(destination)
                if (
                    destination_path == summary_path
                    and source.name.startswith(".pmemo_pilot_summary.")
                    and ".backup." not in source.name
                ):
                    raise OSError("cannot publish summary")
                return original_replace(source, destination_path)

            with patch.object(Path, "replace", new=fail_only_summary_publish):
                with self.assertRaisesRegex(OSError, "cannot publish summary"):
                    write_outputs([candidate], songs_total=1, songs_with_lyrics=1, output_dir=output_dir)

            self.assertEqual(csv_path.read_bytes(), b"old-csv")
            self.assertEqual(summary_path.read_bytes(), b"old-json")
            self.assertEqual(list(output_dir.glob(".pmemo_pilot_*")), [])

    def test_write_outputs_preserves_untouched_summary_when_its_backup_move_fails(self) -> None:
        candidate = {
            "music_id": "7",
            "title": "Battle Song",
            "artist": "Artist",
            "tag": "活力",
            "valence": 0.7,
            "arousal": 0.8,
            "audio_evidence": "within_configured_range",
            "lyric_evidence": "",
            "evidence_state": "audio_only",
            "review_status": "needs_human_review",
            "is_final_label": False,
        }
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            output_dir.mkdir()
            csv_path = output_dir / "pmemo_pilot_candidates.csv"
            summary_path = output_dir / "pmemo_pilot_summary.json"
            csv_path.write_bytes(b"old-csv")
            summary_path.write_bytes(b"old-json")
            original_replace = Path.replace

            def fail_only_summary_backup(source: Path, destination: str | Path) -> Path:
                destination_path = Path(destination)
                if (
                    source == summary_path
                    and destination_path.name.startswith(".pmemo_pilot_summary.backup.")
                ):
                    raise OSError("cannot back up summary")
                return original_replace(source, destination_path)

            with patch.object(Path, "replace", new=fail_only_summary_backup):
                with self.assertRaisesRegex(OSError, "cannot back up summary"):
                    write_outputs([candidate], songs_total=1, songs_with_lyrics=1, output_dir=output_dir)

            self.assertEqual(csv_path.read_bytes(), b"old-csv")
            self.assertEqual(summary_path.read_bytes(), b"old-json")
            self.assertEqual(list(output_dir.glob(".pmemo_pilot_*")), [])

    def test_main_reports_missing_pmemo_root_as_parser_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stderr = StringIO()
            arguments = [
                "emotion_tagging.cli",
                "--pmemo-root",
                str(root / "missing-pmemo"),
                "--tag-cards",
                str(root / "tag_cards.json"),
                "--output-dir",
                str(root / "outputs"),
            ]

            with patch("sys.argv", arguments), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--pmemo-root", stderr.getvalue())
        self.assertIn("not a directory", stderr.getvalue())

    def test_main_reports_output_file_as_parser_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pmemo_root = root / "pmemo"
            (pmemo_root / "annotations").mkdir(parents=True)
            (pmemo_root / "metadata.csv").write_text(
                "musicId,title,artist,duration,fileName\n",
                encoding="utf-8",
            )
            (pmemo_root / "annotations" / "static_annotations.csv").write_text(
                "musicId,Valence(mean),Arousal(mean)\n",
                encoding="utf-8",
            )
            tag_cards = root / "tag_cards.json"
            tag_cards.write_text("[]", encoding="utf-8")
            output_file = root / "output-file"
            output_file.write_text("not a directory", encoding="utf-8")
            stderr = StringIO()
            arguments = [
                "emotion_tagging.cli",
                "--pmemo-root",
                str(pmemo_root),
                "--tag-cards",
                str(tag_cards),
                "--output-dir",
                str(output_file),
            ]

            with patch("sys.argv", arguments), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--output-dir", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    @staticmethod
    def _song(
        *,
        lyric_path: Path | None = None,
        valence: float | None = 0.5,
        arousal: float | None = 0.5,
    ) -> SongRecord:
        return SongRecord(
            music_id="7",
            title="Battle Song",
            artist="Artist",
            duration_seconds=180.0,
            valence=valence,
            arousal=arousal,
            lyric_path=lyric_path,
            chorus_path=None,
            has_netease_comments=False,
        )

    @staticmethod
    def _card(
        *,
        name: str,
        min_valence: float | None = None,
        max_valence: float | None = None,
        min_arousal: float | None = None,
        max_arousal: float | None = None,
        lyric_includes: tuple[str, ...] = (),
        lyric_excludes: tuple[str, ...] = (),
        requires_human_review: bool = True,
    ) -> TagCard:
        return TagCard(
            name=name,
            min_valence=min_valence,
            max_valence=max_valence,
            min_arousal=min_arousal,
            max_arousal=max_arousal,
            lyric_includes=lyric_includes,
            lyric_excludes=lyric_excludes,
            requires_human_review=requires_human_review,
        )


if __name__ == "__main__":
    unittest.main()
