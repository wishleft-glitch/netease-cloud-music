import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from emotion_tagging.models import SongRecord
from emotion_tagging.pmemo import load_pmemo_records


class LoadPMEmoRecordsTests(unittest.TestCase):
    def test_joins_metadata_annotations_and_available_assets_with_bom_csv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(
                root,
                metadata=[{
                    "musicId": "42",
                    "title": "夜空",
                    "artist": "歌手",
                    "duration": "241.5",
                    "fileName": "42_chorus.mp3",
                }],
                annotations=[{
                    "musicId": "42",
                    "Valence(mean)": "0.71",
                    "Arousal(mean)": "0.28",
                }],
                bom=True,
            )
            lyric_path = root / "lyrics" / "42.lrc"
            chorus_path = root / "chorus" / "42_chorus.mp3"
            comment_path = root / "comments" / "netease" / "42.txt"
            self._touch(lyric_path)
            self._touch(chorus_path)
            self._touch(comment_path)

            records = load_pmemo_records(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0],
                SongRecord(
                    music_id="42",
                    title="夜空",
                    artist="歌手",
                    duration_seconds=241.5,
                    valence=0.71,
                    arousal=0.28,
                    lyric_path=lyric_path,
                    chorus_path=chorus_path,
                    has_netease_comments=True,
                ),
            )

    def test_marks_missing_assets_as_unavailable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(
                root,
                metadata=[{
                    "musicId": "8",
                    "title": "No files",
                    "artist": "Artist",
                    "duration": "120",
                    "fileName": "missing.mp3",
                }],
                annotations=[{
                    "musicId": "8",
                    "Valence(mean)": "0.4",
                    "Arousal(mean)": "0.6",
                }],
            )

            record = load_pmemo_records(root)[0]

            self.assertIsNone(record.lyric_path)
            self.assertIsNone(record.chorus_path)
            self.assertFalse(record.has_netease_comments)

    def test_marks_existing_comment_directory_as_available(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(
                root,
                metadata=[{
                    "musicId": "9",
                    "title": "Directory comment",
                    "artist": "Artist",
                    "duration": "120",
                    "fileName": "missing.mp3",
                }],
                annotations=[{
                    "musicId": "9",
                    "Valence(mean)": "0.4",
                    "Arousal(mean)": "0.6",
                }],
            )
            (root / "comments" / "netease" / "9.txt").mkdir(parents=True)

            record = load_pmemo_records(root)[0]

            self.assertTrue(record.has_netease_comments)

    def test_ignores_blank_annotation_music_id(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(
                root,
                metadata=[{
                    "musicId": "10",
                    "title": "Valid annotation",
                    "artist": "Artist",
                    "duration": "120",
                    "fileName": "missing.mp3",
                }],
                annotations=[
                    {"musicId": "10", "Valence(mean)": "0.4", "Arousal(mean)": "0.6"},
                    {"musicId": "", "Valence(mean)": "", "Arousal(mean)": ""},
                ],
            )

            records = load_pmemo_records(root)

            self.assertEqual([record.music_id for record in records], ["10"])

    def test_skips_annotations_with_blank_malformed_or_non_finite_scores(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(
                root,
                metadata=[
                    {"musicId": "8", "title": "Valid", "artist": "A", "duration": "1", "fileName": "8.mp3"},
                    {"musicId": "9", "title": "Blank valence", "artist": "A", "duration": "1", "fileName": "9.mp3"},
                    {"musicId": "10", "title": "Blank arousal", "artist": "A", "duration": "1", "fileName": "10.mp3"},
                    {"musicId": "11", "title": "Malformed valence", "artist": "A", "duration": "1", "fileName": "11.mp3"},
                    {"musicId": "12", "title": "Malformed arousal", "artist": "A", "duration": "1", "fileName": "12.mp3"},
                    {"musicId": "13", "title": "NaN valence", "artist": "A", "duration": "1", "fileName": "13.mp3"},
                    {"musicId": "14", "title": "Infinite arousal", "artist": "A", "duration": "1", "fileName": "14.mp3"},
                ],
                annotations=[
                    {"musicId": "8", "Valence(mean)": "0.4", "Arousal(mean)": "0.6"},
                    {"musicId": "9", "Valence(mean)": "", "Arousal(mean)": "0.6"},
                    {"musicId": "10", "Valence(mean)": "0.4", "Arousal(mean)": " "},
                    {"musicId": "11", "Valence(mean)": "invalid", "Arousal(mean)": "0.6"},
                    {"musicId": "12", "Valence(mean)": "0.4", "Arousal(mean)": "invalid"},
                    {"musicId": "13", "Valence(mean)": "nan", "Arousal(mean)": "0.6"},
                    {"musicId": "14", "Valence(mean)": "0.4", "Arousal(mean)": "inf"},
                ],
            )

            records = load_pmemo_records(root)

            self.assertEqual([record.music_id for record in records], ["8"])

    def test_rejects_annotations_missing_required_header(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_csv(
                root / "metadata.csv",
                ["musicId", "title", "artist", "duration", "fileName"],
                [{"musicId": "8", "title": "Song", "artist": "A", "duration": "1", "fileName": "8.mp3"}],
            )
            self._write_csv(
                root / "annotations" / "static_annotations.csv",
                ["musicId", "Valence(mean)"],
                [{"musicId": "8", "Valence(mean)": "0.4"}],
            )

            with self.assertRaisesRegex(ValueError, "Arousal\\(mean\\)"):
                load_pmemo_records(root)

    def test_skips_unannotated_metadata_and_sorts_by_numeric_music_id(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(
                root,
                metadata=[
                    {"musicId": "12", "title": "Twelve", "artist": "A", "duration": "1", "fileName": "12.mp3"},
                    {"musicId": "3", "title": "Three", "artist": "B", "duration": "2", "fileName": "3.mp3"},
                    {"musicId": "7", "title": "Skip", "artist": "C", "duration": "3", "fileName": "7.mp3"},
                ],
                annotations=[
                    {"musicId": "12", "Valence(mean)": "0.1", "Arousal(mean)": "0.2"},
                    {"musicId": "3", "Valence(mean)": "0.3", "Arousal(mean)": "0.4"},
                ],
            )

            records = load_pmemo_records(root)

            self.assertEqual([record.music_id for record in records], ["3", "12"])
            self.assertEqual([record.title for record in records], ["Three", "Twelve"])

    @staticmethod
    def _write_dataset(
        root: Path,
        *,
        metadata: list[dict[str, str]],
        annotations: list[dict[str, str]],
        bom: bool = False,
    ) -> None:
        LoadPMEmoRecordsTests._write_csv(
            root / "metadata.csv",
            ["musicId", "title", "artist", "duration", "fileName"],
            metadata,
            bom=bom,
        )
        LoadPMEmoRecordsTests._write_csv(
            root / "annotations" / "static_annotations.csv",
            ["musicId", "Valence(mean)", "Arousal(mean)"],
            annotations,
            bom=bom,
        )

    @staticmethod
    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], *, bom: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


if __name__ == "__main__":
    unittest.main()
