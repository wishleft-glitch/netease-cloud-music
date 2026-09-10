from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from competition_emotion.data import (
    clean_lyric,
    load_official_songs,
    workbook_provenance,
    workbook_snapshot,
)


REQUIRED_COLUMNS = [
    "歌曲id",
    "情绪类型",
    "歌曲名称",
    "一级曲风标签",
    "演唱艺人",
    "文本歌词",
    "音频下载地址",
    "lrc歌词（滚词）",
    "翻译歌词",
]


class OfficialSongLoaderTests(unittest.TestCase):
    def write_workbook(self, rows: list[dict[str, object]]) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "official_songs.xlsx"
        pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_excel(path, index=False)
        return path

    def test_merges_same_song_and_strips_lrc_timestamps(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "10",
                    "情绪类型": "欢快",
                    "歌曲名称": "晴天",
                    "一级曲风标签": "流行",
                    "演唱艺人": "歌手 A",
                    "文本歌词": "阳光 正好",
                    "音频下载地址": "https://audio.example/10",
                    "lrc歌词（滚词）": "[01:02.34] 微风吹来",
                    "翻译歌词": "A gentle breeze",
                },
                {
                    "歌曲id": "10",
                    "情绪类型": "活力",
                    "歌曲名称": "should not replace",
                    "一级曲风标签": "摇滚",
                    "演唱艺人": "other artist",
                    "文本歌词": "other lyric",
                    "音频下载地址": "https://audio.example/other",
                    "lrc歌词（滚词）": "[02:03] other lrc",
                    "翻译歌词": "other translation",
                },
            ]
        )

        songs = load_official_songs(path)

        self.assertEqual(len(songs), 1)
        song = songs[0]
        self.assertEqual(song.labels, frozenset({"欢快", "活力"}))
        self.assertEqual(song.name, "晴天")
        self.assertEqual(song.genre, "流行")
        self.assertEqual(song.audio_url, "https://audio.example/10")
        self.assertEqual(song.text, "晴天 歌手 A 阳光 正好\n微风吹来\nA gentle breeze")

    def test_uses_lrc_when_plaintext_and_translation_are_empty(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "11",
                    "情绪类型": "平静",
                    "歌曲名称": "LRC only",
                    "一级曲风标签": "流行",
                    "演唱艺人": "Singer",
                    "文本歌词": "",
                    "音频下载地址": "https://audio.example/11",
                    "lrc歌词（滚词）": "[00:02] only  lrc",
                    "翻译歌词": "",
                }
            ]
        )

        song = load_official_songs(path)[0]

        self.assertEqual(song.text, "LRC only Singer only lrc")

    def test_workbook_provenance_has_safe_name_sha256_and_raw_row_count(self) -> None:
        rows = [
            {
                "歌曲id": "1",
                "情绪类型": "平静",
                "歌曲名称": "one",
                "一级曲风标签": "",
                "演唱艺人": "",
                "文本歌词": "",
                "音频下载地址": "",
                "lrc歌词（滚词）": "",
                "翻译歌词": "",
            },
            {
                "歌曲id": "",
                "情绪类型": "未知标签",
                "歌曲名称": "invalid but raw",
                "一级曲风标签": "",
                "演唱艺人": "",
                "文本歌词": "",
                "音频下载地址": "",
                "lrc歌词（滚词）": "",
                "翻译歌词": "",
            },
        ]
        path = self.write_workbook(rows)

        provenance = workbook_provenance(path)

        self.assertEqual(provenance["file_name"], "official_songs.xlsx")
        self.assertEqual(
            provenance["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(provenance["rows"], 2)
        self.assertEqual(set(provenance), {"file_name", "sha256", "rows"})

    def test_workbook_provenance_uses_one_snapshot_when_source_is_replaced(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "1",
                    "情绪类型": "平静",
                    "歌曲名称": "original",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                }
            ]
        )
        expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        replacement = path.with_name("replacement.xlsx")
        pd.DataFrame(
            [
                {
                    "歌曲id": "2",
                    "情绪类型": "欢快",
                    "歌曲名称": "replacement one",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                },
                {
                    "歌曲id": "3",
                    "情绪类型": "活力",
                    "歌曲名称": "replacement two",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                },
            ],
            columns=REQUIRED_COLUMNS,
        ).to_excel(replacement, index=False)
        original_read_excel = pd.read_excel
        replaced = False

        def replace_source_before_read(snapshot_path: object, *args: object, **kwargs: object) -> pd.DataFrame:
            nonlocal replaced
            if not replaced:
                replacement.replace(path)
                replaced = True
            return original_read_excel(snapshot_path, *args, **kwargs)

        with patch(
            "competition_emotion.data.pd.read_excel",
            side_effect=replace_source_before_read,
        ):
            provenance = workbook_provenance(path)

        self.assertTrue(replaced)
        self.assertEqual(provenance["sha256"], expected_sha256)
        self.assertEqual(provenance["rows"], 1)

    def test_workbook_snapshot_cleans_up_after_success_and_error(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "1",
                    "情绪类型": "平静",
                    "歌曲名称": "original",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                }
            ]
        )

        with workbook_snapshot(path) as (snapshot_path, provenance):
            successful_snapshot = snapshot_path
            self.assertTrue(snapshot_path.is_file())
            self.assertEqual(provenance["rows"], 1)
        self.assertFalse(successful_snapshot.exists())

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with workbook_snapshot(path) as (snapshot_path, _):
                failed_snapshot = snapshot_path
                raise RuntimeError("test failure")
        self.assertFalse(failed_snapshot.exists())

    def test_none_lyrics_produce_valid_metadata_text(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "1",
                    "情绪类型": "平静",
                    "歌曲名称": "  安静  ",
                    "一级曲风标签": None,
                    "演唱艺人": "  某人 ",
                    "文本歌词": None,
                    "音频下载地址": None,
                    "lrc歌词（滚词）": None,
                    "翻译歌词": None,
                }
            ]
        )

        song = load_official_songs(path)[0]

        self.assertEqual(song.name, "安静")
        self.assertEqual(song.artists, "某人")
        self.assertEqual(song.genre, "")
        self.assertEqual(song.audio_url, "")
        self.assertEqual(song.text, "安静 某人")

    def test_clean_lyric_removes_timestamps_and_handles_none(self) -> None:
        self.assertEqual(clean_lyric(" [00:12] hello\n[01:02.34] world "), "hello world")
        self.assertEqual(clean_lyric(None), "")

    def test_missing_required_header_names_missing_column(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "missing_header.xlsx"
        pd.DataFrame(
            [
                {
                    "歌曲id": "1",
                    "情绪类型": "平静",
                    "歌曲名称": "歌",
                    "一级曲风标签": "流行",
                    "演唱艺人": "人",
                    "文本歌词": "词",
                    "音频下载地址": "url",
                    "lrc歌词（滚词）": "lrc",
                }
            ]
        ).to_excel(path, index=False)

        with self.assertRaisesRegex(ValueError, "翻译歌词"):
            load_official_songs(path)

    def test_skips_empty_ids_and_invalid_labels(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "",
                    "情绪类型": "欢快",
                    "歌曲名称": "empty",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                },
                {
                    "歌曲id": "3",
                    "情绪类型": "未知标签",
                    "歌曲名称": "invalid",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                },
            ]
        )

        self.assertEqual(load_official_songs(path), [])

    def test_sorts_numeric_song_ids_numerically(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": "10",
                    "情绪类型": "欢快",
                    "歌曲名称": "ten",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                },
                {
                    "歌曲id": "2",
                    "情绪类型": "平静",
                    "歌曲名称": "two",
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                },
            ]
        )

        self.assertEqual([song.song_id for song in load_official_songs(path)], ["2", "10"])

    def test_sorts_special_song_ids_without_crashing(self) -> None:
        path = self.write_workbook(
            [
                {
                    "歌曲id": song_id,
                    "情绪类型": "平静",
                    "歌曲名称": song_id,
                    "一级曲风标签": "",
                    "演唱艺人": "",
                    "文本歌词": "",
                    "音频下载地址": "",
                    "lrc歌词（滚词）": "",
                    "翻译歌词": "",
                }
                for song_id in ("sNaN", "NaN", "Infinity", "1", "10")
            ]
        )

        songs = load_official_songs(path)

        self.assertEqual(
            [song.song_id for song in songs], ["1", "10", "Infinity", "NaN", "sNaN"]
        )

if __name__ == "__main__":
    unittest.main()
