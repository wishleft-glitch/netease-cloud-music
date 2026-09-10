import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from emotion_tagging.models import TagCard
from emotion_tagging.tag_cards import load_tag_cards


class LoadTagCardsTests(unittest.TestCase):
    def test_loads_ordered_pilot_cards_that_require_human_review(self) -> None:
        cards = load_tag_cards(self._pilot_config_path())

        self.assertEqual([card.name for card in cards], ["热血", "活力", "治愈", "思念", "孤独"])
        self.assertTrue(all(card.requires_human_review for card in cards))

    def test_parses_audio_bounds_including_absent_bounds_as_none(self) -> None:
        cards = {card.name: card for card in load_tag_cards(self._pilot_config_path())}

        self.assertEqual((cards["热血"].min_valence, cards["热血"].max_valence), (0.55, None))
        self.assertEqual((cards["治愈"].min_arousal, cards["治愈"].max_arousal), (0.25, 0.60))
        self.assertEqual(
            (cards["思念"].min_valence, cards["思念"].max_valence, cards["思念"].min_arousal, cards["思念"].max_arousal),
            (None, None, None, None),
        )
        self.assertEqual((cards["孤独"].min_valence, cards["孤独"].max_valence, cards["孤独"].max_arousal), (None, 0.35, 0.45))

    def test_casefolds_lyric_terms_into_tuples(self) -> None:
        cards = self._load_payload([
            self._card(lyrics={"includes": ["RİSE", "HOLD ON"], "excludes": ["NO"]}),
        ])

        self.assertEqual(cards[0].lyric_includes, ("ri\u0307se", "hold on"))
        self.assertEqual(cards[0].lyric_excludes, ("no",))

    def test_returns_frozen_tag_card_instances(self) -> None:
        card = load_tag_cards(self._pilot_config_path())[0]

        self.assertIsInstance(card, TagCard)
        with self.assertRaises(FrozenInstanceError):
            card.name = "changed"

    def test_rejects_missing_required_card_keys(self) -> None:
        for key in ("name", "audio", "lyrics", "requires_human_review"):
            with self.subTest(key=key):
                card = self._card()
                del card[key]

                with self.assertRaisesRegex(ValueError, "missing required fields"):
                    self._load_payload([card])

    def test_rejects_non_list_top_level_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a JSON list"):
            self._load_payload({"name": "not a card list"})

    def test_rejects_non_object_cards_and_nested_sections(self) -> None:
        cases = (
            (["not an object"], "must be an object"),
            ([self._card(audio=[])], "audio and lyrics must be objects"),
            ([self._card(lyrics=[])], "audio and lyrics must be objects"),
            ([self._card(lyrics={"includes": "rise", "excludes": []})], "lyrics.includes must be a list of strings"),
            ([self._card(lyrics={"includes": ["rise"], "excludes": [1]})], "lyrics.excludes must be a list of strings"),
        )
        for payload, message in cases:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, message):
                    self._load_payload(payload)

    def test_rejects_invalid_audio_thresholds(self) -> None:
        invalid_values = (True, "0.5", float("nan"), float("inf"), float("-inf"), -0.01, 1.01)
        for field in ("min_valence", "max_valence", "min_arousal", "max_arousal"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, f"audio.{field}"):
                        self._load_payload([self._card(audio={field: value})])

    @staticmethod
    def _pilot_config_path() -> Path:
        return Path(__file__).resolve().parents[1] / "config" / "pilot_tag_cards.json"

    @staticmethod
    def _card(
        *,
        audio: dict[str, object] | None = None,
        lyrics: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "name": "Test",
            "audio": {} if audio is None else audio,
            "lyrics": {"includes": [], "excludes": []} if lyrics is None else lyrics,
            "requires_human_review": True,
        }

    @staticmethod
    def _load_payload(payload: object) -> list[TagCard]:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "tag_cards.json"
            config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return load_tag_cards(config_path)


if __name__ == "__main__":
    unittest.main()
