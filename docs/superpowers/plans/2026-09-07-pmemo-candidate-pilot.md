# PMEmo Emotion Candidate Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PMEmo pilot that creates evidence-backed candidate rows for five configurable emotion labels without presenting those candidates as final business labels.

**Architecture:** A standard-library Python package loads PMEmo metadata, static valence/arousal annotations and local asset availability into one song record per ID. Configured tag cards apply transparent audio ranges and lyric cues, then emit candidate CSV and summary JSON files with separate modality evidence and review status.

**Tech Stack:** Python 3.12, `csv`, `json`, `pathlib`, `dataclasses`, `unittest`; PMEmo 2019 extracted dataset.

---

## Planned file structure

- `F:\netease\_music\emotion_tagging\README.md` — installation, input contract, commands and pilot limitations.
- `F:\netease\_music\emotion_tagging\config\pilot_tag_cards.json` — five tag cards and their audio/lyric candidate rules.
- `F:\netease\_music\emotion_tagging\src\emotion_tagging\__init__.py` — package marker.
- `F:\netease\_music\emotion_tagging\src\emotion_tagging\models.py` — immutable record and output data classes.
- `F:\netease\_music\emotion_tagging\src\emotion_tagging\pmemo.py` — PMEmo CSV/asset loader.
- `F:\netease\_music\emotion_tagging\src\emotion_tagging\tag_cards.py` — tag-card validation and loading.
- `F:\netease\_music\emotion_tagging\src\emotion_tagging\candidates.py` — evidence scoring and candidate decisions.
- `F:\netease\_music\emotion_tagging\src\emotion_tagging\cli.py` — command-line generation of output artifacts.
- `F:\netease\_music\emotion_tagging\tests\test_pmemo.py` — fixture-based data join tests.
- `F:\netease\_music\emotion_tagging\tests\test_tag_cards.py` — configuration validation tests.
- `F:\netease\_music\emotion_tagging\tests\test_candidates.py` — deterministic evidence and review-state tests.
- `F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_candidates.csv` — generated candidate output, excluded from source control if a repository is later created.
- `F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_summary.json` — generated counts and missing-asset audit.

### Task 1: Establish the package and configurable pilot label contract

**Files:**
- Create: `F:\netease\_music\emotion_tagging\src\emotion_tagging\__init__.py`
- Create: `F:\netease\_music\emotion_tagging\src\emotion_tagging\models.py`
- Create: `F:\netease\_music\emotion_tagging\config\pilot_tag_cards.json`
- Create: `F:\netease\_music\emotion_tagging\tests\test_tag_cards.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path
import unittest

from emotion_tagging.tag_cards import load_tag_cards


class TagCardTests(unittest.TestCase):
    def test_loads_all_five_pilot_cards(self) -> None:
        config = Path(__file__).parents[1] / "config" / "pilot_tag_cards.json"
        cards = load_tag_cards(config)
        self.assertEqual(
            [card.name for card in cards],
            ["热血", "活力", "治愈", "思念", "孤独"],
        )
        self.assertTrue(all(card.requires_human_review for card in cards))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify the missing-module failure**

Run:

```powershell
$env:PYTHONPATH='F:\netease\_music\emotion_tagging\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_tag_cards -v
```

Expected: `ModuleNotFoundError: No module named 'emotion_tagging.tag_cards'`.

- [ ] **Step 3: Add the data classes, loader and tag-card configuration**

```python
# src/emotion_tagging/models.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TagCard:
    name: str
    min_valence: float | None
    max_valence: float | None
    min_arousal: float | None
    max_arousal: float | None
    lyric_includes: tuple[str, ...]
    lyric_excludes: tuple[str, ...]
    requires_human_review: bool


@dataclass(frozen=True)
class SongRecord:
    music_id: int
    title: str
    artist: str
    duration_seconds: float
    valence: float
    arousal: float
    lyric_path: Path | None
    chorus_path: Path | None
    has_netease_comments: bool
```

```python
# src/emotion_tagging/tag_cards.py
import json
from pathlib import Path

from .models import TagCard


def load_tag_cards(path: Path) -> list[TagCard]:
    raw_cards = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "audio", "lyrics", "requires_human_review"}
    cards: list[TagCard] = []
    for raw in raw_cards:
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"tag card {raw.get('name', '<unnamed>')} missing {sorted(missing)}")
        audio = raw["audio"]
        lyrics = raw["lyrics"]
        cards.append(TagCard(
            name=str(raw["name"]),
            min_valence=audio.get("min_valence"),
            max_valence=audio.get("max_valence"),
            min_arousal=audio.get("min_arousal"),
            max_arousal=audio.get("max_arousal"),
            lyric_includes=tuple(term.casefold() for term in lyrics.get("includes", [])),
            lyric_excludes=tuple(term.casefold() for term in lyrics.get("excludes", [])),
            requires_human_review=bool(raw["requires_human_review"]),
        ))
    return cards
```

```json
[
  {"name":"热血","audio":{"min_valence":0.55,"min_arousal":0.72},"lyrics":{"includes":["fight","rise","stronger","warrior","battle"],"excludes":[]},"requires_human_review":true},
  {"name":"活力","audio":{"min_valence":0.55,"min_arousal":0.65},"lyrics":{"includes":[],"excludes":[]},"requires_human_review":true},
  {"name":"治愈","audio":{"min_valence":0.65,"min_arousal":0.25,"max_arousal":0.60},"lyrics":{"includes":["heal","peace","home","safe","hold on"],"excludes":[]},"requires_human_review":true},
  {"name":"思念","audio":{},"lyrics":{"includes":["miss you","missing you","without you","come back","remember"],"excludes":[]},"requires_human_review":true},
  {"name":"孤独","audio":{"max_valence":0.35,"max_arousal":0.45},"lyrics":{"includes":["lonely","alone","nobody","empty"],"excludes":[]},"requires_human_review":true}
]
```

- [ ] **Step 4: Run the test to verify the configuration loads**

Run the command from Step 2.

Expected: `OK`.

- [ ] **Step 5: Commit the task if the directory has been initialized as a Git repository**

```powershell
git add src/emotion_tagging/models.py src/emotion_tagging/tag_cards.py config/pilot_tag_cards.json tests/test_tag_cards.py
git commit -m "feat: add configurable pilot tag cards"
```

### Task 2: Load PMEmo songs and asset-availability evidence

**Files:**
- Create: `F:\netease\_music\emotion_tagging\src\emotion_tagging\pmemo.py`
- Create: `F:\netease\_music\emotion_tagging\tests\test_pmemo.py`

- [ ] **Step 1: Write the failing data-join test**

```python
from pathlib import Path
import tempfile
import unittest

from emotion_tagging.pmemo import load_pmemo_records


class PMEmoTests(unittest.TestCase):
    def test_joins_metadata_annotations_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "annotations").mkdir()
            (root / "lyrics").mkdir()
            (root / "chorus").mkdir()
            (root / "comments" / "netease").mkdir(parents=True)
            (root / "metadata.csv").write_text(
                "musicId,fileName,title,artist,album,duration,chorus_start_time,chorus_end_time\n"
                "7,7.mp3,Sample,Artist,Album,30.0,00:00,00:30\n", encoding="utf-8"
            )
            (root / "annotations" / "static_annotations.csv").write_text(
                "musicId,Arousal(mean),Valence(mean)\n7,0.8,0.7\n", encoding="utf-8"
            )
            (root / "lyrics" / "7.lrc").write_text("fight on", encoding="utf-8")
            (root / "chorus" / "7.mp3").write_bytes(b"audio")
            (root / "comments" / "netease" / "7.txt").write_text("great", encoding="utf-8")
            records = load_pmemo_records(root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].music_id, 7)
        self.assertTrue(records[0].lyric_path is not None)
        self.assertTrue(records[0].chorus_path is not None)
        self.assertTrue(records[0].has_netease_comments)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify the missing-loader failure**

Run:

```powershell
$env:PYTHONPATH='F:\netease\_music\emotion_tagging\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_pmemo -v
```

Expected: `ModuleNotFoundError: No module named 'emotion_tagging.pmemo'`.

- [ ] **Step 3: Implement the local-only PMEmo loader**

```python
# src/emotion_tagging/pmemo.py
import csv
from pathlib import Path

from .models import SongRecord


def load_pmemo_records(root: Path) -> list[SongRecord]:
    annotation_path = root / "annotations" / "static_annotations.csv"
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        annotations = {
            int(row["musicId"]): row
            for row in csv.DictReader(handle)
            if row["musicId"].strip()
        }
    records: list[SongRecord] = []
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            music_id = int(row["musicId"])
            annotation = annotations.get(music_id)
            if annotation is None:
                continue
            lyric_path = root / "lyrics" / f"{music_id}.lrc"
            chorus_path = root / "chorus" / row["fileName"]
            comment_path = root / "comments" / "netease" / f"{music_id}.txt"
            records.append(SongRecord(
                music_id=music_id,
                title=row["title"],
                artist=row["artist"],
                duration_seconds=float(row["duration"]),
                valence=float(annotation["Valence(mean)"]),
                arousal=float(annotation["Arousal(mean)"]),
                lyric_path=lyric_path if lyric_path.is_file() else None,
                chorus_path=chorus_path if chorus_path.is_file() else None,
                has_netease_comments=comment_path.is_file(),
            ))
    return sorted(records, key=lambda record: record.music_id)
```

- [ ] **Step 4: Run the test to verify the join**

Run the command from Step 2.

Expected: `OK`.

- [ ] **Step 5: Commit the task if a Git repository exists**

```powershell
git add src/emotion_tagging/pmemo.py tests/test_pmemo.py
git commit -m "feat: load PMEmo metadata and assets"
```

### Task 3: Generate transparent, review-only tag candidates

**Files:**
- Create: `F:\netease\_music\emotion_tagging\src\emotion_tagging\candidates.py`
- Create: `F:\netease\_music\emotion_tagging\tests\test_candidates.py`

- [ ] **Step 1: Write the failing candidate-decision tests**

```python
from pathlib import Path
import tempfile
import unittest

from emotion_tagging.candidates import build_candidate
from emotion_tagging.models import SongRecord, TagCard


class CandidateTests(unittest.TestCase):
    def test_requires_both_audio_and_lyric_evidence_when_card_has_lyrics(self) -> None:
        card = TagCard("热血", 0.55, None, 0.72, None, ("fight",), (), True)
        with tempfile.TemporaryDirectory() as temp:
            lyric = Path(temp) / "song.lrc"
            lyric.write_text("we will fight", encoding="utf-8")
            song = SongRecord(1, "Song", "Artist", 30.0, 0.70, 0.80, lyric, None, False)
            candidate = build_candidate(song, card)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["evidence_state"], "audio_and_lyrics")
        self.assertEqual(candidate["review_status"], "needs_human_review")

    def test_returns_no_candidate_when_audio_rule_fails(self) -> None:
        card = TagCard("活力", 0.55, None, 0.65, None, (), (), True)
        song = SongRecord(2, "Song", "Artist", 30.0, 0.50, 0.80, None, None, False)
        self.assertIsNone(build_candidate(song, card))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify the missing-module failure**

Run:

```powershell
$env:PYTHONPATH='F:\netease\_music\emotion_tagging\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_candidates -v
```

Expected: `ModuleNotFoundError: No module named 'emotion_tagging.candidates'`.

- [ ] **Step 3: Implement candidate scoring with no automatic final-label state**

```python
# src/emotion_tagging/candidates.py
from pathlib import Path

from .models import SongRecord, TagCard


def _within(value: float, minimum: float | None, maximum: float | None) -> bool:
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _lyric_matches(path: Path | None, includes: tuple[str, ...], excludes: tuple[str, ...]) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8", errors="replace").casefold()
    if any(term in text for term in excludes):
        return []
    return [term for term in includes if term in text]


def build_candidate(song: SongRecord, card: TagCard) -> dict[str, object] | None:
    audio_ok = _within(song.valence, card.min_valence, card.max_valence) and _within(
        song.arousal, card.min_arousal, card.max_arousal
    )
    lyric_hits = _lyric_matches(song.lyric_path, card.lyric_includes, card.lyric_excludes)
    lyric_required = bool(card.lyric_includes)
    has_audio_rule = any(value is not None for value in (
        card.min_valence, card.max_valence, card.min_arousal, card.max_arousal
    ))
    if has_audio_rule and not audio_ok:
        return None
    if lyric_required and not lyric_hits:
        return None
    if lyric_required and audio_ok:
        evidence_state = "audio_and_lyrics"
    elif lyric_required:
        evidence_state = "lyrics_only"
    else:
        evidence_state = "audio_only"
    return {
        "music_id": song.music_id,
        "title": song.title,
        "artist": song.artist,
        "tag": card.name,
        "valence": song.valence,
        "arousal": song.arousal,
        "audio_evidence": "within_configured_range" if audio_ok else "no_audio_rule",
        "lyric_evidence": "|".join(lyric_hits),
        "evidence_state": evidence_state,
        "review_status": "needs_human_review" if card.requires_human_review else "candidate_only",
        "is_final_label": False,
    }
```

- [ ] **Step 4: Run the tests to verify candidate behavior**

Run the command from Step 2.

Expected: two passing tests and no candidate marked `is_final_label=True`.

- [ ] **Step 5: Commit the task if a Git repository exists**

```powershell
git add src/emotion_tagging/candidates.py tests/test_candidates.py
git commit -m "feat: score review-only tag candidates"
```

### Task 4: Write candidate CSV and audit summary artifacts

**Files:**
- Create: `F:\netease\_music\emotion_tagging\src\emotion_tagging\cli.py`
- Create: `F:\netease\_music\emotion_tagging\README.md`

- [ ] **Step 1: Write the failing output-artifact test**

Add `from emotion_tagging.cli import write_outputs` beside the existing imports, then add this method inside the existing `CandidateTests` class:

```python
    def test_write_outputs_includes_candidate_and_coverage_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            candidate = {
                "music_id": 1, "title": "Song", "artist": "Artist", "tag": "活力",
                "valence": 0.8, "arousal": 0.8, "audio_evidence": "within_configured_range",
                "lyric_evidence": "", "evidence_state": "audio_only",
                "review_status": "needs_human_review", "is_final_label": False,
            }
            write_outputs([candidate], 3, 2, output_dir)
            self.assertIn("活力", (output_dir / "pmemo_pilot_candidates.csv").read_text(encoding="utf-8-sig"))
            self.assertIn('"songs_with_lyrics": 2', (output_dir / "pmemo_pilot_summary.json").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run the test to verify the missing-output failure**

Run:

```powershell
$env:PYTHONPATH='F:\netease\_music\emotion_tagging\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_candidates -v
```

Expected: `ImportError: cannot import name 'write_outputs'`.

- [ ] **Step 3: Implement CSV and JSON outputs plus the command-line entry point**

```python
# src/emotion_tagging/cli.py
import argparse
import csv
import json
from pathlib import Path

from .candidates import build_candidate
from .pmemo import load_pmemo_records
from .tag_cards import load_tag_cards


def write_outputs(candidates: list[dict[str, object]], songs_total: int, songs_with_lyrics: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "music_id", "title", "artist", "tag", "valence", "arousal", "audio_evidence",
        "lyric_evidence", "evidence_state", "review_status", "is_final_label",
    ]
    with (output_dir / "pmemo_pilot_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    by_tag: dict[str, int] = {}
    for candidate in candidates:
        tag = str(candidate["tag"])
        by_tag[tag] = by_tag.get(tag, 0) + 1
    summary = {
        "songs_total": songs_total,
        "songs_with_lyrics": songs_with_lyrics,
        "songs_missing_lyrics": songs_total - songs_with_lyrics,
        "candidate_rows": len(candidates),
        "candidates_by_tag": by_tag,
        "final_labels_emitted": 0,
    }
    (output_dir / "pmemo_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmemo-root", type=Path, required=True)
    parser.add_argument("--tag-cards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_pmemo_records(args.pmemo_root)
    cards = load_tag_cards(args.tag_cards)
    candidates = [candidate for song in records for card in cards if (candidate := build_candidate(song, card))]
    write_outputs(candidates, len(records), sum(song.lyric_path is not None for song in records), args.output_dir)


if __name__ == "__main__":
    main()
```

```markdown
# PMEmo candidate pilot

This project generates review-only emotion-tag candidates from PMEmo. It does not create final business labels.

Run from `F:\netease\_music\emotion_tagging`:

```powershell
$env:PYTHONPATH="$PWD\src"
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m emotion_tagging.cli --pmemo-root 'F:\netease\_music\PMEmo\data_2019\extracted\PMEmo2019' --tag-cards '.\config\pilot_tag_cards.json' --output-dir '.\outputs'
```

Each candidate row keeps audio and lyric evidence separate and has `is_final_label=False`. Replace `pilot_tag_cards.json` with official tag cards only after the competition definitions and gold examples are available.
```

- [ ] **Step 4: Run all tests and the pilot command**

Run:

```powershell
$env:PYTHONPATH='F:\netease\_music\emotion_tagging\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -s tests -v
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m emotion_tagging.cli --pmemo-root 'F:\netease\_music\PMEmo\data_2019\extracted\PMEmo2019' --tag-cards 'F:\netease\_music\emotion_tagging\config\pilot_tag_cards.json' --output-dir 'F:\netease\_music\emotion_tagging\outputs'
Get-Content -LiteralPath 'F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_summary.json'
```

Expected: all tests pass; candidate CSV and summary JSON exist; `final_labels_emitted` equals `0`.

- [ ] **Step 5: Commit the task if a Git repository exists**

```powershell
git add src/emotion_tagging/cli.py README.md tests/test_candidates.py
git commit -m "feat: export PMEmo pilot candidates"
```

### Task 5: Inspect the actual pilot result and record the limitation

**Files:**
- Modify: `F:\netease\_music\emotion_tagging\README.md`

- [ ] **Step 1: Inspect candidate distribution and duplicate rows**

Run:

```powershell
Import-Csv 'F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_candidates.csv' |
  Group-Object tag | Select-Object Name,Count
Import-Csv 'F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_candidates.csv' |
  Group-Object music_id,tag | Where-Object Count -gt 1
```

Expected: one count per configured tag and no duplicate `music_id`/`tag` pair.

- [ ] **Step 2: Add the observed counts and non-production limitation to README**

Append a `Pilot result` section containing the generated tag counts, the lyric-coverage count from the summary file, and this exact statement:

```markdown
PMEmo candidates are not final labels and must not be used to claim the competition's 95% precision target. Official tag definitions, authorized competition songs and an independent human-labelled evaluation set are required before production calibration.
```

- [ ] **Step 3: Re-run tests and inspect final artifacts**

Run:

```powershell
$env:PYTHONPATH='F:\netease\_music\emotion_tagging\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -s tests -v
Get-Item 'F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_candidates.csv','F:\netease\_music\emotion_tagging\outputs\pmemo_pilot_summary.json' | Select-Object FullName,Length,LastWriteTime
```

Expected: all tests pass and both output files have a non-zero size.

- [ ] **Step 4: Commit the task if a Git repository exists**

```powershell
git add README.md outputs/pmemo_pilot_summary.json
git commit -m "docs: record PMEmo pilot limitations"
```

## Plan self-review

- Spec coverage: shared song features (Task 2), tag cards (Task 1), dual evidence and non-final candidate status (Task 3), output/audit artifacts (Task 4), and PMEmo limitations (Task 5) are all covered.
- No-placeholder check: every task gives file paths, commands, tests and expected outcomes.
- Interface check: `TagCard`, `SongRecord`, `load_pmemo_records`, `load_tag_cards`, `build_candidate` and `write_outputs` use the same names in every dependent task.
