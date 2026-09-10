# Competition Baseline Emotion Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible CPU-first baseline that trains from the official multi-label song table, selects one highest-confidence emotion, and serves the required internal HTTP API.

**Architecture:** A competition-only Python package keeps official-data preparation, group-level validation, text/audio scoring, fusion, and HTTP serving separate from the PMEmo pilot. Training merges all positive labels for a song, validates by song ID, produces calibrated per-label scores, and exports a local model bundle. The online service downloads and analyzes request audio, combines it with available lyrics, returns the highest-ranked label, and records only measured evidence.

**Tech Stack:** Python 3.12; `pandas`, `openpyxl`, `numpy`, `scikit-learn`, `joblib`, `httpx`, `fastapi`, `uvicorn`, `pydantic`; `ffmpeg` executable for audio decoding; `unittest`.

---

## Planned file structure

- `competition_service/requirements.txt` — pinned Python dependencies.
- `competition_service/README.md` — local setup, train, test, serve and benchmark commands.
- `competition_service/src/competition_emotion/constants.py` — 15 labels and model/version constants.
- `competition_service/src/competition_emotion/types.py` — immutable song, split, score and evidence types.
- `competition_service/src/competition_emotion/data.py` — official workbook loader and lyrics normalization.
- `competition_service/src/competition_emotion/splits.py` — deterministic song-group, multilabel-aware holdout and fold assignment.
- `competition_service/src/competition_emotion/audio.py` — bounded download, `ffmpeg` decode and measured audio features.
- `competition_service/src/competition_emotion/models.py` — text, audio and fusion scorers plus model-bundle persistence.
- `competition_service/src/competition_emotion/evaluate.py` — strict/any-positive Top-1, macro recall, confusion and latency report.
- `competition_service/src/competition_emotion/train.py` — CLI that trains baselines and writes a versioned bundle/report.
- `competition_service/src/competition_emotion/service.py` — FastAPI request validation, inference, evidence and JSON error responses.
- `competition_service/src/competition_emotion/runtime.py` — startup model loading, bounded worker pools and request analyzer.
- `competition_service/src/competition_emotion/main.py` — environment-driven ASGI application entry point.
- `competition_service/scripts/start-service.ps1` — bind to a supplied internal host/port.
- `competition_service/scripts/restart-service.ps1` — stop and relaunch the saved service process.
- `competition_service/scripts/health-check.ps1` — test the local health endpoint.
- `competition_service/scripts/load_test.py` — 5 QPS and 10 QPS request runner.
- `competition_service/tests/` — unit and integration tests.
- `competition_service/docs/cost-estimate.md` — populated cost formula and measurement fields.
- `competition_service/docs/operations.md` — architecture, deployment, monitoring and five troubleshooting cases.

The existing `emotion_tagging/` package remains a PMEmo regression pilot. No file under it is changed by this plan.

### Task 1: Create the competition package and load official multi-label songs

**Files:**
- Create: `competition_service/requirements.txt`
- Create: `competition_service/src/competition_emotion/__init__.py`
- Create: `competition_service/src/competition_emotion/constants.py`
- Create: `competition_service/src/competition_emotion/types.py`
- Create: `competition_service/src/competition_emotion/data.py`
- Create: `competition_service/tests/test_data.py`

- [ ] **Step 1: Add the dependencies and constants**

```text
# competition_service/requirements.txt
fastapi==0.115.12
httpx==0.28.1
joblib==1.4.2
numpy==2.2.5
openpyxl==3.1.5
pandas==2.2.3
pydantic==2.11.4
scikit-learn==1.6.1
uvicorn==0.34.2
```

```python
# competition_service/src/competition_emotion/constants.py
LABELS = (
    "狂欢", "热血励志", "活力", "欢快", "甜蜜", "自信", "浪漫", "放松",
    "治愈", "思念", "平静", "孤独", "悲伤", "苦情", "抑郁",
)
LABEL_SET = frozenset(LABELS)
MODEL_VERSION = "baseline-v1"
MAX_AUDIO_SECONDS = 45
```

- [ ] **Step 2: Write failing loader tests**

```python
# competition_service/tests/test_data.py
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from competition_emotion.data import clean_lyric, load_official_songs


class OfficialDataTests(unittest.TestCase):
    def test_merges_rows_by_song_id_and_keeps_all_positive_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "official.xlsx"
            pd.DataFrame([
                {"歌曲id": 1, "情绪类型": "孤独", "歌曲名称": "歌", "一级曲风标签": "流行", "演唱艺人": "甲", "文本歌词": None, "音频下载地址": "https://a/1.mp3", "lrc歌词（滚词）": "[00:01]一个人", "翻译歌词": None},
                {"歌曲id": 1, "情绪类型": "悲伤", "歌曲名称": "歌", "一级曲风标签": "流行", "演唱艺人": "甲", "文本歌词": None, "音频下载地址": "https://a/1.mp3", "lrc歌词（滚词）": "[00:01]一个人", "翻译歌词": None},
                {"歌曲id": 2, "情绪类型": "活力", "歌曲名称": "Song", "一级曲风标签": "电子", "演唱艺人": "B", "文本歌词": "move", "音频下载地址": "https://a/2.mp3", "lrc歌词（滚词）": None, "翻译歌词": None},
            ]).to_excel(path, index=False)
            songs = load_official_songs(path)
        self.assertEqual([song.song_id for song in songs], ["1", "2"])
        self.assertEqual(songs[0].labels, frozenset({"孤独", "悲伤"}))
        self.assertIn("一个人", songs[0].text)

    def test_clean_lyric_removes_lrc_timestamps_and_null_text(self) -> None:
        self.assertEqual(clean_lyric("[00:01.23] hello\n[01:02]世界"), "hello 世界")
        self.assertEqual(clean_lyric(None), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to confirm the missing-module failure**

Run:

```powershell
Set-Location 'F:\netease\_music\competition_service'
$env:PYTHONPATH = "$PWD\src"
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_data -v
```

Expected: `ModuleNotFoundError: No module named 'competition_emotion.data'`.

- [ ] **Step 4: Implement the immutable source contract and loader**

```python
# competition_service/src/competition_emotion/types.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Song:
    song_id: str
    labels: frozenset[str]
    name: str
    artists: str
    genre: str
    text: str
    audio_url: str


@dataclass(frozen=True)
class SplitAssignment:
    train_ids: frozenset[str]
    test_ids: frozenset[str]
```

```python
# competition_service/src/competition_emotion/data.py
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .constants import LABEL_SET
from .types import Song

REQUIRED_COLUMNS = frozenset({
    "歌曲id", "情绪类型", "歌曲名称", "一级曲风标签", "演唱艺人",
    "文本歌词", "音频下载地址", "lrc歌词（滚词）", "翻译歌词",
})
TIMESTAMP = re.compile(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]")


def clean_lyric(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(TIMESTAMP.sub(" ", str(value)).split())


def _string(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def load_official_songs(path: Path) -> list[Song]:
    frame = pd.read_excel(path, dtype={"歌曲id": str})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"official workbook missing columns: {sorted(missing)}")
    grouped: dict[str, dict[str, object]] = {}
    labels_by_song: defaultdict[str, set[str]] = defaultdict(set)
    for row in frame.to_dict(orient="records"):
        song_id = _string(row["歌曲id"])
        label = _string(row["情绪类型"])
        if not song_id or label not in LABEL_SET:
            continue
        labels_by_song[song_id].add(label)
        grouped.setdefault(song_id, row)
    songs = []
    for song_id in sorted(grouped, key=lambda item: (len(item), item)):
        row = grouped[song_id]
        parts = [
            _string(row["歌曲名称"]), _string(row["演唱艺人"]),
            clean_lyric(row["文本歌词"]), clean_lyric(row["lrc歌词（滚词）"]),
            clean_lyric(row["翻译歌词"]),
        ]
        songs.append(Song(
            song_id=song_id, labels=frozenset(labels_by_song[song_id]),
            name=_string(row["歌曲名称"]), artists=_string(row["演唱艺人"]),
            genre=_string(row["一级曲风标签"]), text=" ".join(part for part in parts if part),
            audio_url=_string(row["音频下载地址"]),
        ))
    return songs
```

- [ ] **Step 5: Run the focused tests and commit**

Run the command from Step 3. Expected: `Ran 2 tests ... OK`.

```powershell
git add competition_service/requirements.txt competition_service/src/competition_emotion/constants.py competition_service/src/competition_emotion/types.py competition_service/src/competition_emotion/data.py competition_service/tests/test_data.py
git commit -m "feat: load official multi-label emotion songs"
```

### Task 2: Add deterministic group-level holdout and validation metrics

**Files:**
- Create: `competition_service/src/competition_emotion/splits.py`
- Create: `competition_service/src/competition_emotion/evaluate.py`
- Create: `competition_service/tests/test_splits.py`

- [ ] **Step 1: Write the failing split and metric tests**

```python
# competition_service/tests/test_splits.py
import unittest

import numpy as np

from competition_emotion.evaluate import metric_report
from competition_emotion.splits import make_holdout
from competition_emotion.types import Song


def song(song_id: str, labels: set[str]) -> Song:
    return Song(song_id, frozenset(labels), "n", "a", "g", "t", "https://a")


class SplitTests(unittest.TestCase):
    def test_holdout_never_splits_song_and_is_seeded(self) -> None:
        songs = [song(str(index), {"狂欢" if index % 2 else "孤独"}) for index in range(20)]
        first = make_holdout(songs, test_ratio=0.20, seed=7)
        second = make_holdout(songs, test_ratio=0.20, seed=7)
        self.assertEqual(first, second)
        self.assertFalse(first.train_ids.intersection(first.test_ids))
        self.assertEqual(len(first.test_ids), 4)

    def test_metric_report_has_strict_and_any_positive_top_one(self) -> None:
        labels = ("狂欢", "孤独")
        truth = np.array([[1, 0], [0, 1], [0, 1]])
        scores = np.array([[.8, .1], [.1, .9], [.3, .6]])
        report = metric_report(truth, scores, labels)
        self.assertEqual(report["strict_top1_accuracy"], 1.0)
        self.assertEqual(report["any_positive_top1_accuracy"], 1.0)
        self.assertEqual(report["macro_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_splits -v
```

Expected: `ModuleNotFoundError` for `competition_emotion.splits`.

- [ ] **Step 3: Implement a deterministic song grouping and score report**

```python
# competition_service/src/competition_emotion/splits.py
import random
from collections import Counter

from .types import Song, SplitAssignment


def make_holdout(songs: list[Song], test_ratio: float, seed: int) -> SplitAssignment:
    if not 0.05 <= test_ratio < 0.5:
        raise ValueError("test_ratio must be in [0.05, 0.5)")
    target_size = round(len(songs) * test_ratio)
    target_labels = Counter(label for song in songs for label in song.labels)
    target_labels = {label: count * test_ratio for label, count in target_labels.items()}
    shuffled = list(songs)
    random.Random(seed).shuffle(shuffled)
    randomized_rank = {song.song_id: index for index, song in enumerate(shuffled)}
    shuffled.sort(key=lambda song: (-len(song.labels), randomized_rank[song.song_id]))
    chosen: set[str] = set()
    chosen_labels: Counter[str] = Counter()
    for index, song in enumerate(shuffled):
        if len(chosen) >= target_size:
            break
        deficit = sum(max(0.0, target_labels[label] - chosen_labels[label]) for label in song.labels)
        remaining = target_size - len(chosen)
        remaining_candidates = len(shuffled) - index - 1
        if deficit > 0 or remaining > remaining_candidates:
            chosen.add(song.song_id)
            chosen_labels.update(song.labels)
    if len(chosen) < target_size:
        for song in shuffled:
            if song.song_id not in chosen:
                chosen.add(song.song_id)
                if len(chosen) == target_size:
                    break
    all_ids = frozenset(song.song_id for song in songs)
    return SplitAssignment(train_ids=all_ids.difference(chosen), test_ids=frozenset(chosen))
```

```python
# competition_service/src/competition_emotion/evaluate.py
import numpy as np
from sklearn.metrics import recall_score


def metric_report(y_true: np.ndarray, scores: np.ndarray, labels: tuple[str, ...]) -> dict[str, object]:
    if y_true.shape != scores.shape or y_true.shape[1] != len(labels):
        raise ValueError("truth, scores and labels have incompatible shapes")
    predicted = scores.argmax(axis=1)
    any_positive = y_true[np.arange(len(y_true)), predicted].astype(bool)
    singleton = y_true.sum(axis=1) == 1
    strict = float(any_positive[singleton].mean()) if singleton.any() else None
    strict_truth = y_true[singleton].argmax(axis=1)
    strict_predicted = predicted[singleton]
    per_label_recall = recall_score(
        strict_truth, strict_predicted, labels=list(range(len(labels))), average=None, zero_division=0,
    ) if singleton.any() else np.zeros(len(labels))
    return {
        "strict_top1_accuracy": strict,
        "any_positive_top1_accuracy": float(any_positive.mean()),
        "macro_recall": float(per_label_recall.mean()),
        "per_label_recall": {label: float(value) for label, value in zip(labels, per_label_recall)},
    }
```

- [ ] **Step 4: Run the focused tests and add a real-data split audit command**

Run the command from Step 2. Expected: `Ran 2 tests ... OK`.

Then run:

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -c "from pathlib import Path; from competition_emotion.data import load_official_songs; from competition_emotion.splits import make_holdout; s=load_official_songs(Path('F:\netease\_music\competition\data\emotion_songs_20260908_with_lrc.xlsx')); x=make_holdout(s,.15,20260910); print(len(s),len(x.train_ids),len(x.test_ids),len(x.train_ids & x.test_ids))"
```

Expected: 5,097 total songs, non-empty train/test IDs, and final overlap `0`.

- [ ] **Step 5: Commit the split and metric implementation**

```powershell
git add competition_service/src/competition_emotion/splits.py competition_service/src/competition_emotion/evaluate.py competition_service/tests/test_splits.py
git commit -m "feat: add group-safe validation metrics"
```

### Task 3: Train and persist a lyrics-only baseline

**Files:**
- Create: `competition_service/src/competition_emotion/models.py`
- Create: `competition_service/tests/test_models.py`

- [ ] **Step 1: Write the failing classifier test**

```python
# competition_service/tests/test_models.py
import unittest

from competition_emotion.models import TextScorer
from competition_emotion.types import Song


class TextScorerTests(unittest.TestCase):
    def test_scores_every_label_and_uses_available_text(self) -> None:
        songs = [
            Song("1", frozenset({"狂欢"}), "派对", "甲", "电子", "今晚一起跳舞", "https://a"),
            Song("2", frozenset({"孤独"}), "独处", "乙", "流行", "一个人没有你", "https://b"),
            Song("3", frozenset({"狂欢"}), "dance", "C", "电子", "dance all night", "https://c"),
            Song("4", frozenset({"孤独"}), "alone", "D", "流行", "alone in the rain", "https://d"),
        ]
        scorer = TextScorer.fit(songs, ("狂欢", "孤独"))
        scores = scorer.score("一个人孤独没有你")
        self.assertEqual(set(scores), {"狂欢", "孤独"})
        self.assertGreater(scores["孤独"], scores["狂欢"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_models.TextScorerTests -v
```

Expected: `ImportError: cannot import name 'TextScorer'`.

- [ ] **Step 3: Implement the calibrated multi-label text scorer**

```python
# competition_service/src/competition_emotion/models.py
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .types import Song


@dataclass
class TextScorer:
    labels: tuple[str, ...]
    vectorizer: TfidfVectorizer
    classifier: OneVsRestClassifier

    @classmethod
    def fit(cls, songs: list[Song], labels: tuple[str, ...]) -> "TextScorer":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=80000, sublinear_tf=True)
        matrix = vectorizer.fit_transform([song.text or song.name for song in songs])
        target = MultiLabelBinarizer(classes=labels).fit_transform([song.labels for song in songs])
        classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
        classifier.fit(matrix, target)
        return cls(labels=labels, vectorizer=vectorizer, classifier=classifier)

    def score(self, text: str) -> dict[str, float]:
        matrix = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(matrix)[0]
        return {label: float(value) for label, value in zip(self.labels, probabilities)}
```

- [ ] **Step 4: Run focused and full tests**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_models -v
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -s tests -v
```

Expected: all existing competition-service tests pass.

- [ ] **Step 5: Commit the baseline**

```powershell
git add competition_service/src/competition_emotion/models.py competition_service/tests/test_models.py
git commit -m "feat: add multilabel lyrics baseline"
```

### Task 4: Add bounded audio download, measured features and an audio baseline

**Files:**
- Create: `competition_service/src/competition_emotion/audio.py`
- Modify: `competition_service/src/competition_emotion/models.py`
- Create: `competition_service/tests/test_audio.py`

- [ ] **Step 1: Write failing feature and timeout tests**

```python
# competition_service/tests/test_audio.py
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from competition_emotion.audio import measured_features


class AudioTests(unittest.TestCase):
    def test_measured_features_are_finite_and_named(self) -> None:
        waveform = np.sin(np.linspace(0, 80, 22050, dtype=np.float32))
        features = measured_features(waveform, 22050)
        self.assertEqual(set(features), {"rms_db", "zero_crossing_rate", "spectral_centroid_hz", "dynamic_range_db"})
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_download_rejects_payload_larger_than_limit(self) -> None:
        from competition_emotion.audio import download_audio
        response = type("Response", (), {"iter_bytes": lambda self: iter([b"1234", b"5678"]), "raise_for_status": lambda self: None})()
        with patch("competition_emotion.audio.httpx.stream", return_value=type("Context", (), {"__enter__": lambda self: response, "__exit__": lambda *args: None})()):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, "audio payload exceeds"):
                    download_audio("https://audio", Path(temp) / "song.mp3", max_bytes=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to confirm the module failure**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_audio -v
```

Expected: `ModuleNotFoundError: No module named 'competition_emotion.audio'`.

- [ ] **Step 3: Implement bounded download and features**

```python
# competition_service/src/competition_emotion/audio.py
from pathlib import Path

import httpx
import numpy as np


def download_audio(url: str, destination: Path, max_bytes: int = 25_000_000, timeout_seconds: float = 12.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                written += len(chunk)
                if written > max_bytes:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise ValueError(f"audio payload exceeds {max_bytes} bytes")
                handle.write(chunk)
    return destination


def measured_features(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1 or len(samples) < 512 or sample_rate <= 0:
        raise ValueError("waveform must be mono with at least 512 samples")
    rms = float(np.sqrt(np.mean(samples ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    zero_crossing = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1])))
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    centroid = float((frequencies * spectrum).sum() / max(spectrum.sum(), 1e-12))
    frame = max(sample_rate // 2, 1)
    frame_rms = [np.sqrt(np.mean(samples[index:index + frame] ** 2)) for index in range(0, len(samples) - frame + 1, frame)]
    dynamic_range = 20.0 * np.log10(max(np.percentile(frame_rms, 95), 1e-12) / max(np.percentile(frame_rms, 5), 1e-12))
    return {"rms_db": rms_db, "zero_crossing_rate": zero_crossing, "spectral_centroid_hz": centroid, "dynamic_range_db": float(dynamic_range)}
```

Add `AudioScorer` in `models.py` using one `StandardScaler` plus one `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))`; its `score(features)` method must accept the four keys above in that exact order and return a score for every configured label.

- [ ] **Step 4: Run audio tests and record the missing decoder prerequisite**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_audio -v
Get-Command ffmpeg -ErrorAction SilentlyContinue | Select-Object Source,Version
```

Expected: audio unit tests pass. If `ffmpeg` is absent, record it in `README.md` as a deployment prerequisite; do not silently substitute an audio decoder.

- [ ] **Step 5: Commit the audio boundary**

```powershell
git add competition_service/src/competition_emotion/audio.py competition_service/src/competition_emotion/models.py competition_service/tests/test_audio.py
git commit -m "feat: add bounded audio feature extraction"
```

### Task 5: Train text/audio/fusion candidates and write reproducible reports

**Files:**
- Modify: `competition_service/src/competition_emotion/models.py`
- Modify: `competition_service/src/competition_emotion/evaluate.py`
- Create: `competition_service/src/competition_emotion/train.py`
- Create: `competition_service/tests/test_train.py`

- [ ] **Step 1: Write a failing fusion-ranking test**

```python
# competition_service/tests/test_train.py
import unittest

from competition_emotion.models import fuse_scores


class FusionTests(unittest.TestCase):
    def test_fusion_uses_only_available_modalities_and_ranks_one_label(self) -> None:
        result = fuse_scores(
            labels=("狂欢", "孤独"),
            text_scores={"狂欢": .2, "孤独": .9},
            audio_scores={"狂欢": .8, "孤独": .1},
            text_available=True,
            audio_available=False,
        )
        self.assertEqual(result["top_emotion"], "孤独")
        self.assertEqual(result["second_emotion"], "狂欢")
        self.assertLessEqual(result["second_confidence"], result["top_confidence"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify failure**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_train -v
```

Expected: `ImportError: cannot import name 'fuse_scores'`.

- [ ] **Step 3: Implement score fusion and the training CLI**

```python
# append to competition_service/src/competition_emotion/models.py
def fuse_scores(
    labels: tuple[str, ...], text_scores: dict[str, float], audio_scores: dict[str, float],
    text_available: bool, audio_available: bool,
) -> dict[str, object]:
    if not text_available and not audio_available:
        raise ValueError("at least one modality is required")
    weights = (0.65 if text_available else 0.0, 0.35 if audio_available else 0.0)
    denominator = sum(weights)
    ranking = sorted(
        ((label, (weights[0] * text_scores.get(label, 0.0) + weights[1] * audio_scores.get(label, 0.0)) / denominator) for label in labels),
        key=lambda item: (-item[1], item[0]),
    )
    top, second = ranking[:2]
    return {
        "top_emotion": top[0], "top_confidence": round(float(top[1]), 4),
        "second_emotion": second[0], "second_confidence": round(float(min(second[1], top[1])), 4),
        "scores": dict(ranking),
    }
```

```python
# competition_service/src/competition_emotion/train.py
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from .constants import LABELS, MODEL_VERSION
from .data import load_official_songs
from .evaluate import metric_report
from .models import TextScorer
from .splits import make_holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    songs = load_official_songs(args.workbook)
    split = make_holdout(songs, .15, args.seed)
    train = [song for song in songs if song.song_id in split.train_ids]
    test = [song for song in songs if song.song_id in split.test_ids]
    scorer = TextScorer.fit(train, LABELS)
    encoder = MultiLabelBinarizer(classes=LABELS).fit([LABELS])
    truth = encoder.transform([song.labels for song in test])
    scores = np.array([[scorer.score(song.text)[label] for label in LABELS] for song in test])
    report = metric_report(truth, scores, LABELS)
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": MODEL_VERSION, "labels": LABELS, "text_scorer": scorer}, args.bundle_dir / "model.joblib")
    (args.bundle_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

The initial CLI intentionally exports the text baseline only. Add audio and stacking fusion only after an offline feature cache covers the same train/test song IDs, and write each candidate report under a unique non-versioned `competition/runs/<timestamp>/` directory ignored by Git.

- [ ] **Step 4: Run the text baseline on the official workbook and inspect the report**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m competition_emotion.train --workbook 'F:\netease\_music\competition\data\emotion_songs_20260908_with_lrc.xlsx' --bundle-dir 'F:\netease\_music\competition\runs\text-baseline'
Get-Content -Raw 'F:\netease\_music\competition\runs\text-baseline\validation_report.json'
```

Expected: `model.joblib` and JSON report exist; report includes strict Top-1, any-positive Top-1, macro recall and all 15 per-label recalls. Do not state this result as a hidden-test prediction.

- [ ] **Step 5: Commit code only**

```powershell
git add competition_service/src/competition_emotion/models.py competition_service/src/competition_emotion/evaluate.py competition_service/src/competition_emotion/train.py competition_service/tests/test_train.py
git commit -m "feat: train and evaluate emotion baseline"
```

### Task 6: Implement the required HTTP schema, CPU runtime and truthful evidence

**Files:**
- Create: `competition_service/src/competition_emotion/runtime.py`
- Create: `competition_service/src/competition_emotion/service.py`
- Create: `competition_service/src/competition_emotion/main.py`
- Create: `competition_service/tests/test_service.py`

- [ ] **Step 1: Write failing API tests**

```python
# competition_service/tests/test_service.py
import unittest
from fastapi.testclient import TestClient

from competition_emotion.service import create_app


class FakeRuntime:
    def recognize(self, request: dict[str, object]) -> dict[str, object]:
        return {"top_emotion": "孤独", "top_confidence": .8123, "second_emotion": "悲伤", "second_confidence": .4000, "evidence": {"modalities": ["lyrics"], "lyric_matches": ["一个人"]}, "cost_ms": 9}


class ServiceTests(unittest.TestCase):
    def test_recognize_returns_required_json_shape(self) -> None:
        client = TestClient(create_app(FakeRuntime()))
        response = client.post("/api/v1/emotion/recognize", json={"audio_url": "https://a/song.mp3", "song_name": "歌", "song_id": "1", "text_lyric": "一个人"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["code"], 200)
        self.assertEqual(response.json()["data"]["top_emotion"], "孤独")

    def test_missing_audio_url_returns_json_400(self) -> None:
        client = TestClient(create_app(FakeRuntime()))
        response = client.post("/api/v1/emotion/recognize", json={"song_name": "歌", "song_id": "1"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], 400)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify the missing-service failure**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_service -v
```

Expected: `ModuleNotFoundError: No module named 'competition_emotion.service'`.

- [ ] **Step 3: Implement schema and response handling**

```python
# competition_service/src/competition_emotion/service.py
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class RecognizeRequest(BaseModel):
    audio_url: str = Field(min_length=1)
    song_name: str = Field(min_length=1)
    song_id: str = Field(min_length=1)
    album_name: str | None = None
    artists: str | None = None
    text_lyric: str | None = None
    lrc_lyric: str | None = None
    lrc_translation: str | None = None


def create_app(runtime: object) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: object, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"code": 400, "message": "invalid request", "details": exc.errors()})

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"code": 200, "data": {"status": "ok"}, "error": ""}

    @app.post("/api/v1/emotion/recognize")
    async def recognize(request: RecognizeRequest) -> JSONResponse:
        try:
            data = runtime.recognize(request.model_dump())
            return JSONResponse(status_code=200, content={"code": 200, "data": data, "error": ""})
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"code": 422, "message": str(exc)})
        except TimeoutError as exc:
            return JSONResponse(status_code=504, content={"code": 504, "message": str(exc)})
        except OSError as exc:
            return JSONResponse(status_code=502, content={"code": 502, "message": str(exc)})

    return app
```

`runtime.py` must load `model.joblib` once in its constructor, measure start/end with `time.perf_counter()`, clean LRC with `clean_lyric`, invoke bounded audio download/decode only when `audio_url` is supplied, and use `fuse_scores`. It must return `cost_ms` as measured elapsed milliseconds and evidence containing only `model_version`, available modalities, actual `measured_features` values, and actual configured lyric cues that occur in the request text.

```python
# competition_service/src/competition_emotion/main.py
import os
from pathlib import Path

from .runtime import EmotionRuntime
from .service import create_app

bundle = os.environ.get("MODEL_BUNDLE")
if not bundle:
    raise RuntimeError("MODEL_BUNDLE must point to a trained model directory")
app = create_app(EmotionRuntime(Path(bundle)))
```

- [ ] **Step 4: Run integration tests and local HTTP smoke test**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_service -v
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -s tests -v
```

Expected: the API tests return JSON for 200 and 400 cases; all tests pass.

- [ ] **Step 5: Commit the API boundary**

```powershell
git add competition_service/src/competition_emotion/runtime.py competition_service/src/competition_emotion/service.py competition_service/tests/test_service.py
git commit -m "feat: add internal emotion recognition API"
```

### Task 7: Add startup, health, load-test and delivery documentation

**Files:**
- Create: `competition_service/scripts/start-service.ps1`
- Create: `competition_service/scripts/restart-service.ps1`
- Create: `competition_service/scripts/health-check.ps1`
- Create: `competition_service/scripts/load_test.py`
- Create: `competition_service/README.md`
- Create: `competition_service/docs/cost-estimate.md`
- Create: `competition_service/docs/operations.md`

- [ ] **Step 1: Write a failing script-content test**

```python
# append to competition_service/tests/test_service.py
from pathlib import Path

    def test_start_script_binds_non_localhost_host(self) -> None:
        script = (Path(__file__).parents[1] / "scripts" / "start-service.ps1").read_text(encoding="utf-8")
        self.assertIn("--host $BindHost", script)
        self.assertNotIn("--host 127.0.0.1", script)
```

- [ ] **Step 2: Run the test to verify the missing-script failure**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest tests.test_service.ServiceTests.test_start_script_binds_non_localhost_host -v
```

Expected: `FileNotFoundError` for `start-service.ps1`.

- [ ] **Step 3: Add executable scripts and documents**

```powershell
# competition_service/scripts/start-service.ps1
param(
  [Parameter(Mandatory = $true)][string]$ModelBundle,
  [string]$BindHost = '0.0.0.0',
  [int]$Port = 8080
)
$root = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe'
$env:PYTHONPATH = Join-Path $root 'src'
$env:MODEL_BUNDLE = $ModelBundle
& $python -m uvicorn competition_emotion.main:app --host $BindHost --port $Port --workers 1
```

```powershell
# competition_service/scripts/health-check.ps1
param([string]$BaseUrl = 'http://127.0.0.1:8080')
Invoke-RestMethod -Uri "$BaseUrl/healthz" -Method Get | ConvertTo-Json -Compress
```

`restart-service.ps1` must stop only the process ID stored in `competition_service/run/service.pid`, wait for it to exit, remove the stale PID file, and invoke `start-service.ps1` with the same `ModelBundle`, `BindHost`, and `Port` arguments. It must never use a broad process-name kill.

`load_test.py` must accept `--base-url`, `--fixture-json`, `--qps`, and `--seconds`; issue POST requests at the configured rate; and write JSON containing request count, status-code counts, P50/P95/P99, and error rate. It must reject `qps < 1` and `seconds < 1`.

`README.md` must include Python dependency installation, `ffmpeg` verification, model-training command, non-localhost startup command, one request example, and exact output locations. `cost-estimate.md` must contain fields for P50/P99, CPU core-seconds, model size, 10-QPS replicas, per-call yuan and 10,000-calls monthly arithmetic. `operations.md` must include the request pipeline, deployment steps, log fields, health check, and these five fault cases: signed URL expired, audio download timeout, malformed audio, model bundle missing, and 10-QPS queue saturation.

- [ ] **Step 4: Run the complete test suite and verify scripts are parseable**

```powershell
$env:PYTHONPATH = 'F:\netease\_music\competition_service\src'
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -s tests -v
Get-Command ffmpeg -ErrorAction SilentlyContinue | Select-Object Source,Version
& 'C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe' 'F:\netease\_music\competition_service\scripts\load_test.py' --help
```

Expected: all tests pass; the load-test help lists all four arguments. If `ffmpeg` is absent, the command may show no output, but the README must list it as a required deployment dependency.

- [ ] **Step 5: Commit delivery material and push only after review**

```powershell
git add competition_service/scripts competition_service/README.md competition_service/docs competition_service/tests/test_service.py
git commit -m "docs: add emotion service operations materials"
git push origin main
```

Run the push only after reviewing `git status`, the staged file list, and the generated report for source-data or model-weight leakage.

## Plan self-review

- Spec coverage: official multi-label loading (Task 1), group-safe locked evaluation (Task 2), competing lyrics/audio/fusion baselines (Tasks 3–5), required API and truthful evidence (Task 6), and operating/cost delivery material (Task 7).
- Data protection: `.gitignore` excludes official workbooks, PMEmo, model assets and runs; Task 7 requires a leakage check before pushing.
- Boundary check: every online error path returns JSON rather than a fabricated label; every evidence field originates from current request text, measured features or loaded model metadata.
- Iteration boundary: the baseline is the first measurable version. A pretrained-encoder upgrade is considered only after this baseline has a song-group evaluation and CPU load report, as specified in the approved design.
