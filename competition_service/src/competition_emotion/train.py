"""Reproducible training entry point for the official lyrics baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from .constants import LABELS, MODEL_VERSION
from .data import load_official_songs, workbook_snapshot
from .evaluate import metric_report
from .models import TextScorer, save_text_scorer
from .splits import make_holdout
from .types import Song


REPORT_SCHEMA_VERSION = 2
MODEL_TYPE = "lyrics_tfidf_logreg"
BUNDLE_POINTER_SCHEMA_VERSION = 1


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_text(path: Path, content: str) -> None:
    """Write a fully materialized file into an unpublished staging directory."""
    with path.open("w", encoding="utf-8", newline="") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _save_model_atomically(scorer: TextScorer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        save_text_scorer(scorer, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _targets(songs: Sequence[Song], labels: tuple[str, ...]) -> np.ndarray:
    configured = set(labels)
    unknown = {label for song in songs for label in song.labels if label not in configured}
    if unknown:
        raise ValueError("songs contain labels outside the configuration: " + ", ".join(sorted(unknown)))
    return np.asarray(
        [[int(label in song.labels) for label in labels] for song in songs], dtype=np.int8
    )


def _text_for_score(song: Song) -> str:
    return song.text.strip() or song.name.strip() or "[no_text]"


def _label_support(songs: Sequence[Song], labels: tuple[str, ...]) -> dict[str, int]:
    return {label: sum(label in song.labels for song in songs) for label in labels}


def _prediction_records(
    songs: Sequence[Song], scores: np.ndarray, labels: tuple[str, ...]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for song, song_scores in zip(songs, scores, strict=True):
        ordered_indices = np.argsort(-song_scores, kind="stable")[: min(3, len(labels))]
        predicted_index = int(ordered_indices[0])
        records.append(
            {
                "song_id": song.song_id,
                "title": song.name,
                "artist": song.artists,
                # The official source schema does not supply a language column.
                "language": "",
                "predicted_label": labels[predicted_index],
                "confidence": float(song_scores[predicted_index]),
                "top_labels": [labels[int(index)] for index in ordered_indices],
                "top_scores": [float(song_scores[int(index)]) for index in ordered_indices],
            }
        )
    return records


def _publish_bundle(
    bundle_root: Path,
    scorer: TextScorer,
    report: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
) -> Path:
    """Publish a complete immutable bundle by atomically switching one pointer.

    A reader resolves ``current.json`` first.  It therefore sees either the
    complete previous bundle or the complete new version, never a mixture of
    model, report, and prediction files from two training runs.
    """
    bundle_root.mkdir(parents=True, exist_ok=True)
    versions_dir = bundle_root / "versions"
    versions_dir.mkdir(exist_ok=True)
    bundle_id = uuid4().hex
    staging_dir = bundle_root / f".staging-{bundle_id}"
    published_dir = versions_dir / bundle_id
    staging_dir.mkdir()
    try:
        _save_model_atomically(scorer, staging_dir / "model.joblib")
        _write_text(
            staging_dir / "report.json",
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _write_text(
            staging_dir / "predictions.jsonl",
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in predictions
            ),
        )
        os.replace(staging_dir, published_dir)
        pointer = {
            "pointer_schema_version": BUNDLE_POINTER_SCHEMA_VERSION,
            "active_bundle": f"versions/{bundle_id}",
        }
        _atomic_write_text(
            bundle_root / "current.json",
            json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    # Earlier development runs used a flat layout.  They are not valid in the
    # pointer-based contract, so retire them only after the new pointer exists.
    for legacy_name in ("model.joblib", "report.json", "predictions.jsonl"):
        legacy_path = bundle_root / legacy_name
        try:
            if legacy_path.is_file():
                legacy_path.unlink()
        except OSError:
            # Cleanup cannot make the newly published pointer inconsistent.
            pass
    return published_dir


def train_text_baseline(
    workbook: Path,
    bundle_dir: Path,
    *,
    labels: Sequence[str] = LABELS,
    seed: int = 20260910,
    test_ratio: float = 0.2,
) -> dict[str, Any]:
    """Train and evaluate the lyrics baseline with a song-group-safe split."""
    configured_labels = tuple(labels)
    source_workbook = Path(workbook)
    with workbook_snapshot(source_workbook) as (snapshot_path, source):
        songs = load_official_songs(snapshot_path)
    assignment = make_holdout(songs, test_ratio=test_ratio, seed=seed)
    train_songs = [song for song in songs if song.song_id in assignment.train_ids]
    test_songs = [song for song in songs if song.song_id in assignment.test_ids]
    overlap = set(assignment.train_ids) & set(assignment.test_ids)
    if overlap:
        raise ValueError("train and test song IDs overlap")
    if len(train_songs) != len(assignment.train_ids) or len(test_songs) != len(assignment.test_ids):
        raise ValueError("split song IDs do not reconcile with the loaded data")

    scorer = TextScorer().fit(train_songs, configured_labels)
    scores = scorer.score_many([_text_for_score(song) for song in test_songs])
    evaluation = metric_report(_targets(test_songs, configured_labels), scores, configured_labels)
    predictions = _prediction_records(test_songs, scores, configured_labels)

    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "source": source,
        "labels": list(configured_labels),
        "seed": seed,
        "test_ratio": test_ratio,
        "counts": {
            "total_songs": len(songs),
            "train_songs": len(train_songs),
            "test_songs": len(test_songs),
            "split_group_overlap": len(overlap),
        },
        "label_support": {
            "train": _label_support(train_songs, configured_labels),
            "test": _label_support(test_songs, configured_labels),
        },
        "evaluation_sample_counts": {
            "any_positive_sample_count": int(np.any(_targets(test_songs, configured_labels), axis=1).sum()),
            "strict_singleton_sample_count": int((np.sum(_targets(test_songs, configured_labels), axis=1) == 1).sum()),
            "multi_label_sample_count": int((np.sum(_targets(test_songs, configured_labels), axis=1) > 1).sum()),
        },
        "evaluation_metrics": evaluation,
    }

    _publish_bundle(Path(bundle_dir), scorer, report, predictions)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the official lyrics emotion baseline")
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260910, type=int)
    parser.add_argument("--test-ratio", default=0.2, type=float)
    arguments = parser.parse_args(argv)

    report = train_text_baseline(
        arguments.workbook,
        arguments.bundle_dir,
        seed=arguments.seed,
        test_ratio=arguments.test_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
