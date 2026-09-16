from __future__ import annotations

from collections import Counter
from collections import defaultdict
import random

from .types import Song, SplitAssignment


def make_holdout(
    songs: list[Song], test_ratio: float, seed: int
) -> SplitAssignment:
    """Create a deterministic song-level holdout with greedy label coverage."""
    if not 0.05 <= test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0.05 (inclusive) and 0.5 (exclusive)")
    if not songs:
        raise ValueError("songs cannot be empty")

    song_ids = [song.song_id for song in songs]
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("songs must have unique song IDs")

    test_size = round(len(songs) * test_ratio)
    candidates = sorted(songs, key=lambda song: song.song_id)
    random.Random(seed).shuffle(candidates)

    total_by_label = Counter(label for song in songs for label in song.labels)
    target_by_label = {
        label: count * test_ratio for label, count in total_by_label.items()
    }
    selected_by_label: Counter[str] = Counter()
    test_ids: set[str] = set()

    for _ in range(test_size):
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(candidates):
            score = sum(
                max(target_by_label[label] - selected_by_label[label], 0.0)
                / target_by_label[label]
                for label in candidate.labels
            )
            if score > best_score:
                best_index = index
                best_score = score

        selected = candidates.pop(best_index)
        test_ids.add(selected.song_id)
        selected_by_label.update(selected.labels)

    all_ids = frozenset(song_ids)
    return SplitAssignment(
        train_ids=all_ids - test_ids,
        test_ids=frozenset(test_ids),
    )


def make_stratified_holdout(
    songs: list[Song], test_ratio: float, seed: int
) -> SplitAssignment:
    """Split by song, preserving singleton labels and multi-label cardinality.

    This is intended for future development splits. A fresh split of already
    inspected labels is not an untouched test set.
    """
    if not 0.05 <= test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0.05 (inclusive) and 0.5 (exclusive)")
    if not songs:
        raise ValueError("songs cannot be empty")
    song_ids = [song.song_id for song in songs]
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("songs must have unique song IDs")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for song in songs:
        if len(song.labels) == 1:
            key = ("single", next(iter(song.labels)))
        else:
            key = ("multi", str(len(song.labels)))
        strata[key].append(song.song_id)

    target = round(len(songs) * test_ratio)
    quotas = {key: len(ids) * test_ratio for key, ids in strata.items()}
    # Keep at least one example of every stratum in fit, especially a label
    # with only one annotated song.
    capacities = {key: len(ids) - 1 for key, ids in strata.items()}
    counts = {key: min(int(quota), capacities[key]) for key, quota in quotas.items()}
    remainder = target - sum(counts.values())
    while remainder:
        eligible = [key for key in strata if counts[key] < capacities[key]]
        if not eligible:
            raise ValueError("not enough songs to keep each stratum represented in fit")
        key = min(eligible, key=lambda item: (-(quotas[item] - counts[item]), item))
        counts[key] += 1
        remainder -= 1

    rng = random.Random(seed)
    test_ids: set[str] = set()
    for key in sorted(strata):
        candidates = sorted(strata[key])
        rng.shuffle(candidates)
        test_ids.update(candidates[:counts[key]])
    all_ids = frozenset(song_ids)
    return SplitAssignment(all_ids - test_ids, frozenset(test_ids))
