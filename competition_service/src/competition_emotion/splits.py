from __future__ import annotations

from collections import Counter
import random

from .types import Song, SplitAssignment


def make_holdout(
    songs: list[Song], test_ratio: float, seed: int
) -> SplitAssignment:
    """Create a deterministic song-level holdout with greedy label coverage."""
    if not 0.05 <= test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0.05 (inclusive) and 0.5 (exclusive)")

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
