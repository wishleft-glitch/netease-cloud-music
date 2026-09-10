"""Command-line entry point for review-only PMEmo pilot candidates."""

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from .candidates import build_candidate
from .pmemo import load_pmemo_records
from .tag_cards import load_tag_cards


OUTPUT_FIELDS = (
    "music_id",
    "title",
    "artist",
    "tag",
    "valence",
    "arousal",
    "audio_evidence",
    "lyric_evidence",
    "evidence_state",
    "review_status",
    "is_final_label",
)


def write_outputs(
    candidates: list[dict[str, object]],
    songs_total: int,
    songs_with_lyrics: int,
    output_dir: Path,
) -> None:
    """Write candidate rows and their review-only pilot summary."""
    _validate_review_only_candidates(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_by_tag = Counter(str(candidate["tag"]) for candidate in candidates)
    summary = {
        "songs_total": songs_total,
        "songs_with_lyrics": songs_with_lyrics,
        "songs_missing_lyrics": songs_total - songs_with_lyrics,
        "candidate_rows": len(candidates),
        "candidates_by_tag": dict(candidates_by_tag),
        "final_labels_emitted": sum(candidate["is_final_label"] is True for candidate in candidates),
    }
    csv_target = output_dir / "pmemo_pilot_candidates.csv"
    summary_target = output_dir / "pmemo_pilot_summary.json"
    csv_temporary = _temporary_output_path(output_dir, "pmemo_pilot_candidates")
    summary_temporary: Path | None = None
    try:
        with csv_temporary.open("w", encoding="utf-8-sig", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(candidates)

        serialized_summary = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        summary_temporary = _temporary_output_path(output_dir, "pmemo_pilot_summary")
        summary_temporary.write_text(serialized_summary, encoding="utf-8")

        _publish_pair(csv_temporary, summary_temporary, csv_target, summary_target)
    except BaseException:
        _remove_temporary_file(csv_temporary)
        if summary_temporary is not None:
            _remove_temporary_file(summary_temporary)
        raise


def _validate_review_only_candidates(candidates: list[dict[str, object]]) -> None:
    for index, candidate in enumerate(candidates):
        if candidate.get("is_final_label") is not False:
            raise ValueError(f"Candidate {index} must have is_final_label exactly False")


def _temporary_output_path(output_dir: Path, label: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{label}.",
        suffix=".tmp",
        delete=False,
    ) as file_handle:
        return Path(file_handle.name)


def _publish_pair(
    csv_temporary: Path,
    summary_temporary: Path,
    csv_target: Path,
    summary_target: Path,
) -> None:
    targets = ((csv_target, "pmemo_pilot_candidates"), (summary_target, "pmemo_pilot_summary"))
    backups: dict[Path, Path] = {}
    created_backups: list[Path] = []
    published_targets: set[Path] = set()
    try:
        for target, label in targets:
            if target.exists():
                backup = _temporary_output_path(target.parent, f"{label}.backup")
                created_backups.append(backup)
                target.replace(backup)
                backups[target] = backup

        csv_temporary.replace(csv_target)
        published_targets.add(csv_target)
        summary_temporary.replace(summary_target)
        published_targets.add(summary_target)
    except BaseException:
        _restore_published_pair(backups, published_targets)
        raise
    finally:
        for backup in created_backups:
            _remove_temporary_file(backup)


def _restore_published_pair(backups: dict[Path, Path], published_targets: set[Path]) -> None:
    for target in published_targets:
        _remove_temporary_file(target)
    for target, backup in backups.items():
        backup.replace(target)


def _remove_temporary_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    """Build and write all review-only candidates for the supplied PMEmo data."""
    parser = argparse.ArgumentParser(description="Generate review-only PMEmo pilot candidates.")
    parser.add_argument("--pmemo-root", required=True, type=Path)
    parser.add_argument("--tag-cards", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    _preflight_input_paths(parser, args.pmemo_root, args.tag_cards, args.output_dir)
    try:
        records = load_pmemo_records(args.pmemo_root)
        cards = load_tag_cards(args.tag_cards)
    except (OSError, ValueError) as error:
        parser.error(f"could not load PMEmo inputs: {error}")
    candidates = [
        candidate
        for song in records
        for card in cards
        if (candidate := build_candidate(song, card)) is not None
    ]
    try:
        write_outputs(
            candidates,
            songs_total=len(records),
            songs_with_lyrics=sum(record.lyric_path is not None for record in records),
            output_dir=args.output_dir,
        )
    except OSError as error:
        parser.error(f"could not write --output-dir {args.output_dir}: {error}")


def _preflight_input_paths(
    parser: argparse.ArgumentParser,
    pmemo_root: Path,
    tag_cards: Path,
    output_dir: Path,
) -> None:
    if not pmemo_root.is_dir():
        parser.error(f"--pmemo-root is not a directory: {pmemo_root}")
    for relative_path in (Path("metadata.csv"), Path("annotations") / "static_annotations.csv"):
        expected_path = pmemo_root / relative_path
        if not expected_path.is_file():
            parser.error(f"--pmemo-root is missing required file: {expected_path}")
    if not tag_cards.is_file():
        parser.error(f"--tag-cards is not a file: {tag_cards}")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output-dir is not a directory: {output_dir}")


if __name__ == "__main__":
    main()
