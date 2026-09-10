"""Build transparent, review-only emotion-tag candidates."""

from pathlib import Path

from .models import SongRecord, TagCard


def _within(value: float | None, minimum: float | None, maximum: float | None) -> bool:
    """Return whether a value satisfies inclusive optional bounds."""
    if minimum is None and maximum is None:
        return True
    if value is None:
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _lyric_matches(path: Path | None, includes: tuple[str, ...], excludes: tuple[str, ...]) -> list[str]:
    """Return matched include terms unless an exclusion appears in the lyric."""
    if path is None:
        return []

    lyrics = _read_lyrics(path)
    if _contains_excluded_term(lyrics, excludes):
        return []
    return _included_terms(lyrics, includes)


def _read_lyrics(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").casefold()


def _contains_excluded_term(lyrics: str, excludes: tuple[str, ...]) -> bool:
    return any(term.casefold() in lyrics for term in excludes)


def _included_terms(lyrics: str, includes: tuple[str, ...]) -> list[str]:
    return [term for term in includes if term.casefold() in lyrics]


def build_candidate(song: SongRecord, card: TagCard) -> dict[str, object] | None:
    """Build a non-final candidate when the configured evidence requirements pass."""
    audio_rule_exists = any(
        bound is not None
        for bound in (
            card.min_valence,
            card.max_valence,
            card.min_arousal,
            card.max_arousal,
        )
    )
    lyric_rule_exists = bool(card.lyric_includes or card.lyric_excludes)
    if not audio_rule_exists and not card.lyric_includes:
        return None

    audio_passes = _within(song.valence, card.min_valence, card.max_valence) and _within(
        song.arousal,
        card.min_arousal,
        card.max_arousal,
    )
    if audio_rule_exists and not audio_passes:
        return None

    lyric_required = bool(card.lyric_includes)
    lyric_hits: list[str] = []
    if lyric_rule_exists:
        if song.lyric_path is None:
            return None
        try:
            lyrics = _read_lyrics(song.lyric_path)
        except OSError:
            return None
        if _contains_excluded_term(lyrics, card.lyric_excludes):
            return None
        lyric_hits = _included_terms(lyrics, card.lyric_includes)
        if lyric_required and not lyric_hits:
            return None

    evidence_state = (
        "audio_and_lyrics"
        if lyric_required and audio_rule_exists
        else "lyrics_only"
        if lyric_required
        else "audio_only"
    )
    return {
        "music_id": song.music_id,
        "title": song.title,
        "artist": song.artist,
        "tag": card.name,
        "valence": song.valence,
        "arousal": song.arousal,
        "audio_evidence": "within_configured_range" if audio_rule_exists else "no_audio_rule",
        "lyric_evidence": "|".join(lyric_hits),
        "evidence_state": evidence_state,
        "review_status": "needs_human_review" if card.requires_human_review else "candidate_only",
        "is_final_label": False,
    }
