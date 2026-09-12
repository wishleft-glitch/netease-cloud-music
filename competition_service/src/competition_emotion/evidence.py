"""Safe, factual explanations for synchronous emotion predictions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rubric import Rubric

_LYRIC_EVIDENCE = "基于可用歌词文本进行情绪判定。"
_TITLE_EVIDENCE = "仅基于歌曲名称进行情绪判定。"


def _lyric_excerpt(lyric_text: str, max_chars: int = 36) -> str:
    """Return a short verbatim lyric fragment for human review."""
    for line in lyric_text.splitlines():
        candidate = re.sub(r"\[\d{1,3}:\d{2}(?:\.\d+)?\]", "", line)
        candidate = " ".join(candidate.split()).strip()
        if not candidate or candidate.startswith("[by:"):
            continue
        return candidate[:max_chars]
    return ""


def _metadata_phrase(metadata_fields: tuple[str, ...] | None) -> str:
    fields = metadata_fields or ("歌曲名称", "艺人", "专辑元数据")
    cleaned = tuple(dict.fromkeys(field.strip() for field in fields if field.strip()))
    return "、".join(cleaned)


def build_evidence(
    *,
    lyric_used: bool,
    title_used: bool,
    audio_state: str | None = None,
    metadata_used: bool = False,
    metadata_fields: tuple[str, ...] | None = None,
    lyric_text: str | None = None,
    label: str | None = None,
    rubric: "Rubric | None" = None,
) -> str:
    """Describe input facts and optionally attach one verified lyric cue.

    The optional cue is quoted only when it occurs verbatim in the supplied
    lyrics and is present in the versioned rubric.  This keeps evidence
    auditable and avoids inventing lyric text in an explanation.
    """
    if lyric_used and not title_used:
        base = (
            f"基于{_metadata_phrase(metadata_fields)}与可用歌词进行情绪判定。"
            if metadata_used
            else _LYRIC_EVIDENCE
        )
    elif title_used and not lyric_used:
        base = (
            f"基于{_metadata_phrase(metadata_fields)}进行情绪判定。"
            if metadata_used
            else _TITLE_EVIDENCE
        )
    else:
        raise ValueError("evidence requires exactly one scored input")
    if audio_state is None:
        evidence = base
    elif audio_state == "measured":
        evidence = f"{base} 音频已完成测量，未用于当前标签判定。"
    elif audio_state == "unavailable":
        evidence = f"{base} 音频未参与当前判定。"
    else:
        raise ValueError("audio_state must be measured or unavailable")

    if lyric_text and label and rubric is not None:
        entries = [entry for entry in rubric.entries if entry.label == label]
        if len(entries) != 1:
            raise ValueError("unknown rubric label: " + str(label))
        # Prefer the most specific cue so a short generic token cannot mask a
        # longer, more useful contiguous quote from the lyrics.
        matched_cue: str | None = None
        for cue in sorted(entries[0].positive_cues, key=len, reverse=True):
            if cue in lyric_text:
                matched_cue = cue
                break
        if matched_cue is not None:
            evidence = f'{evidence} 歌词证据：“{matched_cue}”（命中Rubric正向线索）。'
        else:
            excerpt = _lyric_excerpt(lyric_text)
            if excerpt:
                evidence = f'{evidence} 输入歌词片段：“{excerpt}”。'
        evidence = f'{evidence} 标签定义：“{entries[0].definition}”。'
    return evidence[:500]
