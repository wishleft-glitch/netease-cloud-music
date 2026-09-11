"""Safe, factual explanations for synchronous emotion predictions."""

from __future__ import annotations

_LYRIC_EVIDENCE = "基于可用歌词文本进行情绪判定。"
_TITLE_EVIDENCE = "仅基于歌曲名称进行情绪判定。"
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
) -> str:
    """Describe the input facts without claiming audio changed the label."""
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
        return base
    if audio_state == "measured":
        return f"{base} 音频已完成测量，未用于当前标签判定。"
    if audio_state == "unavailable":
        return f"{base} 音频未参与当前判定。"
    raise ValueError("audio_state must be measured or unavailable")
