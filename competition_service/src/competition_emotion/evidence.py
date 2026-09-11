"""Safe, factual explanations for synchronous emotion predictions."""

from __future__ import annotations

_LYRIC_EVIDENCE = "基于可用歌词文本进行情绪判定。"
_TITLE_EVIDENCE = "仅基于歌曲名称进行情绪判定。"
_LYRIC_METADATA_EVIDENCE = "基于歌曲名称、艺人、专辑/风格元数据与可用歌词进行情绪判定。"
_TITLE_METADATA_EVIDENCE = "基于歌曲名称、艺人、专辑/风格元数据进行情绪判定。"


def build_evidence(
    *,
    lyric_used: bool,
    title_used: bool,
    audio_state: str | None = None,
    metadata_used: bool = False,
) -> str:
    """Describe the input facts without claiming audio changed the label."""
    if lyric_used and not title_used:
        base = _LYRIC_METADATA_EVIDENCE if metadata_used else _LYRIC_EVIDENCE
    elif title_used and not lyric_used:
        base = _TITLE_METADATA_EVIDENCE if metadata_used else _TITLE_EVIDENCE
    else:
        raise ValueError("evidence requires exactly one scored input")
    if audio_state is None:
        return base
    if audio_state == "measured":
        return f"{base} 音频已完成测量，未用于当前标签判定。"
    if audio_state == "unavailable":
        return f"{base} 音频未参与当前判定。"
    raise ValueError("audio_state must be measured or unavailable")
