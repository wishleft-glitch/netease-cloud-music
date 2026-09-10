"""Safe, factual explanations for synchronous emotion predictions."""

from __future__ import annotations

_LYRIC_EVIDENCE = "基于可用歌词文本进行情绪判定。"
_TITLE_EVIDENCE = "仅基于歌曲名称进行情绪判定。"


def build_evidence(
    *, lyric_used: bool, title_used: bool
) -> str:
    """Describe the single input actually sent to the text scorer."""
    if lyric_used and not title_used:
        return _LYRIC_EVIDENCE
    if title_used and not lyric_used:
        return _TITLE_EVIDENCE
    raise ValueError("evidence requires exactly one scored input")
