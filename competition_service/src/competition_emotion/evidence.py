"""Safe, factual explanations for synchronous emotion predictions."""

from __future__ import annotations

from .lyrics import compose_lyrics


_LYRIC_EVIDENCE = "基于歌曲名称、艺人及可用歌词文本进行情绪判定。"
_METADATA_EVIDENCE = "仅基于歌曲名称和艺人元数据进行情绪判定。"


def build_evidence(
    *, text_lyric: object, lrc_lyric: object, lrc_translation: object
) -> str:
    """Describe only inputs that contributed without exposing lyric content."""
    if compose_lyrics(text_lyric, lrc_lyric, lrc_translation):
        return _LYRIC_EVIDENCE
    return _METADATA_EVIDENCE
