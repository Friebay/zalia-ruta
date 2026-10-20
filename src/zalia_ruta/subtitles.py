from pathlib import Path
from typing import List, Optional

from zalia_ruta.asr import WordTimestamp
from zalia_ruta.profanity import BleepInterval, clean_token


def format_srt_timestamp(seconds: float) -> str:
    """Formats float seconds into SRT timestamp format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60

    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(
    words: List[WordTimestamp],
    profanity_detector: Optional[object] = None,
    output_srt_path: Optional[str | Path] = None,
    mark_bleeps: bool = True,
) -> str:
    """
    Generates standard SRT subtitles with word-level timestamps.
    If mark_bleeps is True and a word matches profanity, it is tagged as [BLEEP: word].

    Args:
        words: List of WordTimestamp objects from ASR.
        profanity_detector: Optional ProfanityDetector instance to tag profane words.
        output_srt_path: Optional file path to save the .srt file.
        mark_bleeps: Whether to annotate bleeped words.

    Returns:
        Full SRT text content.
    """
    lines = []
    idx = 1

    for wt in words:
        start_str = format_srt_timestamp(wt.start)
        end_str = format_srt_timestamp(wt.end)

        is_profane = False
        if profanity_detector is not None and hasattr(profanity_detector, "is_profanity"):
            is_profane = profanity_detector.is_profanity(wt.word)

        text = wt.word
        if mark_bleeps and is_profane:
            text = f"[BLEEP: {clean_token(wt.word)}]"

        lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        idx += 1

    srt_content = "\n".join(lines)

    if output_srt_path is not None:
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

    return srt_content

