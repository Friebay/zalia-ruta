import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from zalia_ruta.subtitles import format_srt_timestamp


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def parse_srt_timestamp(ts_str: str) -> float:
    """Parses 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' into seconds (float)."""
    match = _TIME_RE.match(ts_str.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp format: '{ts_str}'")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(content: str) -> List[Cue]:
    """
    Parses an SRT string into a list of Cue objects.
    Robust to Windows/Unix line endings and multi-line cue text.
    """
    lines = [line.strip() for line in content.splitlines()]
    cues: List[Cue] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line:
            i += 1
            continue

        # Check if line is an integer cue index
        try:
            cue_index = int(line)
            i += 1
            if i >= n:
                break
            time_line = lines[i]
        except ValueError:
            # Fallback if index line was omitted
            cue_index = len(cues) + 1
            time_line = line

        if "-->" not in time_line:
            i += 1
            continue

        parts = time_line.split("-->")
        if len(parts) != 2:
            i += 1
            continue

        try:
            start_sec = parse_srt_timestamp(parts[0])
            end_sec = parse_srt_timestamp(parts[1])
        except ValueError:
            i += 1
            continue

        i += 1
        text_lines: List[str] = []
        while i < n and lines[i]:
            text_lines.append(lines[i])
            i += 1

        text = " ".join(text_lines).strip()
        cues.append(Cue(index=cue_index, start=start_sec, end=end_sec, text=text))

    return cues


def write_srt(cues: List[Cue], file_path: Optional[str | Path] = None) -> str:
    """Formats cues into standard SRT string and optionally writes to file."""
    blocks: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        s_str = format_srt_timestamp(cue.start)
        e_str = format_srt_timestamp(cue.end)
        blocks.append(f"{idx}\n{s_str} --> {e_str}\n{cue.text}\n")

    content = "\n".join(blocks)
    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    return content
