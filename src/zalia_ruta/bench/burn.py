import subprocess
from pathlib import Path
from typing import List, Optional

from zalia_ruta.asr import WordTimestamp
from zalia_ruta.bench.srt_io import Cue


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def generate_comparison_ass(
    refined_words: List[WordTimestamp],
    acoustic_words: List[WordTimestamp],
    ref_cues: Optional[List[Cue]] = None,
) -> str:
    """
    Generates ASS subtitle script content with distinctly styled and positioned tracks:
    - Top (Yellow): paprika_refined (MMS Forced Alignment baseline)
    - Bottom (Cyan): paprika_acoustic (Acoustic Energy Refined)
    - Above Bottom (Green): Ground Truth human reference (if provided)
    """
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Top: Yellow with black outline (Alignment 8 = Top Center)
        "Style: TopRefined,Arial,36,&H0000E5FF,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3,2,8,20,20,50,1",
        # Bottom: Cyan with black outline (Alignment 2 = Bottom Center)
        "Style: BottomAcoustic,Arial,36,&H00FFFF00,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3,2,2,20,20,50,1",
        # Reference (if present): Green with black outline, stacked slightly above Bottom (MarginV 110)
        "Style: GroundTruth,Arial,28,&H0055FF55,&H00000000,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,110,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    # 1. TopRefined events
    for w in refined_words:
        s = format_ass_time(w.start)
        e = format_ass_time(w.end)
        lines.append(f"Dialogue: 0,{s},{e},TopRefined,,0,0,0,,{{\\b1}}[MMS Refined]{{\\b0}}  {w.word}")

    # 2. BottomAcoustic events
    for w in acoustic_words:
        s = format_ass_time(w.start)
        e = format_ass_time(w.end)
        lines.append(f"Dialogue: 0,{s},{e},BottomAcoustic,,0,0,0,,{{\\b1}}[Acoustic]{{\\b0}}  {w.word}")

    # 3. Ground Truth events (if present)
    if ref_cues:
        for cue in ref_cues:
            s = format_ass_time(cue.start)
            e = format_ass_time(cue.end)
            lines.append(f"Dialogue: 0,{s},{e},GroundTruth,,0,0,0,,{{\\b1}}[Etalonas]{{\\b0}}  {cue.text}")

    return "\n".join(lines) + "\n"


def burn_comparison_video(
    video_path: Path,
    output_path: Path,
    refined_words: List[WordTimestamp],
    acoustic_words: List[WordTimestamp],
    ref_cues: Optional[List[Cue]] = None,
) -> None:
    """Burns side-by-side comparative subtitles onto video using FFmpeg and libass."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ass_content = generate_comparison_ass(refined_words, acoustic_words, ref_cues)
    temp_ass = output_path.with_suffix(".temp.ass")
    temp_ass.write_text(ass_content, encoding="utf-8")

    try:
        import os
        ass_filter_path = os.path.relpath(temp_ass.resolve()).replace("\\", "/")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"ass={ass_filter_path}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(output_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg burning failed (code {res.returncode}):\n{res.stderr[-1000:]}")
    finally:
        if temp_ass.exists():
            temp_ass.unlink()
