from pathlib import Path
from typing import List, Optional

from zalia_ruta.bench.srt_io import Cue, parse_srt


def validate_cues(cues: List[Cue], max_duration: Optional[float] = None) -> List[str]:
    """
    Validates a list of word cues.
    Returns a list of error descriptions (empty if valid).
    """
    errors: List[str] = []
    if not cues:
        return ["File contains no cues."]

    prev_start = -1.0
    prev_end = -1.0

    for idx, cue in enumerate(cues, start=1):
        # 1. Empty text
        text = cue.text.strip()
        if not text:
            errors.append(f"Cue #{idx} (time {cue.start:.3f}s -> {cue.end:.3f}s): text is empty.")

        # 2. Unresolved disagreement marker
        if "|" in text:
            errors.append(
                f"Cue #{idx} (time {cue.start:.3f}s): unresolved disagreement marker '|' in '{text}'."
            )

        # 3. Invalid time bounds
        if cue.start < 0:
            errors.append(f"Cue #{idx} ('{text}'): start time is negative ({cue.start:.3f}s).")

        if cue.start >= cue.end:
            errors.append(
                f"Cue #{idx} ('{text}'): start ({cue.start:.3f}s) >= end ({cue.end:.3f}s)."
            )

        # 4. Monotonic ordering & overlap check
        if cue.start < prev_start:
            errors.append(
                f"Cue #{idx} ('{text}'): start time {cue.start:.3f}s is earlier than previous cue start {prev_start:.3f}s."
            )

        # Allow small floating point tolerance (5ms) for adjacent cues
        if cue.start < prev_end - 0.005:
            errors.append(
                f"Cue #{idx} ('{text}'): overlaps previous cue (start {cue.start:.3f}s < previous end {prev_end:.3f}s)."
            )

        # 5. Beyond clip duration
        if max_duration is not None and cue.end > max_duration + 0.5:
            errors.append(
                f"Cue #{idx} ('{text}'): end time {cue.end:.3f}s exceeds clip duration {max_duration:.2f}s."
            )

        prev_start = cue.start
        prev_end = cue.end

    return errors


def validate_reference_file(srt_path: Path, max_duration: Optional[float] = None) -> List[str]:
    """Reads and validates an SRT reference file."""
    p = Path(srt_path)
    if not p.exists():
        return [f"Reference file not found: {p}"]

    try:
        content = p.read_text(encoding="utf-8")
        cues = parse_srt(content)
        return validate_cues(cues, max_duration=max_duration)
    except Exception as e:
        return [f"Failed to parse SRT file {p}: {e}"]
