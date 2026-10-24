import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from zalia_ruta.asr import WordTimestamp


def fetch_youtube_captions(
    youtube_id: str,
    output_path: Path,
) -> Optional[Path]:
    """
    Downloads YouTube automatic captions in lt-orig json3 format using yt-dlp.
    Returns output_path if successful, None otherwise.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_stem = output_path.parent / f"tmp_{youtube_id}"

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs",
        "lt-orig",
        "--sub-format",
        "json3",
        "-o",
        str(temp_stem),
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        expected_file = output_path.parent / f"tmp_{youtube_id}.lt-orig.json3"
        if expected_file.exists():
            if output_path.exists():
                output_path.unlink()
            expected_file.rename(output_path)
            return output_path
        return None
    except Exception as e:
        print(f"[YouTube] Error fetching captions for {youtube_id}: {e}")
        return None


def parse_youtube_json3(json_data: str | Dict[str, Any] | Path) -> List[WordTimestamp]:
    """
    Parses YouTube json3 format to extract individual words and start timestamps.
    Note: YouTube json3 does not provide reliable word end times, only start offsets.
    """
    if isinstance(json_data, (str, Path)):
        p = Path(json_data)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(str(json_data))
    else:
        data = json_data

    words: List[WordTimestamp] = []
    events = data.get("events", [])

    for ev in events:
        event_start_ms = ev.get("tStartMs", 0)
        segs = ev.get("segs", [])
        if not segs:
            continue

        for seg in segs:
            text = seg.get("utf8", "").strip()
            if not text or text == "\n":
                continue

            offset_ms = seg.get("tOffsetMs", 0)
            start_sec = round((event_start_ms + offset_ms) / 1000.0, 3)

            # Some segs might contain multiple whitespace-separated words
            sub_words = text.split()
            for sw in sub_words:
                sw_clean = sw.strip()
                if sw_clean:
                    # Default estimated end is start + 0.25s since YouTube json3 has no word end times
                    words.append(
                        WordTimestamp(
                            word=sw_clean,
                            start=start_sec,
                            end=round(start_sec + 0.25, 3),
                        )
                    )

    return words
