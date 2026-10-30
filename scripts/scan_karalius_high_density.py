import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector, clean_token

VIDEO_IDS = [
    "XnkFuZ5cBB8",
    "PetdMzrPg_M",
    "x1a0uu8-J2M",
    "3_wAkycMk0c",
    "6hIRVqm5kdo",
]

def scan_video(video_id: str, window_size: float = 20.0, step: float = 1.0, min_dist: float = 25.0):
    json3_path = Path(f"benchmark/youtube/{video_id}.lt-orig.json3")
    if not json3_path.exists():
        print(f"File not found: {json3_path}")
        return []

    words = parse_youtube_json3(json3_path)
    if not words:
        print(f"No words parsed for {video_id}")
        return []

    detector = ProfanityDetector()

    # Identify profanity status for each word
    word_info = []
    for w in words:
        is_prof = detector.is_profanity(w.word)
        word_info.append({
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "is_prof": is_prof,
            "clean": clean_token(w.word),
        })

    total_profs = sum(1 for w in word_info if w["is_prof"])
    print(f"\n==========================================")
    print(f"Video {video_id}: {len(words)} words, {total_profs} detector hits")

    max_t = words[-1].start
    t_start = 0.0

    raw_windows = []
    while t_start + window_size <= max_t + 1.0:
        t_end = t_start + window_size
        in_win = [w for w in word_info if t_start <= w["start"] < t_end]
        profs_in_win = [w for w in in_win if w["is_prof"]]
        count = len(profs_in_win)

        if count > 0:
            raw_windows.append({
                "video_id": video_id,
                "start": round(t_start, 2),
                "end": round(t_end, 2),
                "count": count,
                "profs": [w["word"] for w in profs_in_win],
                "profs_clean": [w["clean"] for w in profs_in_win],
                "text": " ".join(w["word"] for w in in_win),
            })
        t_start += step

    raw_windows.sort(key=lambda x: x["count"], reverse=True)

    max_count = raw_windows[0]["count"] if raw_windows else 0
    print(f"Max profanities in any 20s window: {max_count}")

    # Windows with >= 5
    qualifying = [w for w in raw_windows if w["count"] >= 5]
    print(f"Windows with >= 5 profanities: {len(qualifying)}")

    # Deduplicate overlapping windows (minimum distance between starts: >= min_dist)
    deduped = []
    for w in raw_windows:
        if all(abs(w["start"] - d["start"]) >= min_dist for d in deduped):
            deduped.append(w)

    print(f"Top 5 distinct windows (by max profanities):")
    for i, d in enumerate(deduped[:5], 1):
        print(f"  #{i} [{d['start']}s - {d['end']}s] Count: {d['count']} | Profanities: {d['profs']}")
        print(f"      Text: {d['text'][:120]}...")

    return deduped

if __name__ == "__main__":
    all_results = {}
    for vid in VIDEO_IDS:
        all_results[vid] = scan_video(vid)
