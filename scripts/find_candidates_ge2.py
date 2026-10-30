import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector, clean_token

FALSE_POSITIVE_STEMS = {
    "naci", "nacist", "nacional", "epistem", "problem",
    "pistol", "tipisk", "prisikur", "sukel", "sukaust", "atsuk", "blacklist", "blackk", "sukyt"
}

def is_true_profanity(word: str, detector: ProfanityDetector) -> bool:
    if not detector.is_profanity(word):
        return False
    cl = clean_token(word)
    for fp in FALSE_POSITIVE_STEMS:
        if fp in cl:
            return False
    return True

VIDEO_IDS = [
    "XnkFuZ5cBB8",
    "PetdMzrPg_M",
    "x1a0uu8-J2M",
    "3_wAkycMk0c",
    "6hIRVqm5kdo",
]

detector = ProfanityDetector()
all_candidates = []

for vid in VIDEO_IDS:
    p = Path(f"benchmark/youtube/{vid}.lt-orig.json3")
    words = parse_youtube_json3(p)
    max_t = words[-1].start
    t = 0.0
    windows = []
    while t + 20.0 <= max_t + 1.0:
        win_words = [w for w in words if t <= w.start < t + 20.0]
        win_profs = [w for w in win_words if is_true_profanity(w.word, detector)]
        c = len(win_profs)
        if c >= 2:
            windows.append({
                "video_id": vid,
                "start": round(t, 2),
                "end": round(t + 20.0, 2),
                "count": c,
                "profs": [w.word for w in win_profs],
                "text": " ".join(w.word for w in win_words)
            })
        t += 1.0
        
    windows.sort(key=lambda x: x["count"], reverse=True)
    deduped = []
    for w in windows:
        if all(abs(w["start"] - d["start"]) >= 25.0 for d in deduped):
            deduped.append(w)
    all_candidates.extend(deduped)

all_candidates.sort(key=lambda x: (x["count"], x["video_id"]), reverse=True)
print(f"Total deduplicated candidates with >= 2 profanities: {len(all_candidates)}")
for i, c in enumerate(all_candidates, 1):
    print(f"#{i} [{c['video_id']}] {c['start']}s - {c['end']}s | Count: {c['count']} | {c['profs']}")
    print(f"   Text: {c['text'][:120]}...")
