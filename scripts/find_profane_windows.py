import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector

json3_path = Path("benchmark/youtube/PZ2gyZC5Kuw.lt-orig.json3")
words = parse_youtube_json3(json3_path)
print(f"Total words in YouTube auto-captions: {len(words)}")

detector = ProfanityDetector()

profanities = []
for i, w in enumerate(words):
    if detector.is_profanity(w.word):
        profanities.append((i, w))

print(f"Total profanity detections in YouTube transcript: {len(profanities)}")

windows = []
for idx, (w_idx, w) in enumerate(profanities):
    start_t = max(0.0, w.start - 0.5)
    end_t = start_t + 5.5

    window_words = [x for x in words if start_t <= x.start <= end_t]
    window_profs = [x.word for x in window_words if detector.is_profanity(x.word)]

    windows.append({
        "start": start_t,
        "end": end_t,
        "count": len(window_profs),
        "profs": window_profs,
        "text": " ".join(x.word for x in window_words)
    })

windows.sort(key=lambda x: x["count"], reverse=True)

deduped = []
for win in windows:
    if all(abs(win["start"] - d["start"]) > 10.0 for d in deduped):
        deduped.append(win)

print(f"\nTop distinct 5-second profanity windows found:")
for i, d in enumerate(deduped[:10], 1):
    print(f"\nCandidate {i}:")
    print(f"  Time: {d['start']:.1f}s -> {d['end']:.1f}s ({d['start']/60:.2f} min)")
    print(f"  Profanities count: {d['count']} -> {d['profs']}")
    print(f"  Context: {d['text']}")
