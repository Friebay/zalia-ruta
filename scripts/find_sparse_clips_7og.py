import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)
print(f"Total words: {len(words)}")

# Consecutive word gaps
gaps = []
for i in range(1, len(words)):
    gap = words[i].start - words[i-1].end
    gaps.append((gap, i-1, i))

gaps.sort(reverse=True, key=lambda x: x[0])
print("\n--- Top 15 longest consecutive silence gaps ---")
for g, i1, i2 in gaps[:15]:
    m, s = divmod(int(words[i1].end), 60)
    print(f"Gap {g:5.2f}s [{m:02d}:{s:02d}] between '{words[i1].word}' ({words[i1].end:.2f}s) and '{words[i2].word}' ({words[i2].start:.2f}s)")
