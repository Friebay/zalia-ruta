import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)

candidates = [
    ("karalius_prof_07", 4278.0, 4285.0),
]

for cid, c_s, c_e in candidates:
    print(f"\n==================== {cid} (target {c_s}s - {c_e}s) ====================")
    sub = [w for w in words if c_s - 4.0 <= w.start <= c_e + 4.0]
    for i in range(len(sub)):
        w = sub[i]
        gap_before = (w.start - sub[i-1].end) if i > 0 else 0.0
        print(f"  {w.start:7.2f} - {w.end:7.2f} (gap before: {gap_before:+5.2f}s): {w.word}")
