import sys
from pathlib import Path
import json

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

print("--- CT0GB0vd1RI (vaziuok nachuj is cia) ---")
p1 = Path("benchmark/youtube/CT0GB0vd1RI.lt-orig.json3")
if p1.exists():
    words1 = parse_youtube_json3(p1)
    print(f"Total words: {len(words1)}")
    for w in words1:
        print(f"[{w.start:.2f}s - {w.end:.2f}s] {w.word}")
else:
    print("CT0GB0vd1RI.lt-orig.json3 not found")

print("\n--- UBisblxMd5w (Maslobojevas VS Rimkenzo) ---")
p2 = Path("benchmark/youtube/UBisblxMd5w.lt-orig.json3")
if p2.exists():
    words2 = parse_youtube_json3(p2)
    print(f"Total words: {len(words2)}")
    for i, w in enumerate(words2):
        if w.start >= 65.0: # from ~1:05 onwards
            print(f"[{w.start:.2f}s - {w.end:.2f}s] {w.word}")
else:
    print("UBisblxMd5w.lt-orig.json3 not found")
