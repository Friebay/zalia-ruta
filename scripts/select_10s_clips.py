import sys
from pathlib import Path
from zalia_ruta.bench.srt_io import parse_srt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

srt_path = Path("output/Tapinas apie nesusitvarkymą su dronais, blogiausią finansų ministrą ir riaušes ｜ Karalius Reaguoja_dSObkwfw2MA/Tapinas apie nesusitvarkymą su dronais, blogiausią finansų ministrą ir riaušes ｜ Karalius Reaguoja_dSObkwfw2MA.srt")
cues = parse_srt(srt_path.read_text(encoding="utf-8"))
print(f"Total cues in transcript: {len(cues)}")

candidates = []
for i in range(len(cues)):
    for j in range(i + 5, min(i + 35, len(cues))):
        dur = cues[j].end - cues[i].start
        if 8.5 <= dur <= 11.5:
            gap_before = cues[i].start - (cues[i-1].end if i > 0 else 0.0)
            gap_after = (cues[j+1].start - cues[j].end) if j + 1 < len(cues) else 1.0
            if gap_before >= 0.40 and gap_after >= 0.40:
                words = [cues[k].text for k in range(i, j + 1)]
                candidates.append({
                    "start": cues[i].start,
                    "end": cues[j].end,
                    "dur": round(dur, 2),
                    "gap_before": round(gap_before, 2),
                    "gap_after": round(gap_after, 2),
                    "word_count": len(words),
                    "preview": " ".join(words)
                })

print(f"Found {len(candidates)} candidate windows.")

target_times = [250.0, 600.0, 1050.0, 1500.0, 2100.0]
selected = []

for target in target_times:
    best = None
    best_dist = 999999
    for c in candidates:
        if all(abs(c["start"] - s["start"]) > 90 for s in selected):
            dist = abs(c["start"] - target)
            if dist < best_dist:
                best = c
                best_dist = dist
    if best:
        selected.append(best)

for idx, s in enumerate(selected, 1):
    print(f"\nClip {idx}:")
    print(f"  Start: {s['start']:.2f}s, End: {s['end']:.2f}s, Dur: {s['dur']}s")
    print(f"  Silence margin: before {s['gap_before']}s, after {s['gap_after']}s")
    print(f"  Words ({s['word_count']}): {s['preview']}")
