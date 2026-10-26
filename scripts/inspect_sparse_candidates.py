import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)

candidates = [
    ("karalius_silence_06", 2951.0, 2963.5, "galbūt būt nečia kažkokia senesnė"),
    ("karalius_silence_07", 576.5, 586.0, "28ais bus Seimo rinkimai. Jo,"),
    ("karalius_silence_08", 3699.5, 3706.0, "ir jaučiasi jo ne žinau,"),
    ("karalius_silence_09", 45.5, 53.5, "šitie jauni žmonės, čia jum"),
    ("karalius_silence_10", 681.0, 688.0, "Pala, kur mes tą žiūrėjom?"),
]

for cid, t_s, t_e, phrase in candidates:
    print(f"\n=== {cid} [{phrase}] Target: {t_s:.1f}s - {t_e:.1f}s ===")
    sub = [w for w in words if t_s - 3.0 <= w.start <= t_e + 3.0]
    for i, w in enumerate(sub):
        gap = (w.start - sub[i-1].end) if i > 0 else 0.0
        flag = " [INSIDE]" if t_s <= w.start <= t_e else " [OUTSIDE]"
        print(f"  {w.start:7.2f} - {w.end:7.2f} (gap: {gap:+5.2f}s): {w.word:<20}{flag}")
