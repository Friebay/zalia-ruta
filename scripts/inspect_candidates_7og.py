import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)

candidates = [
    # (id, approx_start, approx_end, key_phrase)
    ("karalius_prof_07", 4278.0, 4285.0, "What the fuck? Kodėl?"),
    ("karalius_prof_08", 3691.5, 3698.5, "kaip užpisa, kad tiek heiterių turi"),
    ("karalius_prof_09", 4125.2, 4132.2, "Žodžiu, fucking hell. Ten aš nežinau"),
    ("karalius_prof_10", 4003.8, 4010.8, "visiškai su šūdais tave maišys"),
    ("karalius_prof_11", 4202.7, 4209.7, "išrinks kokį debilą, kuris žadės"),
]

for cid, c_s, c_e, phrase in candidates:
    print(f"\n=== {cid} [{phrase}] Target: {c_s:.1f}s - {c_e:.1f}s ===")
    surrounding = [w for w in words if c_s - 3.0 <= w.start <= c_e + 3.0]
    for w in surrounding:
        flag = " [INSIDE]" if c_s <= w.start <= c_e else " [OUTSIDE]"
        print(f"  {w.start:7.2f} - {w.end:7.2f}: {w.word:<20}{flag}")
