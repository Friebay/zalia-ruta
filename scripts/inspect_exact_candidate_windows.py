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

detector = ProfanityDetector()

CANDIDATES = [
    ("6hIRVqm5kdo", 2055.0, 2085.0, "Candidate 1: debilai, debilams, debilam"),
    ("6hIRVqm5kdo", 7040.0, 7075.0, "Candidate 2: dalbajobu x3"),
    ("6hIRVqm5kdo", 7125.0, 7155.0, "Candidate 3: uzpiso, dalbajobu"),
    ("6hIRVqm5kdo", 4940.0, 4970.0, "Candidate 4: dalbajobai x2"),
    ("6hIRVqm5kdo", 3320.0, 3350.0, "Candidate 5: debilus x2"),
    ("6hIRVqm5kdo", 4755.0, 4785.0, "Candidate 6: debilu, debilus"),
    ("3_wAkycMk0c", 1730.0, 1760.0, "Candidate 7: prapisam, prapisa"),
    ("3_wAkycMk0c", 4215.0, 4245.0, "Candidate 8: ispiso x2"),
    ("XnkFuZ5cBB8", 2170.0, 2200.0, "Candidate 9: debilu, debila"),
    ("XnkFuZ5cBB8", 3105.0, 3135.0, "Candidate 10: debilu, debila"),
]

for vid, s, e, desc in CANDIDATES:
    p = Path(f"benchmark/youtube/{vid}.lt-orig.json3")
    words = parse_youtube_json3(p)
    in_range = [w for w in words if s <= w.start <= e]
    print(f"\n=======================================================")
    print(f"[{vid}] {desc} ({s}s - {e}s)")
    for w in in_range:
        prof_flag = " *** PROFANITY ***" if is_true_profanity(w.word, detector) else ""
        print(f"  {w.start:7.2f}s - {w.end:7.2f}s: {w.word}{prof_flag}")
