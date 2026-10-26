import sys
from pathlib import Path
import re

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)
print(f"Total words parsed in 7og_55op5Kg: {len(words)}")

detector = ProfanityDetector()

# True profanity roots
SWEAR_WORDS = [
    # blet, ble, bliad
    r"^ble+t?$", r"^bliad.*$",
    # nachui, pochui
    r"^(na|po|za)chui.*$", r"^(na|po|za)huj.*$", r"^nach$",
    # kurva
    r"^kurv.*$",
    # pist, piso, pisa, užpisa, supistas, prisipist
    r"^.*(pis[tkaoeųū]|pys[tkaoeųū]|pyzd|pizd|chuj|huj).*$",
    # dalbajob, dolbajob, joban, jeban
    r"^.*(dalba|dolba|joba|jeba).*$",
    # debil
    r"^debil.*$",
    # šūdas (not sudaryti)
    r"^šūd.*$", r"^sud[oąųe]$",
    # English
    r"^fuck.*$", r"^shit.*$",
]
SWEAR_RES = [re.compile(p, re.IGNORECASE) for p in SWEAR_WORDS]

EXCLUDE = {"sudaryti", "sudaryt", "sudaro", "sudarys", "sudarai", "sudarymo", "nacionalinis", "nacionalinio", "nacionalizmas", "nacionalizmą", "puse", "pusei", "pusę", "pusė", "blicas", "blico", "blicą", "blicai", "blokų", "blokas", "suka", "sukasi"}

def is_swear(w_str: str) -> bool:
    clean = re.sub(r"^[^\w]+|[^\w]+$", "", w_str.lower().strip())
    if clean in EXCLUDE:
        return False
    # Check if censored token
    if clean in ["__", "[__]", "___"]:
        return True
    return any(r.search(clean) for r in SWEAR_RES)

profane_indices = []
for i, w in enumerate(words):
    if is_swear(w.word):
        profane_indices.append(i)

print(f"Total verified swear words & censored tokens: {len(profane_indices)}")
for idx in profane_indices:
    w = words[idx]
    s_idx = max(0, idx - 4)
    e_idx = min(len(words), idx + 5)
    ctx = " ".join(words[j].word for j in range(s_idx, e_idx))
    m, s = divmod(int(w.start), 60)
    print(f"[{m:02d}:{s:02d} | {w.start:.1f}s] '{w.word}' -> ...{ctx}...")


# Slide 7.0s windows around each detected profanity
windows = []
for p_idx in profane_indices:
    center_time = words[p_idx].start
    for offset in [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.0]:
        start_t = max(0.0, center_time + offset)
        end_t = start_t + 7.0
        
        win_words = [w for w in words if start_t <= w.start < end_t]
        profs = [w.word for w in win_words if is_swear(w.word)]
        
        if profs:
            windows.append({
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "prof_count": len(profs),
                "profs": profs,
                "word_count": len(win_words),
                "text": " ".join(w.word for w in win_words)
            })

# Sort windows by profanity count descending, then word count descending
windows.sort(key=lambda x: (x["prof_count"], x["word_count"]), reverse=True)

# Deduplicate overlapping windows (at least 15s apart)
deduped = []
for win in windows:
    if all(abs(win["start"] - d["start"]) > 12.0 for d in deduped):
        deduped.append(win)

print("\n================ TOP DISTINCT ~7s PROFANITY WINDOWS ================")
for i, d in enumerate(deduped[:10], 1):
    m, s = divmod(int(d['start']), 60)
    print(f"\nCandidate {i}: [{m:02d}:{s:02d}] {d['start']}s - {d['end']}s | Profanities ({d['prof_count']}): {d['profs']}")
    print(f"  Context: \"{d['text']}\"")
