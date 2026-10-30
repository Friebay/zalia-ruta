import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector, clean_token

FALSE_POSITIVE_STEMS = {
    "naci", "nacist", "nacional", "epistem", "problem",
    "pistol", "tipisk", "prisikur", "sukel", "sukaust", "atsuk"
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

for vid in VIDEO_IDS:
    p = Path(f"benchmark/youtube/{vid}.lt-orig.json3")
    words = parse_youtube_json3(p)
    
    true_profs = []
    for idx, w in enumerate(words):
        if is_true_profanity(w.word, detector):
            true_profs.append((idx, w))
            
    print(f"\n=======================================================")
    print(f"VIDEO: {vid} (Duration ~{words[-1].start/60:.1f} min, Total words: {len(words)})")
    print(f"True profanities detected in YouTube captions: {len(true_profs)}")
    for i, w in true_profs:
        cl = clean_token(w.word)
        print(f"  [{w.start:7.2f}s] {w.word} (clean: {cl})")
        
    # Check max in any 20s window
    best_window = None
    best_count = 0
    all_qualifying = []
    
    max_t = words[-1].start
    t = 0.0
    while t + 20.0 <= max_t + 1.0:
        win_words = [w for w in words if t <= w.start < t + 20.0]
        win_profs = [w for w in win_words if is_true_profanity(w.word, detector)]
        c = len(win_profs)
        if c > best_count:
            best_count = c
            best_window = (t, t + 20.0, [w.word for w in win_profs], " ".join(w.word for w in win_words))
        if c >= 5:
            all_qualifying.append((t, t + 20.0, [w.word for w in win_profs], " ".join(w.word for w in win_words)))
        t += 1.0
        
    print(f"\nMax TRUE profanity density in 20s window: {best_count}")
    if best_window:
        print(f"  Best interval: {best_window[0]:.1f}s - {best_window[1]:.1f}s")
        print(f"  Profanities ({len(best_window[2])}): {best_window[2]}")
        print(f"  Text: {best_window[3]}")
    print(f"Windows with >= 5 profanities: {len(all_qualifying)}")
