import sys
import json
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector

json3_path = Path("benchmark/youtube/B7Mygu8uiYM.lt-orig.json3")
words = parse_youtube_json3(json3_path)
print(f"Total words parsed: {len(words)}")

detector = ProfanityDetector()

# 1. Identify all profanity words
profane_indices = []
for i, w in enumerate(words):
    if detector.is_profanity(w.word):
        profane_indices.append(i)

print(f"Total profane words: {len(profane_indices)}")
for idx in profane_indices:
    w = words[idx]
    s_idx = max(0, idx - 4)
    e_idx = min(len(words), idx + 5)
    ctx = " ".join(words[j].word for j in range(s_idx, e_idx))
    print(f"[{w.start:.1f}s | {w.start/60:.2f}min] '{w.word}' -> ...{ctx}...")


# 2. Find best ~7 second windows for profanity
# We can slide a 7.0s window around profane occurrences or across time
prof_windows = []
for p_idx in profane_indices:
    # Try different offsets around this word to maximize profanity count within 7s
    center_time = words[p_idx].start
    for offset in [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.0]:
        start_t = max(0.0, center_time + offset)
        end_t = start_t + 7.0
        
        # Words in window
        window_words = [w for w in words if start_t <= w.start < end_t]
        profs = [w.word for w in window_words if detector.is_profanity(w.word)]
        
        if len(profs) >= 2:
            prof_windows.append({
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "prof_count": len(profs),
                "profs": profs,
                "word_count": len(window_words),
                "text": " ".join(w.word for w in window_words)
            })

# Sort windows by profanity count descending, then word count descending
prof_windows.sort(key=lambda x: (x["prof_count"], x["word_count"]), reverse=True)

# Deduplicate overlapping windows (at least 15s apart so clips don't overlap)
deduped_prof = []
for win in prof_windows:
    if all(abs(win["start"] - d["start"]) > 10.0 for d in deduped_prof):
        deduped_prof.append(win)

print("\n--- TOP 10 PROFANITY CLIPS (~7s) ---")
for i, d in enumerate(deduped_prof[:10], 1):
    m, s = divmod(int(d['start']), 60)
    print(f"{i}. [{m:02d}:{s:02d}] {d['start']}s - {d['end']}s | Count: {d['prof_count']} | Profanities: {d['profs']}")
    print(f"   Text: {d['text']}")

# 3. Find clean gaps between meaningful multi-character words
clean_gaps = []
for i in range(len(words) - 1):
    w1 = words[i]
    w2 = words[i + 1]
    gap = w2.start - w1.end
    c1 = w1.word.lower().strip(".,!?:;\"'()[]-")
    c2 = w2.word.lower().strip(".,!?:;\"'()[]-")
    
    # Exclude single letter tokens and weak interjections
    if len(c1) >= 2 and len(c2) >= 2 and c1 not in ["nu", "jo", "tai"] and c2 not in ["nu", "jo", "tai"]:
        if 3.5 <= gap <= 5.8:
            # Let's align a 7.0s window so w1 is comfortably inside at start and w2 is inside at end
            # e.g. start at w1.start - 0.8s
            start_t = max(0.0, w1.start - 0.8)
            end_t = start_t + 7.0
            if w2.start + 0.5 <= end_t: # w2 fits before end
                win_words = [w for w in words if start_t <= w.start < end_t]
                clean_gaps.append({
                    "start": round(start_t, 2),
                    "end": round(end_t, 2),
                    "gap": round(gap, 2),
                    "w1": w1.word,
                    "w2": w2.word,
                    "text": " ".join(w.word for w in win_words)
                })

clean_gaps.sort(key=lambda x: x["gap"], reverse=True)
deduped_gaps = []
for g in clean_gaps:
    if all(abs(g["start"] - d["start"]) > 25.0 for d in deduped_gaps):
        deduped_gaps.append(g)

print("\n--- CLEANEST 5+ GAP / SILENCE CLIPS (~7s) (Real Words, No Fillers) ---")
for i, d in enumerate(deduped_gaps[:8], 1):
    m, s = divmod(int(d['start']), 60)
    print(f"{i}. [{m:02d}:{s:02d}] {d['start']}s - {d['end']}s | Silence gap: {d['gap']}s")
    print(f"   Gap between: '{d['w1']}' ... [{d['gap']}s silence] ... '{d['w2']}'")
    print(f"   Clip text: \"{d['text']}\"\n")


