import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)

# Let's find spans of k words (1 <= k <= 5) that maximize the duration (including surrounding pause margins up to next/prev word)
# For a slice defined from start_t to end_t, words inside are those with start_t <= w.start and w.end <= end_t.
# The user wants:
# - maximum of 5 words (1 to 5 words)
# - not longer than 1 minute (<= 60.0s)
# - longest duration clips

# Let's test all candidate sub-sequences of words of length 1, 2, 3, 4, 5:
# For words[i : i + k] (where 1 <= k <= 5):
# Previous word is words[i-1] (if i > 0), next word is words[i+k] (if i+k < len(words))
# The maximum duration we can cut this slice without including any other words is:
# start_t = words[i-1].end + margin
# end_t = words[i+k].start - margin
# Duration = end_t - start_t <= words[i+k].start - words[i-1].end

spans = []
for k in range(1, 6):
    for i in range(len(words) - k + 1):
        w_first = words[i]
        w_last = words[i + k - 1]
        
        # Silence before
        prev_end = words[i-1].end if i > 0 else 0.0
        # Silence after
        next_start = words[i+k].start if (i + k) < len(words) else words[-1].end + 10.0
        
        # Maximum possible duration without including prev or next word
        max_possible_dur = next_start - prev_end
        
        # Realistic clip cut: start midway in prev gap (or 0.2s after prev_end), end midway in next gap (or 0.2s before next_start)
        # To avoid exceeding 60s
        actual_start = max(prev_end + 0.1, w_first.start - 1.0)
        actual_end = min(next_start - 0.1, w_last.end + 1.0)
        
        # What if we take the entire silence gap around them?
        # e.g. start = max(prev_end + 0.15, next_start - 60.0)
        # Let's compute the max duration <= 60.0s that contains ONLY these k words:
        full_silence_dur = next_start - prev_end - 0.3
        clip_dur = min(60.0, full_silence_dur)
        
        # Word text
        text = " ".join(words[j].word for j in range(i, i + k))
        
        spans.append({
            "k": k,
            "start_idx": i,
            "first_word_t": w_first.start,
            "last_word_t": w_last.end,
            "prev_word": words[i-1].word if i > 0 else "<START>",
            "prev_end": prev_end,
            "next_word": words[i+k].word if (i+k) < len(words) else "<END>",
            "next_start": next_start,
            "max_dur": full_silence_dur,
            "text": text,
        })

# Sort by max possible duration descending
spans.sort(key=lambda x: x["max_dur"], reverse=True)

# Deduplicate overlapping spans
deduped = []
for s in spans:
    # check overlap
    if not any(abs(s["first_word_t"] - d["first_word_t"]) < 20.0 for d in deduped):
        deduped.append(s)

print("=== Top 15 Longest Candidate Intervals with <= 5 Words (under 60s) ===")
for idx, s in enumerate(deduped[:15], 1):
    m, sec = divmod(int(s['first_word_t']), 60)
    print(f"\n#{idx}: [{m:02d}:{sec:02d}] Max duration: {s['max_dur']:.2f}s | Words ({s['k']}): \"{s['text']}\"")
    print(f"    Gap before: {s['first_word_t'] - s['prev_end']:.2f}s (after '{s['prev_word']}')")
    print(f"    Speech duration: {s['last_word_t'] - s['first_word_t']:.2f}s")
    print(f"    Gap after: {s['next_start'] - s['last_word_t']:.2f}s (before '{s['next_word']}')")
