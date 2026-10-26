import sys
import re
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

words = parse_youtube_json3(Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3"))

targets = [
    "krč", "karoč", "vapše", "vapse", "nachui", "nahuj", "pizda", "pizdiec", "blet", "bliat",
    "chuj", "chujnia", "debil", "daun", "loch", "cringe", "krinž", "cancel", "stream",
    "react", "reakt", "actually", "ešli", "tafak", "fuck", "fak", "chill",
    "based", "toxic", "post", "vibe", "twitch", "ban", "donate", "hype",
    "chat", "spoil", "podcast", "tiktok", "reels", "shorts", "meme", "wtf", "omg", "woke",
    "banger", "bro", "skibidi", "sigma", "flex", "roast", "hater", "heiter", "užpisa", "pisom"
]

used_spans = [
    (6438, 6448), (6579, 6588), (6609, 6618), (6643, 6652), (6656, 6665),
    (2952, 2962), (577, 588), (3698, 3706), (44, 52), (680, 688)
]

matches = []
for i, w in enumerate(words):
    clean = re.sub(r"[^\w]", "", w.word.lower())
    for t in targets:
        if clean == t or (len(t) > 3 and clean.startswith(t)):
            # Check not in used_spans
            if any(s <= w.start <= e for s, e in used_spans):
                continue
            s_t = max(0.0, w.start - 1.5)
            e_t = s_t + 5.5
            win = [ww for ww in words if s_t <= ww.start <= e_t]
            text = " ".join(ww.word for ww in win)
            matches.append((s_t, e_t, t, clean, text))
            break

deduped = []
for m in matches:
    if all(abs(m[0] - d[0]) > 12.0 for d in deduped):
        deduped.append(m)

print(f"Total target matches: {len(deduped)}")
for idx, (st, et, tgt, cln, txt) in enumerate(deduped, 1):
    m, s = divmod(int(st), 60)
    print(f"{idx}. [{m:02d}:{s:02d} | {st:.1f}s-{et:.1f}s] match='{cln}' (target='{tgt}')")
    print(f"   \"{txt}\"\n")
