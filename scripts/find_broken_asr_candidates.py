import sys
import re
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.youtube import parse_youtube_json3

json3_path = Path("benchmark/youtube/7og_55op5Kg.lt-orig.json3")
words = parse_youtube_json3(json3_path)
print(f"Loaded {len(words)} words from {json3_path}")

# English / gamer / internet slang patterns common in Lithuanian streaming
broken_keywords = [
    # English loanwords / gamer terms
    "cringe", "krinž", "krindž", "cancel", "kensel", "stream", "strim",
    "content", "kontent", "fuck", "fak", "chill", "čil", "based", "beizd",
    "rage", "reidž", "toxic", "toksik", "post", "feed", "fid", "vibe", "vaib",
    "twitch", "tvyč", "ban", "banina", "donate", "doneit", "random",
    "hype", "haip", "view", "vjūv", "like", "laik", "dislike", "subscribe",
    "chat", "čiat", "čat", "spoil", "spoiler", "spoilina", "highlight",
    "podcast", "podkast", "tiktok", "reels", "shorts", "meme", "mem",
    "wtf", "omg", "woke", "vouk", "scam", "skem", "banger", "benger",
    "bro", "skibidi", "sigma", "flex", "fleks", "roast", "roust",
    "hater", "heiter", "follow", "folover", "unfollow", "react", "reakt",
    # Heavy slang / colloquialisms
    "vapše", "vapse", "krč", "krc", "nachuj", "nahuj", "chuj", "chujnia",
    "pizda", "pizdiec", "pize", "blet", "bliat", "debil", "daun", "loch",
    "pisom", "pisa", "uzpisa", "užpisa", "suvalgė", "sukramtė"
]

# Exclude intervals already used by existing clips
# Let's read manifest
import json
manifest = json.loads(Path("benchmark/manifest.json").read_text(encoding="utf-8"))
used_intervals = []
# From existing clips in 7og_55op5Kg:
# prof_07: ~6439s - 6445s
# prof_08: ~6580s - 6586s
# prof_09: ~6610s - 6616s
# prof_10: ~6644s - 6650s
# prof_11: ~6657s - 6663s
# silence_06: 2953.5 - 2961.0
# silence_07: 578.0 - 586.8
# silence_08: 3699.4 - 3704.5
# silence_09: 45.5 - 51.0
# silence_10: 681.0 - 686.5

for i, w in enumerate(words):
    w_clean = re.sub(r"[^\w]", "", w.word.lower())
    for kw in broken_keywords:
        if kw in w_clean:
            # Found a candidate around word i
            # Look at a ~5s window: from w.start - 1.5 to w.start + 3.5
            start_t = max(0.0, w.start - 1.5)
            end_t = start_t + 5.0
            
            # Words in this window
            win_words = [ww for ww in words if start_t <= ww.start <= end_t]
            text = " ".join(ww.word for ww in win_words)
            m, s = divmod(int(start_t), 60)
            print(f"[{m:02d}:{s:02d} | {start_t:.1f}s-{end_t:.1f}s] kw='{kw}' matched='{w.word}' ({len(win_words)} words)")
            print(f"   \"{text}\"")
            break
