import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def inspect_targets():
    sub_path = Path("praat_samples/temp_sub.lt-orig.json3")
    data = json.loads(sub_path.read_text(encoding="utf-8"))

    tokens = []
    for ev in data.get("events", []):
        t_start = ev.get("tStartMs", 0)
        for seg in ev.get("segs", []):
            txt = seg.get("utf8", "").strip()
            if not txt or txt == "\n":
                continue
            offset = seg.get("tOffsetMs", 0)
            t = (t_start + offset) / 1000.0
            dur = seg.get("dDurationMs", 350) / 1000.0
            for part in txt.split():
                tokens.append({"word": part, "start": t, "end": t + dur})

    targets = [
        ("dalbajobu", 1180.68),
        ("pistu", 1280.44),
        ("prisipist", 1327.00),
        ("kurv", 1545.12),
        ("blezkoks", 2424.20),
    ]

    for name, tgt in targets:
        closest = min(range(len(tokens)), key=lambda i: abs(tokens[i]["start"] - tgt))
        print(f"\n=== Target '{name}' ~{tgt:.1f}s (index {closest}: {tokens[closest]['word']}) ===")
        for offset in range(-8, 9):
            idx = closest + offset
            if 0 <= idx < len(tokens):
                marker = " >>> " if offset == 0 else "     "
                w = tokens[idx]
                print(f"{marker}[{idx}] {w['start']:8.2f}s - {w['end']:8.2f}s: {w['word']}")

if __name__ == "__main__":
    inspect_targets()
