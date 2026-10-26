import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def inspect_file(p: Path):
    content = p.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    tiers = {}
    current_tier = None
    curr_xmin = None
    curr_xmax = None
    
    for line in lines:
        name_m = re.search(r'name\s*=\s*"(.*?)"', line)
        if name_m:
            current_tier = name_m.group(1).strip()
            tiers[current_tier] = []
        elif current_tier:
            xmin_m = re.search(r'xmin\s*=\s*([0-9.]+)', line)
            xmax_m = re.search(r'xmax\s*=\s*([0-9.]+)', line)
            text_m = re.search(r'text\s*=\s*"(.*?)"', line)
            if xmin_m:
                curr_xmin = float(xmin_m.group(1))
            elif xmax_m:
                curr_xmax = float(xmax_m.group(1))
            elif text_m:
                txt = text_m.group(1).strip()
                if txt:
                    tiers[current_tier].append((curr_xmin, curr_xmax, txt))
                curr_xmin = None
                curr_xmax = None
                
    print(f"\n==========================================")
    print(f"File: {p.name}")
    for t_name, intervals in tiers.items():
        words = [w for _, _, w in intervals]
        print(f"  Tier '{t_name}': {len(intervals)} words -> {' '.join(words)}")

for p in sorted(Path("praat_samples").glob("karalius_*_adjusted.TextGrid")):
    inspect_file(p)
