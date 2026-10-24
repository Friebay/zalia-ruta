import sys
from pathlib import Path
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from scripts.import_praat_textgrid import parse_praat_textgrid

for cid in ["orlauskas_prof_02", "orlauskas_prof_03", "orlauskas_prof_04", "orlauskas_prof_05"]:
    p = Path(f"praat_samples/{cid}_adjusted.TextGrid")
    for tier in ["Acoustic_Refined", "Etalonas"]:
        try:
            items = parse_praat_textgrid(p, target_tier_name=tier)
            words = " ".join(t for _, _, t in items)
            print(f"{cid} [{tier}]: {len(items)} words -> {words}")
        except Exception as e:
            print(f"{cid} [{tier}]: {e}")
