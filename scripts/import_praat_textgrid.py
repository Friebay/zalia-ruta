import argparse
import re
from pathlib import Path

from zalia_ruta.bench.srt_io import Cue, write_srt
from zalia_ruta.bench.validate import validate_reference_file


def parse_praat_textgrid(tg_path: Path, target_tier_name: str = "Etalonas"):
    content = tg_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines()]

    # 1. Check if verbose format (standard Praat export with xmin = ..., xmax = ..., text = ...)
    is_verbose = any("class = \"IntervalTier\"" in l for l in lines) or any("name =" in l for l in lines)
    if is_verbose:
        intervals = []
        inside_target_tier = False
        curr_xmin = None
        curr_xmax = None

        i = 0
        while i < len(lines):
            line = lines[i]
            # Match tier name
            name_match = re.search(r'name\s*=\s*"(.*?)"', line)
            if name_match:
                tier_name = name_match.group(1).strip()
                inside_target_tier = (tier_name.lower() == target_tier_name.lower())
            elif inside_target_tier and re.match(r'item\s*\[\d+\]:', line):
                # Next tier started!
                break
            elif inside_target_tier:
                xmin_m = re.match(r'xmin\s*=\s*([0-9.]+)', line)
                xmax_m = re.match(r'xmax\s*=\s*([0-9.]+)', line)
                text_m = re.match(r'text\s*=\s*"(.*?)"', line)
                if xmin_m:
                    curr_xmin = float(xmin_m.group(1))
                elif xmax_m:
                    curr_xmax = float(xmax_m.group(1))
                elif text_m:
                    curr_text = text_m.group(1).strip()
                    if curr_xmin is not None and curr_xmax is not None and curr_text:
                        intervals.append((curr_xmin, curr_xmax, curr_text))
                    curr_xmin = None
                    curr_xmax = None
            i += 1

        if intervals:
            return intervals

    # 2. Fallback: Short format (without field names)
    tier_found = False
    intervals = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == '"IntervalTier"':
            name_line = lines[i + 1] if i + 1 < len(lines) else ""
            if f'"{target_tier_name}"'.lower() in name_line.lower():
                tier_found = True
                num_intervals = int(lines[i + 4])
                i += 5
                for _ in range(num_intervals):
                    if i + 2 >= len(lines):
                        break
                    start_s = float(lines[i])
                    end_s = float(lines[i + 1])
                    text = lines[i + 2].strip('"').strip()
                    if text:
                        intervals.append((start_s, end_s, text))
                    i += 3
                break
        i += 1

    if not intervals:
        raise ValueError(f"Tier '{target_tier_name}' not found or had no non-empty intervals in {tg_path}")

    return intervals


def import_textgrid_to_reference(clip_id: str, target_tier: str = "Etalonas"):
    # Strip _adjusted if present to target the true base clip ID
    base_id = clip_id.replace("_adjusted", "")

    praat_dir = Path("praat_samples")
    adj_path = praat_dir / f"{base_id}_adjusted.TextGrid"
    norm_path = praat_dir / f"{base_id}.TextGrid"

    if adj_path.exists():
        tg_path = adj_path
    elif norm_path.exists():
        tg_path = norm_path
    else:
        raise FileNotFoundError(f"Praat TextGrid not found for {base_id} in {praat_dir}")

    intervals = parse_praat_textgrid(tg_path, target_tier_name=target_tier)
    cues = [
        Cue(index=idx + 1, start=round(s, 3), end=round(e, 3), text=txt)
        for idx, (s, e, txt) in enumerate(intervals)
    ]

    ref_path = Path("benchmark/references") / f"{base_id}.srt"
    write_srt(cues, ref_path)
    print(f"[{base_id}] Imported {len(cues)} cues from {tg_path.name} -> {ref_path}")

    # Validate reference file
    errors = validate_reference_file(ref_path)
    if errors:
        print(f"[{base_id}] WARNING: Validation reported {len(errors)} errors:")
        for err in errors:
            print(f"   - {err}")
    else:
        print(f"[{base_id}] SUCCESS: Validated reference strictly OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import edited Praat TextGrid into benchmark reference SRT.")
    parser.add_argument("--clip", default=None, help="Clip ID to import (e.g. karalius_10s_01)")
    parser.add_argument("--all", action="store_true", default=False, help="Import all 5 adjusted 10s clips")
    parser.add_argument("--tier", default="Etalonas", help="Tier name to extract (default: Etalonas)")
    args = parser.parse_args()

    if args.clip:
        import_textgrid_to_reference(args.clip, target_tier=args.tier)
    else:
        # Default: auto-discover all adjusted files in praat_samples
        praat_dir = Path("praat_samples")
        adj_files = sorted(praat_dir.glob("*_adjusted.TextGrid"))
        if not adj_files:
            print("No '*_adjusted.TextGrid' files found in praat_samples/.")
        for f in adj_files:
            cid = f.name.replace("_adjusted.TextGrid", "")
            try:
                import_textgrid_to_reference(cid, target_tier=args.tier)
            except Exception as e:
                print(f"[{cid}] Error: {e}")
