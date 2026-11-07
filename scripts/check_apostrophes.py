import json
import re
from pathlib import Path
from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.bench.score import normalize_word
from zalia_ruta.bench.systems import deserialize_words

refs_dir = Path("benchmark/references")
hyp_dir = Path("benchmark/hyp/paprika_ct2_acoustic")
apostrophe_pattern = re.compile(r"['’`]")

results = []

for srt_path in sorted(refs_dir.glob("*.srt")):
    cid = srt_path.stem
    hyp_file = hyp_dir / f"{cid}.json"
    if not hyp_file.exists():
        continue
    ref_cues = parse_srt(srt_path.read_text(encoding="utf-8"))
    hyp_words = deserialize_words(json.loads(hyp_file.read_text(encoding="utf-8")))

    for c in ref_cues:
        if apostrophe_pattern.search(c.text):
            ref_norm = normalize_word(c.text)
            overlapping_hyps = [
                normalize_word(h.word)
                for h in hyp_words
                if max(c.start - 0.25, h.start) < min(c.end + 0.25, h.end)
            ]
            results.append((cid, c.text, ref_norm, overlapping_hyps))

print(f"Total occurrences of apostrophe words in ground truth: {len(results)}")
print("\n=== REFERENCE vs MODEL HYPOTHESIS ===")
for cid, orig_ref, ref_norm, hyps in results:
    match_status = "MATCH" if ref_norm in hyps else "MISMATCH"
    print(f"[{match_status:8s}] {orig_ref:18s} (norm: {ref_norm:15s}) -> Model: {hyps}")

