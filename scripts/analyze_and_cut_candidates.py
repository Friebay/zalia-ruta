import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

bench_dir = Path("benchmark")
runner = BenchmarkRunner(bench_dir)
raw_dir = bench_dir / "audio" / "raw_sections"

CLIPS = [
    ("karalius_prof_07", 4276.0),
    ("karalius_prof_08", 3690.0),
    ("karalius_prof_09", 4126.0),
    ("karalius_prof_10", 4001.5),
    ("karalius_prof_11", 4204.0),
]

for cid, base_t in CLIPS:
    wav_path = raw_dir / f"{cid}_raw.wav"
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    print(f"\n=======================================================")
    print(f"[{cid}] Duration: {dur:.2f}s (Base time in YT: {base_t}s)")
    print(f"=======================================================")
    
    # Run paprika acoustic
    words = runner.run_paprika_acoustic(f"{cid}_raw", audio, sr=sr, force=True)
    print(f"Total words: {len(words)}")
    for i, w in enumerate(words):
        gap = (w.start - words[i-1].end) if i > 0 else w.start
        print(f"  {w.start:6.3f}s - {w.end:6.3f}s (gap: {gap:+.3f}s): {w.word}")
