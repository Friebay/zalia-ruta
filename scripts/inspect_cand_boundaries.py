import sys
from pathlib import Path
import soundfile as sf
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

temp_dir = Path("temp_candidates")
runner = BenchmarkRunner(Path("benchmark"))

for cid in ["cand_01", "cand_02", "cand_03", "cand_04", "cand_05", "cand_06", "cand_07", "cand_08"]:
    wav_path = temp_dir / f"{cid}.wav"
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)
    if words:
        start_sil = words[0].start
        end_sil = dur - words[-1].end
        print(f"[{cid}] dur={dur:.2f}s | start_sil={start_sil*1000:.1f}ms | end_sil={end_sil*1000:.1f}ms | {len(words)} words")
        print(f"    first='{words[0].word}' last='{words[-1].word}'")
    else:
        print(f"[{cid}] No words detected")
