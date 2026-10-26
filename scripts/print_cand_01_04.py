import sys
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

runner = BenchmarkRunner(Path("benchmark"))
for cid, st_off in [("cand_01", 918.5), ("cand_02", 947.5), ("cand_03", 1187.5), ("cand_04", 1256.5)]:
    wav = Path(f"temp_wide/{cid}_wide.wav")
    audio, sr = sf.read(str(wav))
    words = runner.run_paprika_acoustic(f"{cid}_wide", audio, sr=sr, force=False)
    print(f"\n=== {cid} ===")
    for idx, w in enumerate(words):
        abs_s = round(st_off + w.start, 3)
        abs_e = round(st_off + w.end, 3)
        print(f"{idx:2d}: [{abs_s:7.2f}s - {abs_e:7.2f}s] '{w.word}'")
