import subprocess
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Download 3685.0 to 3695.0
out_m4a = Path("benchmark/audio/raw_sections/test_prof08_wide.m4a")
out_wav = Path("benchmark/audio/raw_sections/test_prof08_wide.wav")

cmd = [
    "yt-dlp",
    "--download-sections", "*3685-3696",
    "-f", "ba[ext=m4a]/ba",
    "https://www.youtube.com/watch?v=7og_55op5Kg",
    "-o", str(out_m4a)
]
subprocess.run(cmd, check=True)

cmd_wav = [
    "ffmpeg", "-y", "-i", str(out_m4a),
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    str(out_wav)
]
subprocess.run(cmd_wav, check=True)
if out_m4a.exists():
    out_m4a.unlink()

from zalia_ruta.bench.systems import BenchmarkRunner
runner = BenchmarkRunner(Path("benchmark"))
audio, sr = sf.read(str(out_wav))
words = runner.run_paprika_acoustic("prof08_wide", audio, sr=sr, force=True)

print(f"Total words in wide section (duration {len(audio)/sr:.2f}s):")
for i, w in enumerate(words):
    gap = (w.start - words[i-1].end) if i > 0 else w.start
    abs_t = 3685.0 + w.start
    print(f"  [{abs_t:7.2f}s | {w.start:5.2f}s - {w.end:5.2f}s] (gap: {gap:+.3f}s): {w.word}")
