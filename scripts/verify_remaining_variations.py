import sys
import subprocess
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

runner = BenchmarkRunner(Path("benchmark"))
temp_dir = Path("temp_exact")
YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"

# Test specific variations for 03, 04, 06
TESTS = [
    # 03: try ending right after 'dalykais' or before
    ("karalius_broken_03_v1", 1189.70, 1195.40),
    ("karalius_broken_03_v2", 1189.70, 1193.50), # ends after 'jisai' (has 'mindfakt kad jisai')
    ("karalius_broken_03_v3", 1189.70, 1197.30), # ends after 'neįtikinama čia'
    # 04: start in 640ms pause at 1260.00
    ("karalius_broken_04_v1", 1260.00, 1262.60), # 'ar jis actually mąsto lygiai taip pat pokerio žaidėjai'
    ("karalius_broken_04_v2", 1260.00, 1266.45), # includes 'sužaidė hendą'
    # 06: find where pause is before kainų
    ("karalius_broken_06_v1", 4030.20, 4036.35),
    ("karalius_broken_06_v2", 4029.50, 4036.35),
]

for cid, s, e in TESTS:
    mp4_path = temp_dir / f"{cid}.mp4"
    wav_path = temp_dir / f"{cid}.wav"
    if mp4_path.exists(): mp4_path.unlink()
    cmd = [
        "yt-dlp", "--download-sections", f"*{s}-{e}",
        "--force-keyframes-at-cuts", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        YOUTUBE_URL, "-o", str(mp4_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    cmd_wav = ["ffmpeg", "-y", "-i", str(mp4_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)]
    subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)
    start_sil = words[0].start if words else 0.0
    end_sil = dur - words[-1].end if words else 0.0
    status = "OK" if start_sil >= 0.095 and end_sil >= 0.095 else "FAIL"
    print(f"[{cid}] Range {s}-{e} ({dur:.2f}s) -> {status}: start={start_sil*1000:.1f}ms, end={end_sil*1000:.1f}ms | {' '.join(w.word for w in words[:6])} ... {' '.join(w.word for w in words[-3:])}")
