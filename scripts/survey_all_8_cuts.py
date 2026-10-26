import sys
import subprocess
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

# Proposed cuts
CANDIDATE_CUTS = {
    "karalius_broken_01": (921.80, 927.40), # verified: start=240ms, end=239ms
    "karalius_broken_02": (949.80, 954.45), # ends after 'čia' in 390ms pause -> ~4.65s
    "karalius_broken_03": (1189.80, 1194.85), # ends after 'tokiais' / before 'dalykais'? Or let's see
    "karalius_broken_04": (1258.50, 1263.65), # ends in 400ms pause
    "karalius_broken_05": (3289.50, 3294.75), # start in 870ms pause, end in 440ms pause
    "karalius_broken_06": (4032.40, 4036.35), # start in 270ms pause, end in 440ms pause
    "karalius_broken_07": (4127.80, 4132.50), # start in 950ms pause, end in 430ms pause
    "karalius_broken_08": (6023.20, 6028.55), # start in 1000ms pause, end in 440ms pause
}

runner = BenchmarkRunner(Path("benchmark"))
temp_dir = Path("temp_exact")
temp_dir.mkdir(exist_ok=True)
YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"

for cid, (s, e) in CANDIDATE_CUTS.items():
    mp4_path = temp_dir / f"{cid}.mp4"
    wav_path = temp_dir / f"{cid}.wav"
    
    if mp4_path.exists():
        mp4_path.unlink()
        
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{s}-{e}",
        "--force-keyframes-at-cuts",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        YOUTUBE_URL,
        "-o", str(mp4_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    cmd_wav = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path)
    ]
    subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)
    
    start_sil = words[0].start if words else 0.0
    end_sil = dur - words[-1].end if words else 0.0
    
    status = "OK" if start_sil >= 0.095 and end_sil >= 0.095 else "FAIL"
    print(f"\n[{cid}] Range: {s}s - {e}s (dur={dur:.2f}s) -> {status}")
    print(f"  Silence: start={start_sil*1000:.1f}ms | end={end_sil*1000:.1f}ms")
    print(f"  Words ({len(words)}): {' '.join(w.word for w in words)}")
