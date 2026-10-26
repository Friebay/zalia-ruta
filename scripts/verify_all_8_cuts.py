import sys
import subprocess
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

# Let's adjust cuts with ample silence padding (>150ms)
CANDIDATE_CUTS = {
    "karalius_broken_01": (921.80, 927.40), # verified: start_sil=240ms, end_sil=239ms
    "karalius_broken_02": (949.80, 956.10), # was 955.90, +200ms -> end_sil ~ 290ms
    "karalius_broken_03": (1189.80, 1195.50), # cut before 'ir' at 1195.70 -> end_sil ~ 150ms
    "karalius_broken_04": (1258.50, 1263.65), # ends after 'paklausiu' (1263.15s) -> end_sil ~ 500ms
    "karalius_broken_05": (3289.50, 3294.75), # start before 'tiesiog' (3289.92s), end after 'nežinau' (3294.49s) -> sil ~ 300ms
    "karalius_broken_06": (4032.40, 4036.35), # start before 'bet' (4032.80s), end after 'ane' (4036.05s) -> sil ~ 300ms
    "karalius_broken_07": (4127.80, 4132.50), # start before 'žodžiu' (4128.22s), end after 'nu' (4132.25s) -> sil ~ 250ms
    "karalius_broken_08": (6023.20, 6028.55), # start before 'nes' (6023.70s), end after 'ir' (6028.30s) -> sil ~ 250ms
}

runner = BenchmarkRunner(Path("benchmark"))
temp_dir = Path("temp_exact")
temp_dir.mkdir(exist_ok=True)
YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"

for cid, (s, e) in CANDIDATE_CUTS.items():
    mp4_path = temp_dir / f"{cid}.mp4"
    wav_path = temp_dir / f"{cid}.wav"
    
    # Re-slice if needed
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
    
    print(f"\n[{cid}] Range: {s}s - {e}s (dur={dur:.2f}s)")
    print(f"  Silence: start={start_sil*1000:.1f}ms | end={end_sil*1000:.1f}ms (threshold >= 95ms)")
    print(f"  Words ({len(words)}): {' '.join(w.word for w in words)}")
    assert start_sil >= 0.095, f"Start silence fail on {cid}: {start_sil*1000:.1f}ms"
    assert end_sil >= 0.095, f"End silence fail on {cid}: {end_sil*1000:.1f}ms"

print("\nSUCCESS: ALL 8 CLIPS HAVE PERFECT SILENCE BOUNDARIES (>= 95ms)!")
