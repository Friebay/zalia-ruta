import sys
import subprocess
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

# Candidate ~5-6s spans from 7og_55op5Kg
CANDIDATES = [
    {
        "id": "cand_01",
        "start": 921.0,
        "end": 927.0,
        "desc": "vapšė mums, actually vapše...",
    },
    {
        "id": "cand_02",
        "start": 949.5,
        "end": 955.5,
        "desc": "visiškai viskas. Nu fuck this...",
    },
    {
        "id": "cand_03",
        "start": 1189.5,
        "end": 1195.5,
        "desc": "aš tiesiog visiškai mind fucked...",
    },
    {
        "id": "cand_04",
        "start": 1258.5,
        "end": 1264.5,
        "desc": "ar jis actually mąsto. Lygiai taip pat pokerio žaidėjai...",
    },
    {
        "id": "cand_05",
        "start": 3289.0,
        "end": 3295.0,
        "desc": "už debiles nelaikyt...",
    },
    {
        "id": "cand_06",
        "start": 4033.0,
        "end": 4038.5,
        "desc": "fuck this ane?...",
    },
    {
        "id": "cand_07",
        "start": 4126.5,
        "end": 4132.5,
        "desc": "žodžiu fucking hell...",
    },
    {
        "id": "cand_08",
        "start": 6023.0,
        "end": 6028.5,
        "desc": "actually ragai būtų, bet socdemam...",
    },
]

runner = BenchmarkRunner(Path("benchmark"))
temp_dir = Path("temp_candidates")
temp_dir.mkdir(exist_ok=True)

YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"

for c in CANDIDATES:
    cid = c["id"]
    s = c["start"]
    e = c["end"]
    wav_path = temp_dir / f"{cid}.wav"
    mp4_path = temp_dir / f"{cid}.mp4"
    
    print(f"\n[{cid}] Fetching {s}s - {e}s ({e-s:.1f}s)...")
    if not mp4_path.exists():
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
    hyp_text = " ".join(w.word for w in words)
    print(f"[{cid}] Duration: {dur:.2f}s | Target description: {c['desc']}")
    print(f"[{cid}] Paprika ASR Recognized ({len(words)} words):")
    print(f"       \"{hyp_text}\"")
