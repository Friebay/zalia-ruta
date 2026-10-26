import sys
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

# Candidate targets in 7og_55op5Kg
# Let's inspect a wider window (+-2s) around each target to find exact clean silence cuts
RAW_CANDIDATES = [
    {
        "id": "cand_01",
        "approx_center": 923.5,
        "desc": "vapšė pizda mums actually vapšė pizda",
    },
    {
        "id": "cand_02",
        "approx_center": 952.5,
        "desc": "visiškai viskas nu fuck this sakau",
    },
    {
        "id": "cand_03",
        "approx_center": 1192.5,
        "desc": "aš tiesiog visiškai mind fucked",
    },
    {
        "id": "cand_04",
        "approx_center": 1261.5,
        "desc": "ar jis actually mąsto lygiai taip pat",
    },
    {
        "id": "cand_05",
        "approx_center": 3291.5,
        "desc": "už debiles nelaikyt",
    },
    {
        "id": "cand_06",
        "approx_center": 4035.5,
        "desc": "palygint negali žodžiu fuck this ane",
    },
    {
        "id": "cand_07",
        "approx_center": 4129.5,
        "desc": "žodžiu fucking hell ten aš nežinau",
    },
    {
        "id": "cand_08",
        "approx_center": 6025.5,
        "desc": "nes čia nu actually ragai būtų bet socdemam pochui",
    },
]

runner = BenchmarkRunner(Path("benchmark"))
temp_dir = Path("temp_wide")
temp_dir.mkdir(exist_ok=True)
YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"

print("Downloading wide audio (10s) around each candidate...")
for c in RAW_CANDIDATES:
    cid = c["id"]
    center = c["approx_center"]
    w_start = round(center - 5.0, 2)
    w_end = round(center + 5.0, 2)
    
    mp4_path = temp_dir / f"{cid}_wide.mp4"
    wav_path = temp_dir / f"{cid}_wide.wav"
    
    if not mp4_path.exists():
        cmd = [
            "yt-dlp",
            "--download-sections", f"*{w_start}-{w_end}",
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
    words = runner.run_paprika_acoustic(f"{cid}_wide", audio, sr=sr, force=True)
    
    print(f"\n==========================================")
    print(f"[{cid}] Wide window: {w_start}s - {w_end}s (dur={dur:.2f}s) | target: {c['desc']}")
    for idx, w in enumerate(words):
        abs_s = round(w_start + w.start, 3)
        abs_e = round(w_start + w.end, 3)
        print(f"  {idx:2d}: [{w.start:5.2f}s - {w.end:5.2f}s | abs: {abs_s:7.2f}s - {abs_e:7.2f}s] '{w.word}'")
