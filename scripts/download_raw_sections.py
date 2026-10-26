import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"
YOUTUBE_ID = "7og_55op5Kg"

# Raw window ranges to download (wide margins to allow safe trimming)
CANDIDATE_WINDOWS = [
    {
        "id": "karalius_prof_07",
        "dl_start": 4276.0,
        "dl_end": 4285.0,
        "notes": "Karalius reaction, 'What the fuck? Kodėl? Kas jum yra...' (~7.0s)",
    },
    {
        "id": "karalius_prof_08",
        "dl_start": 3690.0,
        "dl_end": 3700.0,
        "notes": "Karalius commentary, 'kaip užpisa, kad tiek heiterių turi...' (~7.5s)",
    },
    {
        "id": "karalius_prof_09",
        "dl_start": 4126.0,
        "dl_end": 4135.0,
        "notes": "Karalius reaction, 'Žodžiu, fucking hell. Ten aš nežinau...' (~6.5s)",
    },
    {
        "id": "karalius_prof_10",
        "dl_start": 4001.5,
        "dl_end": 4011.5,
        "notes": "Karalius commentary, 'visiškai su šūdais tave maišys...' (~7.0s)",
    },
    {
        "id": "karalius_prof_11",
        "dl_start": 4204.0,
        "dl_end": 4213.0,
        "notes": "Karalius politics rant, 'išrinks kokį debilą, kuris žadės...' (~6.8s)",
    },
]

raw_dir = Path("benchmark/audio/raw_sections")
raw_dir.mkdir(parents=True, exist_ok=True)

print("--- Step 1: Downloading raw audio sections ---")
for cand in CANDIDATE_WINDOWS:
    cid = cand["id"]
    s = cand["dl_start"]
    e = cand["dl_end"]
    out_m4a = raw_dir / f"{cid}_raw.m4a"
    out_wav = raw_dir / f"{cid}_raw.wav"
    
    if not out_wav.exists():
        print(f"Downloading {cid} ({s}s - {e}s)...")
        cmd_dl = [
            "yt-dlp",
            "--download-sections", f"*{s}-{e}",
            "-f", "ba[ext=m4a]/ba",
            YOUTUBE_URL,
            "-o", str(out_m4a),
        ]
        subprocess.run(cmd_dl, check=True)
        
        # Convert to 16kHz mono WAV
        cmd_wav = [
            "ffmpeg", "-y", "-i", str(out_m4a),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(out_wav)
        ]
        subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if out_m4a.exists():
            out_m4a.unlink()
        print(f"Saved: {out_wav}")
    else:
        print(f"Already downloaded: {out_wav}")

print("\nAll raw audio sections downloaded successfully.")
