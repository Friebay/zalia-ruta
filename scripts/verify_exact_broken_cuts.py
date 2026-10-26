import sys
import subprocess
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

CLIPS_TEST = [
    {
        "id": "karalius_broken_01",
        "start": 921.80,
        "end": 927.40,
        "desc": "vapšė pizda mums actually vapšė pizda",
    },
    {
        "id": "karalius_broken_02",
        "start": 949.80,
        "end": 955.90,
        "desc": "buvo visiškai viskas nu fuck this sakau jaunimas turėtų čia būti informuoti",
    },
    {
        "id": "karalius_broken_03",
        "start": 1189.75,
        "end": 1195.80,
        "desc": "arba aš tiesiog visiškai mind fucked kad jisai...",
    },
    {
        "id": "karalius_broken_04",
        "start": 1258.50,
        "end": 1263.50,
        "desc": "delusional yra ar jis actually mąsto lygiai taip pat pokerio žaidėjai paklausiu",
    },
    {
        "id": "karalius_broken_05",
        "start": 3289.60,
        "end": 3294.75,
        "desc": "tiesiog nu už už debiles nelaikyt...",
    },
    {
        "id": "karalius_broken_06",
        "start": 4031.50,
        "end": 4036.35,
        "desc": "jie per tupę yra... kainų parduotuvių palygint negali žodžiu fuck this ane",
    },
    {
        "id": "karalius_broken_07",
        "start": 4127.80,
        "end": 4132.50,
        "desc": "žodžiu fucking hell ten aš nežinau tiek daug yra va tokių blokų kad kur nu",
    },
    {
        "id": "karalius_broken_08",
        "start": 6023.30,
        "end": 6028.55,
        "desc": "nes čia nu actually ragai būtų bet socdemam pochui ir",
    },
]

runner = BenchmarkRunner(Path("benchmark"))
temp_dir = Path("temp_exact")
temp_dir.mkdir(exist_ok=True)
YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"

for c in CLIPS_TEST:
    cid = c["id"]
    s = c["start"]
    e = c["end"]
    mp4_path = temp_dir / f"{cid}.mp4"
    wav_path = temp_dir / f"{cid}.wav"
    
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
    
    start_sil = words[0].start if words else 0.0
    end_sil = dur - words[-1].end if words else 0.0
    
    print(f"\n[{cid}] Range: {s}s - {e}s (dur={dur:.2f}s)")
    print(f"  Silence: start={start_sil*1000:.1f}ms | end={end_sil*1000:.1f}ms (threshold >= 95ms)")
    print(f"  Words ({len(words)}): {' '.join(w.word for w in words)}")
    assert start_sil >= 0.095, f"Start silence fail on {cid}: {start_sil*1000:.1f}ms"
    assert end_sil >= 0.095, f"End silence fail on {cid}: {end_sil*1000:.1f}ms"
print("\nALL 8 CLIPS STRICTLY PASSED BOUNDARY AND SILENCE CRITERIA!")
