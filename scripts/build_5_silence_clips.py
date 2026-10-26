import json
import shutil
import subprocess
import sys
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

YOUTUBE_URL = "https://www.youtube.com/watch?v=7og_55op5Kg"
YOUTUBE_ID = "7og_55op5Kg"

SILENCE_CLIPS = [
    {
        "id": "karalius_silence_06",
        "start": 2953.50,
        "end": 2961.00,
        "notes": "Karalius sparse speech, 'galbūt būt' (2.8s leading silence, 3.7s trailing silence, 2 words, 7.50s)",
    },
    {
        "id": "karalius_silence_07",
        "start": 578.00,
        "end": 586.80,
        "notes": "Karalius long silence gap, '28ais bus Seimo rinkimai. Jo,' (6.3s leading silence, 5 words, 8.80s)",
    },
    {
        "id": "karalius_silence_08",
        "start": 3699.40,
        "end": 3704.50,
        "notes": "Karalius sparse speech, 'ne žinau' (4.4s trailing silence, 2 words, 5.10s)",
    },
    {
        "id": "karalius_silence_09",
        "start": 45.50,
        "end": 51.00,
        "notes": "Karalius sparse speech, 'šitie jauni žmonės, čia jum' (>1.3s leading/trailing silence, 5 words, 5.50s)",
    },
    {
        "id": "karalius_silence_10",
        "start": 681.00,
        "end": 686.50,
        "notes": "Karalius question with dead silence, 'Pala, kur mes tą žiūrėjom?' (3.6s trailing silence, 5 words, 5.50s)",
    },
]

orig_dir = Path("original_videos")
audio_dir = Path("benchmark/audio")
praat_dir = Path("praat_samples")
ref_dir = Path("benchmark/references")
manifest_path = Path("benchmark/manifest.json")

orig_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)
praat_dir.mkdir(parents=True, exist_ok=True)
ref_dir.mkdir(parents=True, exist_ok=True)

print("=== Step 1: Downloading & Slicing MP4 and 16kHz WAV ===")
for c in SILENCE_CLIPS:
    cid = c["id"]
    s = c["start"]
    e = c["end"]
    dur = round(e - s, 2)
    
    mp4_path = orig_dir / f"{cid}.mp4"
    wav_path = audio_dir / f"{cid}.wav"
    praat_wav_path = praat_dir / f"{cid}.wav"
    
    print(f"\n[{cid}] Processing {dur:.2f}s ({s:.2f}s - {e:.2f}s)...")
    
    if not mp4_path.exists():
        cmd_dl = [
            "yt-dlp",
            "--download-sections", f"*{s}-{e}",
            "--force-keyframes-at-cuts",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
            YOUTUBE_URL,
            "-o", str(mp4_path),
        ]
        subprocess.run(cmd_dl, check=True)
        print(f"Downloaded video: {mp4_path}")
    else:
        print(f"Video exists: {mp4_path}")
        
    cmd_wav = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path)
    ]
    subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    shutil.copy2(wav_path, praat_wav_path)
    print(f"Prepared audio: {wav_path} and {praat_wav_path}")

print("\n=== Step 2: Running ASR + MMS Alignment & Generating Praat TextGrids ===")
runner = BenchmarkRunner(Path("benchmark"))

for c in SILENCE_CLIPS:
    cid = c["id"]
    wav_path = audio_dir / f"{cid}.wav"
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    
    print(f"\n[{cid}] Running ASR and alignment (duration: {dur:.2f}s)...")
    mms_words = runner.run_paprika_refined(cid, audio, sr=sr, force=True)
    ac_words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)
    
    print(f"[{cid}] Recognized {len(ac_words)} words:")
    print("  " + " ".join(w.word for w in ac_words))
    
    first_w = ac_words[0] if ac_words else None
    last_w = ac_words[-1] if ac_words else None
    start_silence = first_w.start if first_w else 0.0
    end_silence = (dur - last_w.end) if last_w else 0.0
    print(f"[{cid}] Start silence: {start_silence*1000:.1f}ms | End silence: {end_silence*1000:.1f}ms")
    assert start_silence >= 0.095, f"Start silence < 100ms: {start_silence*1000:.1f}ms"
    assert end_silence >= 0.095, f"End silence < 100ms: {end_silence*1000:.1f}ms"
    assert len(ac_words) <= 7, f"Too many words for silence clip: {len(ac_words)}"
    
    def build_tier_intervals(items, duration):
        intervals = []
        curr_t = 0.0
        for it in items:
            s = max(0.0, it.start)
            e = min(duration, it.end)
            txt = it.word
            if s > curr_t:
                intervals.append((curr_t, s, ""))
            intervals.append((s, e, txt))
            curr_t = e
        if curr_t < duration:
            intervals.append((curr_t, duration, ""))
        return intervals

    tiers = [
        ("Acoustic_Refined", build_tier_intervals(ac_words, dur)),
        ("Etalonas", build_tier_intervals(ac_words, dur)),
        ("MMS_Refined", build_tier_intervals(mms_words, dur)),
    ]

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "0.0",
        f"{dur:.6f}",
        "<exists>",
        f"{len(tiers)}",
    ]

    for name, intervals in tiers:
        lines.extend([
            '"IntervalTier"',
            f'"{name}"',
            "0.0",
            f"{dur:.6f}",
            f"{len(intervals)}",
        ])
        for s, e, txt in intervals:
            lines.extend([
                f"{s:.6f}",
                f"{e:.6f}",
                f'"{txt}"',
            ])

    tg_path = praat_dir / f"{cid}.TextGrid"
    tg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created Praat TextGrid: {tg_path}")
    
    srt_lines = []
    for idx, w in enumerate(ac_words, 1):
        s_ms = int(round(w.start * 1000))
        e_ms = int(round(w.end * 1000))
        
        sh, rem = divmod(s_ms, 3600000)
        sm, rem = divmod(rem, 60000)
        ss, smilli = divmod(rem, 1000)
        
        eh, rem = divmod(e_ms, 3600000)
        em, rem = divmod(rem, 60000)
        es, emilli = divmod(rem, 1000)
        
        srt_lines.append(str(idx))
        srt_lines.append(f"{sh:02d}:{sm:02d}:{ss:02d},{smilli:03d} --> {eh:02d}:{em:02d}:{es:02d},{emilli:03d}")
        srt_lines.append(w.word)
        srt_lines.append("")
        
    srt_path = ref_dir / f"{cid}.srt"
    srt_path.write_text("\n".join(srt_lines) + "\n", encoding="utf-8")
    print(f"Created baseline reference: {srt_path}")

print("\n=== Step 3: Updating benchmark/manifest.json ===")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
existing_ids = {c["id"]: c for c in manifest.get("clips", [])}

for c in SILENCE_CLIPS:
    cid = c["id"]
    entry = {
        "id": cid,
        "file": f"original_videos/{cid}.mp4",
        "youtube_id": YOUTUBE_ID,
        "category": "silence",
        "notes": c["notes"]
    }
    if cid in existing_ids:
        idx = next(i for i, x in enumerate(manifest["clips"]) if x["id"] == cid)
        manifest["clips"][idx] = entry
    else:
        manifest["clips"].append(entry)
    print(f"Registered {cid} in manifest.json")

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Manifest updated successfully.")
