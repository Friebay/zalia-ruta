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

CLIPS = [
    {
        "id": "karalius_prof_07",
        "start": 4275.00,
        "end": 4282.00,
        "notes": "Karalius reaction, 'What the fuck? Kodėl? Kas jum yra' (7.00s, ~240ms start / >1000ms end silence)",
    },
    {
        "id": "karalius_prof_08",
        "start": 3686.15,
        "end": 3693.20,
        "notes": "Karalius commentary, 'man net nereikia nieko sakyt... blet blet kaip užpisa' (7.05s, ~150ms start / ~200ms end silence)",
    },
    {
        "id": "karalius_prof_09",
        "start": 4127.90,
        "end": 4134.30,
        "notes": "Karalius reaction, 'žodžiu faking hel ten aš nežinau' (6.40s, ~310ms start / ~270ms end silence)",
    },
    {
        "id": "karalius_prof_10",
        "start": 4002.55,
        "end": 4008.50,
        "notes": "Karalius commentary, 'visiškai su šūdais tave maišys' (5.95s, ~190ms start / >1000ms end silence)",
    },
    {
        "id": "karalius_prof_11",
        "start": 4205.60,
        "end": 4212.35,
        "notes": "Karalius rant, 'išrinks kokį debilą, kuris žadės' (6.75s, ~275ms start / ~310ms end silence)",
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
for c in CLIPS:
    cid = c["id"]
    s = c["start"]
    e = c["end"]
    dur = round(e - s, 3)
    
    mp4_path = orig_dir / f"{cid}.mp4"
    wav_path = audio_dir / f"{cid}.wav"
    praat_wav_path = praat_dir / f"{cid}.wav"
    
    print(f"\n[{cid}] Processing {dur:.2f}s ({s:.2f}s - {e:.2f}s)...")
    
    # 1. Download video section
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
        
    # 2. Extract 16kHz mono WAV
    cmd_wav = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path)
    ]
    subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Copy to praat_samples
    shutil.copy2(wav_path, praat_wav_path)
    print(f"Prepared audio: {wav_path} and {praat_wav_path}")

print("\n=== Step 2: Running ASR + MMS Alignment & Generating Praat TextGrids ===")
runner = BenchmarkRunner(Path("benchmark"))

for c in CLIPS:
    cid = c["id"]
    wav_path = audio_dir / f"{cid}.wav"
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    
    print(f"\n[{cid}] Running ASR and alignment (duration: {dur:.2f}s)...")
    mms_words = runner.run_paprika_refined(cid, audio, sr=sr, force=True)
    ac_words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)
    
    print(f"[{cid}] Recognized {len(ac_words)} words:")
    print("  " + " ".join(w.word for w in ac_words))
    
    # Verify non-speech silence margins at start and end
    first_w = ac_words[0] if ac_words else None
    last_w = ac_words[-1] if ac_words else None
    start_silence = first_w.start if first_w else 0.0
    end_silence = (dur - last_w.end) if last_w else 0.0
    print(f"[{cid}] Start silence: {start_silence*1000:.1f}ms | End silence: {end_silence*1000:.1f}ms")
    assert start_silence >= 0.095, f"Start silence < 100ms: {start_silence*1000:.1f}ms"
    assert end_silence >= 0.095, f"End silence < 100ms: {end_silence*1000:.1f}ms"
    
    # Export Praat TextGrid with 3 tiers:
    # 1. Acoustic_Refined (user adjusts this in Praat)
    # 2. Etalonas
    # 3. MMS_Refined
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
    
    # Generate baseline SRT in benchmark/references/<cid>.srt (1 word per cue)
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

for c in CLIPS:
    cid = c["id"]
    entry = {
        "id": cid,
        "file": f"original_videos/{cid}.mp4",
        "youtube_id": YOUTUBE_ID,
        "category": "profanity",
        "notes": c["notes"]
    }
    if cid in existing_ids:
        # Update existing
        idx = next(i for i, x in enumerate(manifest["clips"]) if x["id"] == cid)
        manifest["clips"][idx] = entry
    else:
        manifest["clips"].append(entry)
    print(f"Registered {cid} in manifest.json")

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Manifest updated successfully.")
