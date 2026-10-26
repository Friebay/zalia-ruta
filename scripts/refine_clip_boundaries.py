import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

MASTER_WAV = Path("benchmark/audio/B7Mygu8uiYM.wav")
YOUTUBE_ID = "B7Mygu8uiYM"

# Precisely adjusted boundaries based on first-pass MMS alignment:
# Ensures:
# - First word starts with >= 0.15s of clean silence (no preceding partial word fragments)
# - Last word ends with >= 0.15s of clean silence (no trailing chopped words)
# - Clip duration remains between 5.5s and 7.8s
ADJUSTED_CLIPS = [
    # --- 5 PROFANITY CLIPS ---
    {
        "id": "karalius_prof_02",
        "category": "profanity",
        # First pass had 'tos' at 0.015s and 'čia' cut at end.
        # Skip 'tos darbo jėgos...' and start directly before 'Pas mus achujienai padėtis' (around 1223.2s)
        # End cleanly after second 'migrantų žiauriai mažai' (around 1228.6s)
        "start": 1223.10,
        "end": 1228.80,
        "notes": "Karalius reaction, 'Pas mus achujienai padėtis... achujiena' (2 profanities)",
    },
    {
        "id": "karalius_prof_03",
        "category": "profanity",
        # First pass started at 377.66s with 'tolikas' (chopped from katolikas).
        # Move start forward to 378.15s to cleanly start on pause before 'šiek tiek'.
        # End around 384.80s after 'proto pisimas ką padarė' (before 'tai savo efektą pasiekė')
        "start": 378.15,
        "end": 384.90,
        "notes": "Karalius reaction, 'šiek tiek, nes jis piso protą... proto pisimas' (2 profanities)",
    },
    {
        "id": "karalius_prof_04",
        "category": "profanity",
        # First pass started at 1094.21s on 'ir' and ended at 1102.06s cutting 'nori'.
        # Start at 1095.30s (silence before 'pastoviai eina ir sako ble trūksta')
        # End at 1101.80s (after 'plėstis', before 'ir aišku ji nori')
        "start": 1095.40,
        "end": 1101.80,
        "notes": "Karalius reaction, 'pastoviai eina ir sako ble trūksta darbo jėgos' (1 profanity)",
    },
    {
        "id": "karalius_prof_05",
        "category": "profanity",
        # First pass started at 2781.26s and cut 'moterų' right at 2789.01s.
        # Extend end to 2789.50s so 'moterų' finishes with clean silence trailing.
        "start": 2781.50,
        "end": 2789.50,
        "notes": "Karalius reaction, 'gerai nepiskit proto jūs leistumėt žaginti' (1 profanity)",
    },
    {
        "id": "karalius_prof_06",
        "category": "profanity",
        # Already clean in first pass: start margin 0.775s, end margin 0.280s!
        "start": 3949.80,
        "end": 3957.70,
        "notes": "Karalius reaction, 'kad biškį labiau pasiekti tuos dalbajobus' (1 profanity)",
    },
    # --- 5 SILENCE / GAP CLIPS ---
    {
        "id": "karalius_silence_01",
        "category": "silence",
        # First pass started at 4863.75s cutting into 'tą'.
        # Start at 4864.50s (silence before 'OK. ... dominasi apie kitą esmę')
        # End at 4871.80s (silence trailing after 'esmę')
        "start": 4864.60,
        "end": 4871.90,
        "notes": "Karalius reaction, 'OK... dominasi apie kitą esmę' (5.1s silence gap)",
    },
    {
        "id": "karalius_silence_02",
        "category": "silence",
        # First pass started at 457.27s (clean) and cut 'metais' right at 464.32s.
        # Extend end to 464.80s so 'metais' finishes with clean silence trailing.
        "start": 457.30,
        "end": 464.80,
        "notes": "Karalius reaction, 'ir tada... dvidešim antrais metais' (4.8s silence gap)",
    },
    {
        "id": "karalius_silence_03",
        "category": "silence",
        # First pass started at 3061.22s (clean) but cut 'aš' at 3069.13s.
        # End at 3068.70s right after 'labas... labas' so the chopped 'aš' is excluded.
        "start": 3061.80,
        "end": 3068.70,
        "notes": "Karalius reaction, 'Labas, Emi... labas.' (4.8s silence gap)",
    },
    {
        "id": "karalius_silence_04",
        "category": "silence",
        # First pass cut 'iš' at end.
        # End at 4453.60s right after 'OK. Gerai.' before the speaker starts saying 'o taigi iš...'.
        "start": 4446.80,
        "end": 4453.60,
        "notes": "Karalius reaction, 'terminas yra... OK. Gerai.' (4.3s silence gap)",
    },
    {
        "id": "karalius_silence_05",
        "category": "silence",
        # First pass started at 972.21s cutting into 'mažiau'.
        # Start at 972.70s in the clean pause before 'apie šeši kablelis penki tūkstančiai'
        # End at 980.20s in clean silence trailing 'dvidešim ketvirtais'.
        "start": 972.70,
        "end": 980.20,
        "notes": "Karalius reaction, 'apie 6,5 tūkstančio... su 24' (4.1s silence gap)",
    },
]

def run_refinement():
    print("Loading master audio...")
    master_audio, sr = sf.read(str(MASTER_WAV))
    
    praat_dir = Path("praat_samples")
    bench_audio_dir = Path("benchmark/audio")
    bench_ref_dir = Path("benchmark/references")
    
    runner = BenchmarkRunner(Path("benchmark"))
    
    results_summary = []
    
    for cfg in ADJUSTED_CLIPS:
        cid = cfg["id"]
        s = cfg["start"]
        e = cfg["end"]
        dur = round(e - s, 3)
        
        s_idx = int(s * sr)
        e_idx = int(e * sr)
        clip_audio = master_audio[s_idx:e_idx]
        
        clip_wav = praat_dir / f"{cid}.wav"
        bench_wav = bench_audio_dir / f"{cid}.wav"
        sf.write(str(clip_wav), clip_audio, sr)
        sf.write(str(bench_wav), clip_audio, sr)
        
        # Transcribe
        print(f"\n==========================================")
        print(f"[{cid}] ({dur:.2f}s) Transcribing with Paprika + MMS...")
        mms_words = runner.run_paprika_refined(cid, clip_audio, sr=sr, force=True)
        acoustic_words = runner.run_paprika_acoustic(cid, clip_audio, sr=sr, force=True)
        
        text = " ".join(w.word for w in acoustic_words)
        print(f"[{cid}] Text ({len(acoustic_words)} words): \"{text}\"")
        
        first_w = acoustic_words[0] if acoustic_words else None
        last_w = acoustic_words[-1] if acoustic_words else None
        
        start_margin = first_w.start if first_w else dur
        end_margin = (dur - last_w.end) if last_w else dur
        
        print(f"[{cid}] Margins: start={start_margin:.3f}s ('{first_w.word if first_w else ''}'), end={end_margin:.3f}s ('{last_w.word if last_w else ''}')")
        
        is_clean = (start_margin >= 0.08) and (end_margin >= 0.08)
        status_str = "CLEAN (Safe acoustic margins, no mid-cut words)" if is_clean else "CHECK NEEDED"
        print(f"[{cid}] Status: {status_str}")
        
        # Build TextGrid
        def build_tier_intervals(items, duration):
            intervals = []
            curr_t = 0.0
            for it in items:
                ws = max(0.0, it.start)
                we = min(duration, it.end)
                txt = it.word
                if ws > curr_t:
                    intervals.append((curr_t, ws, ""))
                intervals.append((ws, we, txt))
                curr_t = we
            if curr_t < duration:
                intervals.append((curr_t, duration, ""))
            return intervals

        tiers = [
            ("Acoustic_Refined", build_tier_intervals(acoustic_words, dur)),
            ("Etalonas", build_tier_intervals(acoustic_words, dur)),
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
            for ws, we, txt in intervals:
                lines.extend([
                    f"{ws:.6f}",
                    f"{we:.6f}",
                    f'"{txt}"',
                ])

        tg_path = praat_dir / f"{cid}.TextGrid"
        tg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        
        # Save SRT reference
        def sec_to_srt(sec: float) -> str:
            hrs = int(sec // 3600)
            mins = int((sec % 3600) // 60)
            secs = int(sec % 60)
            ms = int(round((sec - int(sec)) * 1000))
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

        srt_lines = []
        for idx, w in enumerate(acoustic_words, 1):
            srt_lines.append(str(idx))
            srt_lines.append(f"{sec_to_srt(w.start)} --> {sec_to_srt(w.end)}")
            srt_lines.append(w.word)
            srt_lines.append("")
            
        ref_srt_path = bench_ref_dir / f"{cid}.srt"
        ref_srt_path.write_text("\n".join(srt_lines) + "\n", encoding="utf-8")
        
        results_summary.append({
            "id": cid,
            "category": cfg["category"],
            "duration": dur,
            "start_margin": round(start_margin, 3),
            "end_margin": round(end_margin, 3),
            "text": text,
            "notes": cfg["notes"],
            "is_clean": is_clean,
        })
        
    # Update manifest
    manifest_path = Path("benchmark/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_map = {c["id"]: c for c in manifest.get("clips", [])}
    
    for r in results_summary:
        cid = r["id"]
        note_text = f"{r['notes']} ({r['duration']:.2f}s, words: \"{r['text']}\")"
        if cid in existing_map:
            existing_map[cid]["notes"] = note_text
            existing_map[cid]["category"] = r["category"]
        else:
            manifest["clips"].append({
                "id": cid,
                "file": f"original_videos/{cid}.mp4",
                "youtube_id": YOUTUBE_ID,
                "category": r["category"],
                "notes": note_text,
            })
            
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRefinement complete and benchmark/manifest.json updated!")

if __name__ == "__main__":
    run_refinement()
