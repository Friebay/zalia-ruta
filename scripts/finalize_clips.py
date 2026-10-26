import json
import sys
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

MASTER_WAV = Path("benchmark/audio/B7Mygu8uiYM.wav")
YOUTUBE_ID = "B7Mygu8uiYM"

# Final exact cuts targeting complete grammatical sentences and clean acoustic pauses
FINAL_CUTS = [
    {
        "id": "karalius_prof_02",
        "category": "profanity",
        # "Pas mus achujienai padėtis. Migrantų žiauriai mažai. Ekonominė padėtis achujiena, migrantų žiauriai mažai."
        "start": 1223.50,
        "end": 1228.45,
        "notes": "Karalius reaction, 'Pas mus achujienai padėtis... achujiena' (2 profanities)",
    },
    {
        "id": "karalius_prof_03",
        "category": "profanity",
        # "nes jis piso protą dėl visko. Bet o koks skirtumas dabar tas proto pisimas?"
        "start": 378.85,
        "end": 383.95,
        "notes": "Karalius reaction, 'nes jis piso protą... proto pisimas' (2 profanities)",
    },
    {
        "id": "karalius_prof_04",
        "category": "profanity",
        # "pastoviai eina ir sako ble trūksta darbo jėgos mes galėtumėm gal ir plėstis"
        "start": 1095.75,
        "end": 1100.45,
        "notes": "Karalius reaction, 'pastoviai eina ir sako ble trūksta darbo jėgos' (1 profanity)",
    },
    {
        "id": "karalius_prof_05",
        "category": "profanity",
        # "Gerai, nepiskit proto. Jūs leistumėt žaginti ir puldinėt šimtą moterų."
        "start": 2783.15,
        "end": 2789.05,
        "notes": "Karalius reaction, 'gerai nepiskit proto jūs leistumėt žaginti' (1 profanity)",
    },
    {
        "id": "karalius_silence_01",
        "category": "silence",
        # "OK. ... domisi apie kitą esmę tyrimo."
        "start": 4864.55,
        "end": 4871.30,
        "notes": "Karalius reaction, 'OK... dominasi apie kitą esmę tyrimo' (5.1s silence gap)",
    },
    {
        "id": "karalius_silence_02",
        "category": "silence",
        # "ir tada ... dvidešimt antrais metais."
        "start": 457.25,
        "end": 464.35,
        "notes": "Karalius reaction, 'ir tada... dvidešim antrais metais' (4.8s silence gap)",
    },
    {
        "id": "karalius_silence_05",
        "category": "silence",
        # "apie 6,5 tūkstančių ... ar čia dvidešimt trečius lygina su dvidešimt ketvirtais."
        "start": 972.45,
        "end": 980.20,
        "notes": "Karalius reaction, 'apie 6,5 tūkstančio... su 24' (4.1s silence gap)",
    },
]

def main():
    print("Loading master audio...")
    master_audio, sr = sf.read(str(MASTER_WAV))
    
    praat_dir = Path("praat_samples")
    bench_audio_dir = Path("benchmark/audio")
    bench_ref_dir = Path("benchmark/references")
    
    runner = BenchmarkRunner(Path("benchmark"))
    
    for cfg in FINAL_CUTS:
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
        
        print(f"\n[{cid}] ({dur:.2f}s) Transcribing with Paprika + MMS...")
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
        print(f"[{cid}] Safe margins (>=80ms): {is_clean}")
        
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
        
        # Update manifest notes
        manifest_path = Path("benchmark/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for m_clip in manifest.get("clips", []):
            if m_clip["id"] == cid:
                m_clip["notes"] = f"{cfg['notes']} ({dur:.2f}s, words: \"{text}\")"
                break
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
