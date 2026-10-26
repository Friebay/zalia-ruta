import json
import sys
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

CLIPS = [
    {
        "id": "vaziuok_nachuj",
        "file": "original_videos/vaziuok_nachuj.mp4",
        "youtube_id": "CT0GB0vd1RI",
        "category": "profanity",
        "wav_path": "benchmark/audio/vaziuok_nachuj.wav",
        "notes": "Viral clip, 'vaziuok nachuj is cia' street argument (18.5s, heavy profanity)",
    },
    {
        "id": "maslobojevas_rimkenzo",
        "file": "original_videos/maslobojevas_rimkenzo.mp4",
        "youtube_id": "UBisblxMd5w",
        "category": "profanity",
        "wav_path": "benchmark/audio/maslobojevas_rimkenzo.wav",
        "notes": "Maslobojevas vs Rimkenzo press conference from ~1:12 to end ('kodėl vapše tavo vaikai... pydare... šūdo gabal') (32.3s)",
    },
]

def main():
    praat_dir = Path("praat_samples")
    bench_ref_dir = Path("benchmark/references")
    praat_dir.mkdir(parents=True, exist_ok=True)
    bench_ref_dir.mkdir(parents=True, exist_ok=True)
    
    runner = BenchmarkRunner(Path("benchmark"))
    
    for cfg in CLIPS:
        cid = cfg["id"]
        wav_path = Path(cfg["wav_path"])
        
        audio, sr = sf.read(str(wav_path))
        dur = len(audio) / sr
        
        # Copy to praat_samples
        dst_praat_wav = praat_dir / f"{cid}.wav"
        sf.write(str(dst_praat_wav), audio, sr)
        
        print(f"\n==========================================")
        print(f"[{cid}] ({dur:.2f}s) Transcribing with Paprika + MMS...")
        mms_words = runner.run_paprika_refined(cid, audio, sr=sr, force=True)
        acoustic_words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)
        
        text = " ".join(w.word for w in acoustic_words)
        print(f"[{cid}] Text ({len(acoustic_words)} words): \"{text}\"")
        
        first_w = acoustic_words[0] if acoustic_words else None
        last_w = acoustic_words[-1] if acoustic_words else None
        
        start_margin = first_w.start if first_w else dur
        end_margin = (dur - last_w.end) if last_w else dur
        
        print(f"[{cid}] Margins: start={start_margin:.3f}s ('{first_w.word if first_w else ''}'), end={end_margin:.3f}s ('{last_w.word if last_w else ''}')")
        
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
        print(f"[{cid}] Exported Praat TextGrid: {tg_path}")
        
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
        print(f"[{cid}] Exported reference SRT: {ref_srt_path}")
        
    # Update manifest
    manifest_path = Path("benchmark/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_ids = {c["id"] for c in manifest.get("clips", [])}
    
    for cfg in CLIPS:
        cid = cfg["id"]
        if cid not in existing_ids:
            manifest["clips"].append({
                "id": cid,
                "file": cfg["file"],
                "youtube_id": cfg["youtube_id"],
                "category": cfg["category"],
                "notes": cfg["notes"],
            })
            print(f"Added {cid} to benchmark/manifest.json")
            
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBenchmark manifest updated.")

if __name__ == "__main__":
    main()
