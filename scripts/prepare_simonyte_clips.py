import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

MASTER_WAV = Path("benchmark/audio/B7Mygu8uiYM.wav")
YOUTUBE_ID = "B7Mygu8uiYM"

# Initial candidate intervals (approx 7.0 seconds each)
CANDIDATES = [
    # 5 Profanity Clips
    {
        "id": "karalius_prof_02",
        "category": "profanity",
        "approx_start": 1222.0,
        "approx_end": 1229.0,
        "notes": "Karalius reaction, 'achujienai padėtis... achujiena' (2 profanities)",
    },
    {
        "id": "karalius_prof_03",
        "category": "profanity",
        "approx_start": 378.0,
        "approx_end": 385.0,
        "notes": "Karalius reaction, 'piso protą... proto pisimas' (2 profanities)",
    },
    {
        "id": "karalius_prof_04",
        "category": "profanity",
        "approx_start": 1094.5,
        "approx_end": 1101.5,
        "notes": "Karalius reaction, 'Blet, trūksta darbo jėgos' (1 profanity)",
    },
    {
        "id": "karalius_prof_05",
        "category": "profanity",
        "approx_start": 2782.0,
        "approx_end": 2789.0,
        "notes": "Karalius reaction, 'nepiskit proto' (1 profanity)",
    },
    {
        "id": "karalius_prof_06",
        "category": "profanity",
        "approx_start": 3950.0,
        "approx_end": 3957.0,
        "notes": "Karalius reaction, 'tuos dalbajobus' (1 profanity)",
    },
    # 5 Silence / Gap Clips
    {
        "id": "karalius_silence_01",
        "category": "silence",
        "approx_start": 4864.2,
        "approx_end": 4871.2,
        "notes": "Karalius reaction, 'OK... dominasi' with 5.1s silence gap",
    },
    {
        "id": "karalius_silence_02",
        "category": "silence",
        "approx_start": 457.3,
        "approx_end": 464.3,
        "notes": "Karalius reaction, 'ir tada... 22 metais' with 4.8s silence gap",
    },
    {
        "id": "karalius_silence_03",
        "category": "silence",
        "approx_start": 3062.0,
        "approx_end": 3069.0,
        "notes": "Karalius reaction, 'Labas, Emi... labas. Labas.' with 4.8s silence gap",
    },
    {
        "id": "karalius_silence_04",
        "category": "silence",
        "approx_start": 4447.0,
        "approx_end": 4454.0,
        "notes": "Karalius reaction, 'terminas yra... OK. Gerai.' with 4.3s silence gap",
    },
    {
        "id": "karalius_silence_05",
        "category": "silence",
        "approx_start": 972.3,
        "approx_end": 979.3,
        "notes": "Karalius reaction, 'apie 6,5 tūkstančio... Ai, čia 23' with 4.1s silence gap",
    },
]

def find_acoustic_silence(audio: np.ndarray, sr: int, target_sec: float, search_window_sec: float = 1.0, forward: bool = True) -> float:
    """Finds the lowest RMS energy point within search_window_sec around target_sec."""
    center_idx = int(target_sec * sr)
    window_samples = int(search_window_sec * sr)
    
    if forward:
        start_idx = center_idx
        end_idx = min(len(audio), center_idx + window_samples)
    else:
        start_idx = max(0, center_idx - window_samples)
        end_idx = center_idx
        
    slice_data = audio[start_idx:end_idx]
    if len(slice_data) == 0:
        return target_sec
        
    # Frame size for RMS (20ms)
    frame_len = int(0.02 * sr)
    hop_len = int(0.01 * sr)
    
    min_rms = float("inf")
    best_offset = 0
    
    for i in range(0, len(slice_data) - frame_len, hop_len):
        frame = slice_data[i:i + frame_len]
        rms = np.sqrt(np.mean(frame**2))
        if rms < min_rms:
            min_rms = rms
            best_offset = i
            
    best_sample = start_idx + best_offset + frame_len // 2
    return round(best_sample / sr, 3)

def slice_and_transcribe():
    print(f"Loading master audio: {MASTER_WAV}...")
    master_audio, sr = sf.read(str(MASTER_WAV))
    total_dur = len(master_audio) / sr
    print(f"Master audio loaded: {total_dur:.1f}s ({total_dur/60:.1f} min), sample rate={sr}")
    
    praat_dir = Path("praat_samples")
    bench_audio_dir = Path("benchmark/audio")
    bench_ref_dir = Path("benchmark/references")
    praat_dir.mkdir(parents=True, exist_ok=True)
    bench_audio_dir.mkdir(parents=True, exist_ok=True)
    bench_ref_dir.mkdir(parents=True, exist_ok=True)
    
    runner = BenchmarkRunner(Path("benchmark"))
    
    verified_clips = []
    
    for cfg in CANDIDATES:
        cid = cfg["id"]
        appr_s = cfg["approx_start"]
        appr_e = cfg["approx_end"]
        
        # Snap start and end to acoustic valleys (silence gaps)
        # For start, search in [appr_s - 0.6s, appr_s + 0.6s]
        # For end, search in [appr_e - 0.6s, appr_e + 0.6s]
        s_valley = find_acoustic_silence(master_audio, sr, appr_s, search_window_sec=0.8, forward=False)
        e_valley = find_acoustic_silence(master_audio, sr, appr_e, search_window_sec=0.8, forward=True)
        
        # Ensure duration is around 6.5s to 7.8s
        dur = round(e_valley - s_valley, 3)
        if dur < 6.0:
            e_valley = round(s_valley + 6.8, 3)
            dur = round(e_valley - s_valley, 3)
            
        print(f"\n==========================================")
        print(f"[{cid}] Slicing {s_valley:.3f}s -> {e_valley:.3f}s (duration: {dur:.3f}s)")
        
        s_idx = int(s_valley * sr)
        e_idx = int(e_valley * sr)
        clip_audio = master_audio[s_idx:e_idx]
        
        clip_wav = praat_dir / f"{cid}.wav"
        bench_wav = bench_audio_dir / f"{cid}.wav"
        sf.write(str(clip_wav), clip_audio, sr)
        sf.write(str(bench_wav), clip_audio, sr)
        
        # Run Paprika Whisper v3 + MMS Forced Alignment
        print(f"[{cid}] Transcribing with Paprika + MMS Forced Alignment...")
        mms_words = runner.run_paprika_refined(cid, clip_audio, sr=sr, force=True)
        acoustic_words = runner.run_paprika_acoustic(cid, clip_audio, sr=sr, force=True)
        
        transcribed_text = " ".join(w.word for w in acoustic_words)
        print(f"[{cid}] Result ({len(acoustic_words)} words): {transcribed_text}")
        
        # Boundary checks
        if len(acoustic_words) > 0:
            first_w = acoustic_words[0]
            last_w = acoustic_words[-1]
            first_margin = first_w.start
            last_margin = dur - last_w.end
            print(f"[{cid}] Boundary margins: start_margin={first_margin:.3f}s ('{first_w.word}'), end_margin={last_margin:.3f}s ('{last_w.word}')")
            
            # Check for cut-off warning
            cut_start = first_margin < 0.05
            cut_end = last_margin < 0.05
            if cut_start:
                print(f"  [WARNING] First word '{first_w.word}' is very close to start ({first_w.start:.3f}s)")
            if cut_end:
                print(f"  [WARNING] Last word '{last_w.word}' is very close to end (ends at {last_w.end:.3f}s, clip is {dur:.3f}s)")
            if not cut_start and not cut_end:
                print(f"  [CLEAN] Clip has safe acoustic silence at both boundaries!")
        else:
            print(f"  [NOTE] 0 words transcribed (pure silence)!")
            
        # Export Praat TextGrid
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
            for s, e, txt in intervals:
                lines.extend([
                    f"{s:.6f}",
                    f"{e:.6f}",
                    f'"{txt}"',
                ])

        tg_path = praat_dir / f"{cid}.TextGrid"
        tg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        
        # Export initial benchmark reference SRT (1 word per cue)
        def sec_to_srt(s: float) -> str:
            hrs = int(s // 3600)
            mins = int((s % 3600) // 60)
            secs = int(s % 60)
            ms = int(round((s - int(s)) * 1000))
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

        srt_lines = []
        for idx, w in enumerate(acoustic_words, 1):
            srt_lines.append(str(idx))
            srt_lines.append(f"{sec_to_srt(w.start)} --> {sec_to_srt(w.end)}")
            srt_lines.append(w.word)
            srt_lines.append("")
            
        ref_srt_path = bench_ref_dir / f"{cid}.srt"
        ref_srt_path.write_text("\n".join(srt_lines) + "\n", encoding="utf-8")
        
        cfg["actual_start"] = s_valley
        cfg["actual_end"] = e_valley
        cfg["actual_duration"] = dur
        cfg["words"] = transcribed_text
        verified_clips.append(cfg)
        
    # Update benchmark/manifest.json
    manifest_path = Path("benchmark/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_ids = {c["id"] for c in manifest.get("clips", [])}
    
    for vc in verified_clips:
        cid = vc["id"]
        if cid not in existing_ids:
            manifest["clips"].append({
                "id": cid,
                "file": f"original_videos/{cid}.mp4",
                "youtube_id": YOUTUBE_ID,
                "category": vc["category"],
                "notes": f"{vc['notes']} ({vc['actual_duration']:.2f}s, words: {vc['words']})"
            })
            print(f"Added {cid} to benchmark/manifest.json")
            
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBenchmark manifest updated successfully!")

if __name__ == "__main__":
    slice_and_transcribe()
