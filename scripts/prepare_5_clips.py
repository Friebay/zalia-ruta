import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner


SOURCE_VIDEO = Path(
    "original_videos/Tapinas apie nesusitvarkymą su dronais, blogiausią finansų ministrą ir riaušes ｜ Karalius Reaguoja_dSObkwfw2MA.mp4"
)

CLIPS_CONFIG = [
    {
        "id": "karalius_10s_01",
        "start": 248.5,
        "end": 257.8,
        "notes": "Karalius fast speech, opinions on spending (9.3s)",
    },
    {
        "id": "karalius_10s_02",
        "start": 600.3,
        "end": 610.6,
        "notes": "Karalius commentary on Tapinas influence (10.3s)",
    },
    {
        "id": "karalius_10s_03",
        "start": 1045.0,
        "end": 1055.2,
        "notes": "Karalius US politics commentary (10.2s)",
    },
    {
        "id": "karalius_10s_04",
        "start": 1500.0,
        "end": 1509.8,
        "notes": "Karalius sparse deliberate speech (9.8s)",
    },
    {
        "id": "karalius_10s_05",
        "start": 2099.2,
        "end": 2109.8,
        "notes": "Karalius LFF and basketball clubs discussion (10.6s)",
    },
]


def extract_clips():
    orig_dir = Path("original_videos")
    audio_dir = Path("benchmark/audio")
    praat_dir = Path("praat_samples")
    orig_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    praat_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        start_s = cfg["start"]
        dur_s = round(cfg["end"] - cfg["start"], 2)

        dst_mp4 = orig_dir / f"{cid}.mp4"
        dst_wav = audio_dir / f"{cid}.wav"
        dst_praat_wav = praat_dir / f"{cid}.wav"

        print(f"\n--- [{cid}] Extracting {dur_s}s (from {start_s}s) ---")
        # 1. Extract MP4 video
        cmd_mp4 = [
            "ffmpeg", "-y", "-ss", str(start_s), "-i", str(SOURCE_VIDEO),
            "-t", str(dur_s), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", str(dst_mp4)
        ]
        subprocess.run(cmd_mp4, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 2. Extract 16kHz mono WAV
        cmd_wav = [
            "ffmpeg", "-y", "-i", str(dst_mp4),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(dst_wav)
        ]
        subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Copy to praat_samples
        subprocess.run(["cmd", "/c", "copy", "/y", str(dst_wav), str(dst_praat_wav)], stdout=subprocess.DEVNULL, check=True)
        print(f"[{cid}] Extracted video: {dst_mp4} and audio: {dst_wav}")


def transcribe_and_export_praat():
    bench_dir = Path("benchmark")
    praat_dir = Path("praat_samples")
    runner = BenchmarkRunner(bench_dir)

    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        wav_path = bench_dir / "audio" / f"{cid}.wav"
        audio, sr = sf.read(str(wav_path))
        dur = len(audio) / sr

        print(f"\n[{cid}] Running best method (ASR + MMS + Acoustic Refinement)...")
        # Runs paprika_refined (MMS) and paprika_acoustic (Acoustic Refinement)
        mms_words = runner.run_paprika_refined(cid, audio, sr=sr, force=True)
        acoustic_words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)

        print(f"[{cid}] Recognized {len(acoustic_words)} words:")
        print("  " + " ".join(w.word for w in acoustic_words))

        # Build Praat TextGrid with two tiers:
        # 1. Etalonas (starts with acoustic_words, ready for user editing)
        # 2. MMS_Refined (for visual comparison)
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
        print(f"[{cid}] Created Praat TextGrid: {tg_path}")


def update_manifest():
    manifest_path = Path("benchmark/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_ids = {c["id"] for c in manifest.get("clips", [])}

    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        if cid not in existing_ids:
            manifest["clips"].append({
                "id": cid,
                "file": f"original_videos/{cid}.mp4",
                "category": "speech",
                "notes": cfg["notes"]
            })
            print(f"Added {cid} to manifest.json")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Manifest updated successfully.")


if __name__ == "__main__":
    print("Step 1: Extracting 5 clips...")
    extract_clips()
    print("\nStep 2: Transcribing with best pipeline & building Praat TextGrids...")
    transcribe_and_export_praat()
    print("\nStep 3: Updating benchmark manifest...")
    update_manifest()
    print("\nALL 5 CLIPS PREPARED SUCCESSFULLY!")
