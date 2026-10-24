import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner

YOUTUBE_URL = "https://www.youtube.com/watch?v=nGMb_oEVGJk"

# Exact boundaries placed strictly in acoustic silence between phrases:
# Zero trailing word fragments (no 'it', no 'atliko', no 'jinai', no 'ir bus', no 'rašė')
CLIPS_CONFIG = [
    {
        "id": "orlauskas_prof_01",
        "start": 1178.50,
        "end": 1183.45,
        "quote": "bet tiesiog vėlgi į ką pėliuoja? Tų dalbajobų, nes pagrinde ten visi antivakceriai buvo ir panašiai",
        "profanity": "dalbajobų",
        "notes": "Orlauskas commentary, 'tų dalbajobų' (4.95s, clean start in pause before 'bet')",
    },
    {
        "id": "orlauskas_prof_02",
        "start": 1278.75,
        "end": 1281.93,
        "quote": "Kaip jis susitaria, tarkim, su Rusija, kad pistų protą ant Lietuvos. Man šiaip įdomu.",
        "profanity": "pistų",
        "notes": "Orlauskas commentary, 'kad pistų protą' (3.18s, clean start in pause before 'Kaip')",
    },
    {
        "id": "orlauskas_prof_03",
        "start": 1323.65,
        "end": 1328.65,
        "quote": "į tą Europos žmogaus teisių teismą ir ten ieško kaip prisipist ant valstybių ar dar kažką.",
        "profanity": "prisipist",
        "notes": "Orlauskas commentary, 'kaip prisipist ant valstybių' (5.00s, clean start before 'į')",
    },
    {
        "id": "orlauskas_prof_04",
        "start": 1544.55,
        "end": 1548.20,
        "quote": "O iš esmės kurvės visi šitie žvegiantys, kad ojojoj, Europa dega.",
        "profanity": "kurvės",
        "notes": "Orlauskas commentary, 'o iš esmės kurvės visi' (3.65s, clean start in pause before 'O')",
    },
    {
        "id": "orlauskas_prof_05",
        "start": 2422.63,
        "end": 2425.55,
        "quote": "Tai čia ne ne šitas ble kažkoks buvo.",
        "profanity": "ble",
        "notes": "Orlauskas commentary, 'ne šitas ble kažkoks' (2.92s, clean start in pause before 'Tai')",
    },
]


def download_audio_sections():
    praat_dir = Path("praat_samples")
    bench_audio_dir = Path("benchmark/audio")
    praat_dir.mkdir(parents=True, exist_ok=True)
    bench_audio_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        start_s = cfg["start"]
        end_s = cfg["end"]
        dur = round(end_s - start_s, 2)

        dst_m4a = praat_dir / f"{cid}.m4a"
        dst_wav = praat_dir / f"{cid}.wav"
        bench_wav = bench_audio_dir / f"{cid}.wav"

        print(f"\n--- [{cid}] Downloading audio section {start_s}s - {end_s}s ({dur}s) ---")
        cmd_dl = [
            "yt-dlp",
            "--download-sections", f"*{start_s}-{end_s}",
            "--force-keyframes-at-cuts",
            "-f", "ba[ext=m4a]/bestaudio/best",
            YOUTUBE_URL,
            "-o", str(dst_m4a),
        ]
        subprocess.run(cmd_dl, check=True)

        print(f"[{cid}] Converting to 16kHz mono WAV...")
        cmd_wav = [
            "ffmpeg", "-y", "-i", str(dst_m4a),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(dst_wav),
        ]
        subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Copy to benchmark/audio
        subprocess.run(["cmd", "/c", "copy", "/y", str(dst_wav), str(bench_wav)], stdout=subprocess.DEVNULL, check=True)

        # Clean up temporary m4a
        if dst_m4a.exists():
            dst_m4a.unlink()

        print(f"[{cid}] Ready: {dst_wav} ({dur}s)")


def transcribe_and_export_textgrids():
    praat_dir = Path("praat_samples")
    bench_dir = Path("benchmark")
    runner = BenchmarkRunner(bench_dir)

    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        wav_path = praat_dir / f"{cid}.wav"
        audio, sr = sf.read(str(wav_path))
        dur = len(audio) / sr

        print(f"\n[{cid}] Transcribing with Paprika Whisper v3 + MMS + Acoustic Refinement (force=True)...")
        mms_words = runner.run_paprika_refined(cid, audio, sr=sr, force=True)
        acoustic_words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)

        print(f"[{cid}] Acoustic words ({len(acoustic_words)}): {' '.join(w.word for w in acoustic_words)}")

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
        print(f"[{cid}] Exported Praat TextGrid: {tg_path}")


def update_manifest():
    manifest_path = Path("benchmark/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    existing_map = {c["id"]: c for c in manifest.get("clips", [])}

    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        if cid in existing_map:
            existing_map[cid]["notes"] = cfg["notes"]
        else:
            manifest["clips"].append({
                "id": cid,
                "file": f"original_videos/{cid}.mp4",
                "youtube_id": "nGMb_oEVGJk",
                "category": "profanity",
                "notes": cfg["notes"],
            })
            print(f"Added {cid} to benchmark/manifest.json")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Manifest updated.")


if __name__ == "__main__":
    print("Step 1: Downloading 5 clean audio sections without word bleeding...")
    download_audio_sections()
    print("\nStep 2: Transcribing with paprika_acoustic & exporting TextGrids...")
    transcribe_and_export_textgrids()
    print("\nStep 3: Updating benchmark manifest...")
    update_manifest()
    print("\nAll 5 clean clips prepared successfully!")
