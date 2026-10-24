import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import BenchmarkRunner


YOUTUBE_URL = "https://www.youtube.com/watch?v=PZ2gyZC5Kuw"

PROFANE_CLIPS = [
    {
        "id": "karalius_prof_01",
        "start": 577.5,
        "end": 583.5,
        "notes": "Karalius political commentary, 'laižeką pisa' (6.0s)",
    },
    {
        "id": "karalius_prof_02",
        "start": 617.5,
        "end": 623.5,
        "notes": "Karalius coalition discussion, 'apipisa visus' (6.0s)",
    },
    {
        "id": "karalius_prof_03",
        "start": 713.0,
        "end": 721.0,
        "notes": "Karalius LRT audits rant, 'LRT supist... alertą supistie' (8.0s)",
    },
    {
        "id": "karalius_prof_04",
        "start": 1059.5,
        "end": 1065.5,
        "notes": "Karalius party reaction, 'tiesiog pizdiec yra' (6.0s)",
    },
    {
        "id": "karalius_prof_05",
        "start": 1254.5,
        "end": 1260.5,
        "notes": "Karalius čekiukai commentary, 'su pisintu dyzeliu' (6.0s)",
    },
    {
        "id": "karalius_prof_06",
        "start": 2192.5,
        "end": 2198.5,
        "notes": "Karalius rant on politicians, 'nauša suka yra padugnės' (6.0s)",
    },
]


def download_and_extract():
    orig_dir = Path("original_videos")
    audio_dir = Path("benchmark/audio")
    praat_dir = Path("praat_samples")
    orig_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    praat_dir.mkdir(parents=True, exist_ok=True)

    for cfg in PROFANE_CLIPS:
        cid = cfg["id"]
        start_s = cfg["start"]
        end_s = cfg["end"]
        dst_mp4 = orig_dir / f"{cid}.mp4"
        dst_wav = audio_dir / f"{cid}.wav"
        dst_praat_wav = praat_dir / f"{cid}.wav"

        # 1. Download section if not already downloaded
        if not dst_mp4.exists():
            print(f"\n--- [{cid}] Downloading section {start_s}s - {end_s}s from YouTube ---")
            cmd_dl = [
                "yt-dlp",
                "--download-sections", f"*{start_s}-{end_s}",
                "--force-keyframes-at-cuts",
                "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
                YOUTUBE_URL,
                "-o", str(dst_mp4),
            ]
            subprocess.run(cmd_dl, check=True)
        else:
            print(f"[{cid}] Video already downloaded: {dst_mp4}")

        # 2. Extract 16kHz mono WAV
        if not dst_wav.exists():
            print(f"[{cid}] Extracting 16kHz mono WAV...")
            cmd_wav = [
                "ffmpeg", "-y", "-i", str(dst_mp4),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(dst_wav)
            ]
            subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        subprocess.run(["cmd", "/c", "copy", "/y", str(dst_wav), str(dst_praat_wav)], stdout=subprocess.DEVNULL, check=True)


def transcribe_and_export_praat():
    bench_dir = Path("benchmark")
    praat_dir = Path("praat_samples")
    runner = BenchmarkRunner(bench_dir)

    for cfg in PROFANE_CLIPS:
        cid = cfg["id"]
        wav_path = bench_dir / "audio" / f"{cid}.wav"
        audio, sr = sf.read(str(wav_path))
        dur = len(audio) / sr

        print(f"\n[{cid}] Transcribing with best pipeline (ASR + MMS + Acoustic Refinement)...")
        mms_words = runner.run_paprika_refined(cid, audio, sr=sr, force=True)
        acoustic_words = runner.run_paprika_acoustic(cid, audio, sr=sr, force=True)

        print(f"[{cid}] Words ({len(acoustic_words)}): {' '.join(w.word for w in acoustic_words)}")

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

    for cfg in PROFANE_CLIPS:
        cid = cfg["id"]
        if cid not in existing_ids:
            manifest["clips"].append({
                "id": cid,
                "file": f"original_videos/{cid}.mp4",
                "category": "profanity",
                "notes": cfg["notes"]
            })
            print(f"Added {cid} to manifest.json")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Manifest updated successfully.")


if __name__ == "__main__":
    print("Step 1: Downloading 5-8s clip sections...")
    download_and_extract()
    print("\nStep 2: Transcribing and creating Praat TextGrids...")
    transcribe_and_export_praat()
    print("\nStep 3: Updating manifest...")
    update_manifest()
    print("\nALL 6 PROFANITY CLIPS READY FOR PRAAT!")
