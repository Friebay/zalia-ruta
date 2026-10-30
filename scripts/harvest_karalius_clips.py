import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.srt_io import Cue, write_srt
from zalia_ruta.bench.systems import BenchmarkRunner, KNOWN_MODELS
from zalia_ruta.bench.validate import validate_cues
from zalia_ruta.bench.youtube import fetch_youtube_captions, parse_youtube_json3
from zalia_ruta.profanity import ProfanityDetector, clean_token

FALSE_POSITIVE_STEMS = {
    "naci", "nacist", "nacional", "epistem", "problem",
    "pistol", "tipisk", "prisikur", "sukel", "sukaust", "atsuk", "blacklist", "blackk", "sukyt"
}

ROOT_PATTERNS = {
    "ble": re.compile(r"^b[u]?l+[eiyoa]+($|[tdkch]|mb).*$"),
    "nachui": re.compile(r"^(na|po|za)[chx]+u?[ijy]?.*$"),
    "suka": re.compile(r"^suk(a|os?|u|om|omis|yte|ele)$"),
    "kurva": re.compile(r"^kur[wv](a|os?|u|om|omis|yn.*|yt.*|el.*)$"),
    "pis": re.compile(r"^.*(pis[tk]|pisa|piso|pys[tk]|pyzd|pizd|[chx]uj).*$"),
    "dalbajob": re.compile(r"^.*(dalbajob|dolbajob).*$"),
    "job": re.compile(r"^.*(job[atn]|jeb[atn]).*$"),
    "byb": re.compile(r"^b[iy]b([iy][rs]|u|a|k|el|o|ui).*$"),
    "sik": re.compile(r"^(sikn|sikt|prisik|apsisik).*$"),
    "debil": re.compile(r"^debil.*$"),
}

CHANNEL_VIDEOS = [
    # Videos with already downloaded subtitles containing high-density profanities
    "Q7EuWWdNeV0",
    "DAXhjychhoE",
    "grm1qgjoROg",
    "5TvrAjcZYNw",
    "dSObkwfw2MA",
    "P9XZBL2JMLw",
    "8wceu-0oJdE",
    "Oa3JQOnd38s",
    "nGMb_oEVGJk",
    "y3UteynmikA",
    "3NJRjPWKptA",
    "5N1XcEvUZoA",
    "4oajhqE66RQ",
    "B7Mygu8uiYM",
    "PetdMzrPg_M",
    "PZ2gyZC5Kuw",
    "3_wAkycMk0c",
    "XnkFuZ5cBB8",
    "6hIRVqm5kdo",
    "7og_55op5Kg",
    # Additional Karalius video IDs
    "wFNGN7H77KU",
    "tGw1JL9vQ0k",
    "Nkkw8iFU3Lk",
    "CfKsB4pVaW8",
    "84TrK5J_EN0",
    "bT-MuUiZ5Nc",
    "hrqrSXe8fU0",
    "MvBTuXEH65k",
    "8rHg4DAQYDA",
    "QNMnH5WOVIw",
    "Kt_GxRqxGrU",
    "x6L2WERdd9o",
    "m0mXKMR1rb8",
    "S1MvEiNz0Ec",
    "APg0TdAgn1Q",
    "uNOAz16O8jA",
    "Snzj5cexcB8",
    "J5L3R4u_lao",
    "uGKB2xHGKcY",
    "k-gwUarQ1x8",
    "UNdJzx9qHFw",
    "hMH_-4VStw8",
    "C472npN-uNI",
    "YgAM4-smaRo",
    "tZ8hXFvg7qQ",
    "PZNZQgrpq7M",
    "7LYgeXt7nCA",
    "9EOxTR-afZg",
    "ruF5d8WVMpQ",
    "GYAW6OnUGf0",
    "gA__JLz_nVs",
    "dj3AQt4idas",
    "lzJvHPrB4-I",
    "Sm4hI1n6PBI",
    "Q8BEue2pYiA",
    "Zdn8RO7SjEg",
    "Pw9_9Ys9WKc",
    "f17DMdkh0Cc",
    "mXQYkvL3Jq0",
    "fJYHkbwf2Yw",
    "e-vVQuZRxyU",
    "k51ZAhEAdTg",
    "_78AoUxFXtM",
    "uKxflgVXWWo",
    "RuQ9GXg6EMA",
    "fqDPt2MKpK0",
    "MDNKZbq66w0",
    "MUYLQlfSZ6g",
    "1U4N__2JEv0",
    "2lxnf5JQmHE",
    "DMAMYioI5PU",
    "EYtJqfyL738",
    "Kit7NpmiOZo",
    "VOl9WbjU-Sk",
    "ThBH61AtdWc",
    "UoYVLcS735Y",
    "R3WovGpBxFw",
    "PleBuSdKmNw",
    "oetLuchF3L8",
]


def get_profanity_root(token: str) -> Optional[str]:
    cl = clean_token(token)
    if not cl:
        return None
    for fp in FALSE_POSITIVE_STEMS:
        if fp in cl:
            return None
    for root, pat in ROOT_PATTERNS.items():
        if pat.match(cl):
            return root
    return None


def find_best_acoustic_cut(audio: np.ndarray, sr: int, search_start_sec: float, search_end_sec: float, window_ms: float = 60.0) -> float:
    start_sample = max(0, int(search_start_sec * sr))
    end_sample = min(len(audio), int(search_end_sec * sr))
    win_samples = int((window_ms / 1000.0) * sr)
    hop_samples = max(1, int(win_samples / 4))
    
    if start_sample >= end_sample - win_samples:
        return (search_start_sec + search_end_sec) / 2.0
        
    min_energy = float("inf")
    best_time = (search_start_sec + search_end_sec) / 2.0
    for i in range(start_sample, end_sample - win_samples, hop_samples):
        frame = audio[i : i + win_samples]
        energy = float(np.mean(frame ** 2))
        if energy < min_energy:
            min_energy = energy
            best_time = (i + win_samples / 2) / sr
    return best_time


def main():
    bench_dir = Path("benchmark")
    audio_dir = bench_dir / "audio"
    ref_dir = bench_dir / "references"
    yt_dir = bench_dir / "youtube"
    raw_dir = bench_dir / "audio" / "raw_sections"
    manifest_path = bench_dir / "manifest.json"

    audio_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    yt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_clips = manifest["clips"]
    
    highest_idx = 21
    for c in existing_clips:
        m = re.match(r"^karalius_prof_(\d+)$", c["id"])
        if m:
            highest_idx = max(highest_idx, int(m.group(1)))

    target_total_clips = 100
    needed_clips = target_total_clips - len(existing_clips)
    print(f"Current benchmark clips: {len(existing_clips)} | Target: {target_total_clips} (Need {needed_clips} more clips)", flush=True)
    if needed_clips <= 0:
        print("Target already reached!", flush=True)
        return

    runner = BenchmarkRunner(bench_dir)
    detector = ProfanityDetector()

    used_spans: Dict[str, List[Tuple[float, float]]] = {}
    for c in existing_clips:
        vid = c.get("youtube_id")
        if vid:
            if vid not in used_spans:
                used_spans[vid] = []

    current_idx = highest_idx + 1

    for vid in CHANNEL_VIDEOS:
        if len(manifest["clips"]) >= target_total_clips:
            break

        json3_path = yt_dir / f"{vid}.lt-orig.json3"
        if not json3_path.exists():
            print(f"\n--- Fetching subtitles for video {vid} ---", flush=True)
            fetch_youtube_captions(vid, json3_path)
            time.sleep(2.0)

        if not json3_path.exists():
            print(f"[{vid}] No subtitles available, skipping.", flush=True)
            continue

        try:
            words = parse_youtube_json3(json3_path)
        except Exception as e:
            print(f"[{vid}] Error parsing subtitles: {e}", flush=True)
            continue

        if not words:
            continue

        # Find profanity anchor words
        anchor_times = []
        for w in words:
            if detector.is_profanity(w.word):
                cl = clean_token(w.word)
                if not any(fp in cl for fp in FALSE_POSITIVE_STEMS):
                    anchor_times.append(w.start)

        if not anchor_times:
            continue

        candidate_starts = []
        for t in anchor_times:
            win_start = max(0.0, t - 10.0)
            if all(abs(win_start - cs) >= 25.0 for cs in candidate_starts):
                if vid in used_spans:
                    if any(abs(win_start - s[0]) < 25.0 for s in used_spans[vid]):
                        continue
                candidate_starts.append(win_start)

        print(f"\n[{vid}] Found {len(candidate_starts)} candidate windows to inspect", flush=True)

        for c_start in candidate_starts:
            if len(manifest["clips"]) >= target_total_clips:
                break

            c_end = c_start + 20.0
            clip_id = f"karalius_prof_{current_idx:02d}"

            dl_start = max(0.0, c_start - 2.5)
            dl_end = c_end + 2.5
            temp_m4a = raw_dir / f"temp_{clip_id}.m4a"
            temp_wav = raw_dir / f"temp_{clip_id}.wav"

            yt_url = f"https://www.youtube.com/watch?v={vid}"
            cmd_dl = [
                "yt-dlp",
                "--cookies-from-browser", "firefox",
                "--download-sections", f"*{dl_start:.1f}-{dl_end:.1f}",
                "-f", "ba[ext=m4a]/ba",
                yt_url,
                "-o", str(temp_m4a),
            ]
            print(f"  Downloading [{clip_id}] window {dl_start:.1f}s-{dl_end:.1f}s from {vid}...", end="", flush=True)
            t_dl0 = time.time()
            try:
                subprocess.run(cmd_dl, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception as e:
                print(f" FAILED download ({e})", flush=True)
                if temp_m4a.exists(): temp_m4a.unlink()
                continue
            t_dl = time.time() - t_dl0
            print(f" done ({t_dl:.1f}s)", flush=True)

            cmd_wav = [
                "ffmpeg", "-y", "-i", str(temp_m4a),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(temp_wav)
            ]
            try:
                subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception as e:
                print(f"    ffmpeg conversion failed ({e}), skipping", flush=True)
                continue
            finally:
                if temp_m4a.exists():
                    try:
                        temp_m4a.unlink()
                    except Exception:
                        pass

            if not temp_wav.exists():
                continue

            audio, sr = sf.read(str(temp_wav))
            raw_dur = len(audio) / sr
            if raw_dur < 16.0:
                temp_wav.unlink()
                print(f"    Skip: raw audio duration too short ({raw_dur:.1f}s)", flush=True)
                continue

            cut_start = find_best_acoustic_cut(audio, sr, 0.5, 3.5)
            cut_end = find_best_acoustic_cut(audio, sr, raw_dur - 3.5, raw_dur - 0.5)
            final_audio = audio[int(cut_start * sr) : int(cut_end * sr)]
            final_dur = len(final_audio) / sr

            if final_dur < 14.0 or final_dur > 26.0:
                temp_wav.unlink()
                print(f"    Skip: cut duration out of bounds ({final_dur:.1f}s)", flush=True)
                continue

            words_hyp = runner.run_model_acoustic(
                KNOWN_MODELS["paprika_ct2"], "paprika_ct2", clip_id, final_audio, sr=sr, force=True
            )

            detected_profs = []
            for w in words_hyp:
                r = get_profanity_root(w.word)
                if r:
                    detected_profs.append((w.word, r))

            unique_roots = set(p[1] for p in detected_profs)
            words_summary = ", ".join(f"'{p[0]}' [{p[1]}]" for p in detected_profs)

            if len(unique_roots) >= 2:
                final_wav_path = audio_dir / f"{clip_id}.wav"
                sf.write(str(final_wav_path), final_audio, sr)
                temp_wav.unlink()

                cues = []
                prev_end = 0.0
                for idx, w in enumerate(words_hyp, start=1):
                    s = round(float(w.start), 3)
                    e = round(float(w.end), 3)
                    if s < prev_end: s = prev_end
                    if e <= s: e = round(s + 0.05, 3)
                    if e > final_dur:
                        e = round(final_dur, 3)
                        if s >= e: s = max(0.0, round(e - 0.05, 3))
                    cues.append(Cue(index=idx, start=s, end=e, text=w.word))
                    prev_end = e

                ref_path = ref_dir / f"{clip_id}.srt"
                write_srt(cues, ref_path)

                notes_text = f"Karalius commentary with {len(unique_roots)} roots: {words_summary} ({final_dur:.1f}s)"

                manifest["clips"].append({
                    "id": clip_id,
                    "file": f"original_videos/{clip_id}.mp4",
                    "youtube_id": vid,
                    "category": "profanity",
                    "timing_audited": False,
                    "notes": notes_text
                })
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

                if vid not in used_spans:
                    used_spans[vid] = []
                used_spans[vid].append((c_start, c_end))

                print(f"    >>> ACCEPTED [{clip_id}] ({final_dur:.1f}s): {words_summary}", flush=True)
                print(f"        Benchmark clips: {len(manifest['clips'])} / {target_total_clips}", flush=True)
                current_idx += 1
            else:
                temp_wav.unlink()
                print(f"    --- Discarded: {len(unique_roots)} root(s) found: {words_summary}", flush=True)

    print(f"\nHarvesting complete! Total benchmark clips: {len(manifest['clips'])}", flush=True)


if __name__ == "__main__":
    main()
