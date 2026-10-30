import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLIPS_CONFIG = [
    {
        "id": "karalius_prof_12",
        "video_id": "6hIRVqm5kdo",
        "target_start": 2060.0,
        "target_end": 2082.0,
        "search_start_range": (2058.8, 2060.8),  # pause before "80% Lietuvos yra debilai"
        "search_end_range": (2081.3, 2082.8),    # pause after "žmonių yra debilai"
        "swear_count": 4,
        "profs": ["debilai", "debilams", "debilam", "debilai"],
        "notes": "Karalius rant on audience, '80% Lietuvos yra debilai... visą kontą debilams... žmonių yra debilai' (4 profanities)",
    },
    {
        "id": "karalius_prof_13",
        "video_id": "6hIRVqm5kdo",
        "target_start": 7047.0,
        "target_end": 7070.5,
        "search_start_range": (7046.2, 7048.2),  # pause before "ar nėra dalbajobų"
        "search_end_range": (7069.8, 7071.5),    # pause after "dalbajobų nėra valdžioj"
        "swear_count": 4,
        "profs": ["dalbajobų", "dalbajobų", "dalbajobų", "dalbajobų"],
        "notes": "Karalius politics discussion, 'nėra dalbajobų valdžioj... dalbajobų valdžioj... dalbajobų nėra valdžioj' (4 profanities)",
    },
    {
        "id": "karalius_prof_14",
        "video_id": "6hIRVqm5kdo",
        "target_start": 7131.0,
        "target_end": 7150.5,
        "search_start_range": (7129.8, 7132.0),  # pause before "Švytuoklės principas"
        "search_end_range": (7149.0, 7151.0),    # pause after "kur nebus dalbajobų"
        "swear_count": 2,
        "profs": ["užpiso", "dalbajobų"],
        "notes": "Karalius commentary, 'kas juos labiau užpiso... idealaus pasaulio, kur nebus dalbajobų' (2 profanities)",
    },
    {
        "id": "karalius_prof_15",
        "video_id": "6hIRVqm5kdo",
        "target_start": 4950.0,
        "target_end": 4969.5,
        "search_start_range": (4948.0, 4950.5),  # pause before "Aš žinai, va tai yra lengva"
        "search_end_range": (4968.0, 4970.5),    # pause after "ir ką? Čia yra problema."
        "swear_count": 2,
        "profs": ["dalbajobai", "dalbajobai"],
        "notes": "Karalius content rant, 'žiūri žmonės dalbajobai ir jie liks dalbajobai' (2 profanities)",
    },
    {
        "id": "karalius_prof_16",
        "video_id": "6hIRVqm5kdo",
        "target_start": 3328.0,
        "target_end": 3348.0,
        "search_start_range": (3327.2, 3329.0),  # pause before "Nu, toks skirtumas. Aš noriu"
        "search_end_range": (3346.8, 3348.5),    # pause after "būtumėm vertesni geresnio gyvenimo"
        "swear_count": 2,
        "profs": ["debilus", "debilus"],
        "notes": "Karalius democracy commentary, 'išrenka debilus... tupi žmonės išrinktų debilus' (2 profanities)",
    },
    {
        "id": "karalius_prof_17",
        "video_id": "6hIRVqm5kdo",
        "target_start": 4760.0,
        "target_end": 4780.0,
        "search_start_range": (4759.0, 4761.5),  # pause before "Ir kaip sakei, man taip atrodo"
        "search_end_range": (4778.8, 4781.0),    # pause after "edukuoti bent vieną du debilus"
        "swear_count": 2,
        "profs": ["debilų", "debilus"],
        "notes": "Karalius audience rant, 'kuriame yra 80% debilų... edukuoti bent vieną du debilus' (2 profanities)",
    },
    {
        "id": "karalius_prof_18",
        "video_id": "3_wAkycMk0c",
        "target_start": 4224.0,
        "target_end": 4244.5,
        "search_start_range": (4223.8, 4225.5),  # pause before "velnias koksai jis"
        "search_end_range": (4243.5, 4245.5),    # pause after "dėl tų migrantų visos tos situacijos"
        "swear_count": 2,
        "profs": ["išpiso", "išpiso"],
        "notes": "Karalius politics critique, 'Visiem išpiso protus. Visiem išpiso protus dėl migrantų' (2 profanities)",
    },
    {
        "id": "karalius_prof_19",
        "video_id": "3_wAkycMk0c",
        "target_start": 1736.0,
        "target_end": 1757.0,
        "search_start_range": (1734.5, 1736.8),  # pause before "Tai planas yra toks: prapisam"
        "search_end_range": (1755.8, 1758.0),    # pause after "jie prapisa. Ateina konservatoriai"
        "swear_count": 2,
        "profs": ["prapisam", "prapisa"],
        "notes": "Karalius election commentary, 'planas yra toks: prapisam rinkimus... nes jie prapisa' (2 profanities)",
    },
    {
        "id": "karalius_prof_20",
        "video_id": "XnkFuZ5cBB8",
        "target_start": 2173.5,
        "target_end": 2196.5,
        "search_start_range": (2172.5, 2174.2),  # pause before "Gerai, debilu pavadinu žmogų"
        "search_end_range": (2195.5, 2197.5),    # pause after "tai nėra nusikaltimas. Gerai."
        "swear_count": 3,
        "profs": ["debilu", "debilą", "debilu"],
        "notes": "Karalius legal commentary, 'Gerai, debilu pavadinu žmogų... kažkokį debilą debilu pavadinau' (3 profanities)",
    },
    {
        "id": "karalius_prof_21",
        "video_id": "XnkFuZ5cBB8",
        "target_start": 3110.0,
        "target_end": 3131.0,
        "search_start_range": (3109.5, 3111.5),  # pause before "Ne, negaliu sakyt. Tai yra riba peržengta"
        "search_end_range": (3130.0, 3132.2),    # pause after "tai nėra nusikaltimas"
        "swear_count": 2,
        "profs": ["debilų", "debilą"],
        "notes": "Karalius legal commentary, 'Tai yra riba peržengta... debilų žmogų debilą pavadini' (2 profanities)",
    },
]

def find_best_acoustic_cut(audio: np.ndarray, sr: int, search_start_sec: float, search_end_sec: float, base_sec: float, window_ms: float = 60.0) -> float:
    """Finds lowest energy point in search range relative to base_sec."""
    rel_start = max(0.0, search_start_sec - base_sec)
    rel_end = min(len(audio) / sr, search_end_sec - base_sec)
    
    start_sample = int(rel_start * sr)
    end_sample = int(rel_end * sr)
    win_samples = int((window_ms / 1000.0) * sr)
    hop_samples = int((win_samples / 4))
    
    if start_sample >= end_sample - win_samples:
        return (search_start_sec + search_end_sec) / 2.0
        
    min_energy = float("inf")
    best_sample = (start_sample + end_sample) // 2
    
    for i in range(start_sample, end_sample - win_samples, hop_samples):
        frame = audio[i : i + win_samples]
        energy = np.mean(frame**2)
        if energy < min_energy:
            min_energy = energy
            best_sample = i + win_samples // 2
            
    return base_sec + (best_sample / sr)

def process_all_clips():
    praat_dir = Path("praat_samples")
    bench_audio_dir = Path("benchmark/audio")
    praat_dir.mkdir(parents=True, exist_ok=True)
    bench_audio_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for cfg in CLIPS_CONFIG:
        cid = cfg["id"]
        vid = cfg["video_id"]
        yt_url = f"https://www.youtube.com/watch?v={vid}"
        
        # Download wide buffer: target_start - 3.0s to target_end + 3.0s
        raw_start = max(0.0, cfg["search_start_range"][0] - 2.0)
        raw_end = cfg["search_end_range"][1] + 2.0
        
        tmp_m4a = praat_dir / f"{cid}_raw.m4a"
        tmp_raw_wav = praat_dir / f"{cid}_raw.wav"
        final_wav = bench_audio_dir / f"{cid}.wav"
        praat_wav = praat_dir / f"{cid}.wav"
        
        print(f"\n=======================================================")
        print(f"[{cid}] Downloading raw audio buffer ({raw_start:.2f}s - {raw_end:.2f}s) from {vid}...")
        cmd_dl = [
            "yt-dlp",
            "--download-sections", f"*{raw_start:.2f}-{raw_end:.2f}",
            "--force-keyframes-at-cuts",
            "-f", "ba[ext=m4a]/bestaudio/best",
            yt_url,
            "-o", str(tmp_m4a),
        ]
        subprocess.run(cmd_dl, check=True)
        
        # Convert raw to 16kHz mono WAV
        cmd_wav = [
            "ffmpeg", "-y", "-i", str(tmp_m4a),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(tmp_raw_wav),
        ]
        subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if tmp_m4a.exists():
            tmp_m4a.unlink()
            
        # Read raw wav and find exact acoustic cut points
        raw_audio, sr = sf.read(str(tmp_raw_wav), dtype="float32")
        
        cut_start = find_best_acoustic_cut(
            raw_audio, sr,
            cfg["search_start_range"][0],
            cfg["search_start_range"][1],
            base_sec=raw_start
        )
        cut_end = find_best_acoustic_cut(
            raw_audio, sr,
            cfg["search_end_range"][0],
            cfg["search_end_range"][1],
            base_sec=raw_start
        )
        
        dur = cut_end - cut_start
        print(f"[{cid}] Refined acoustic boundaries: {cut_start:.3f}s -> {cut_end:.3f}s (Duration: {dur:.2f}s)")
        
        # Slice exact audio
        rel_start_sample = int((cut_start - raw_start) * sr)
        rel_end_sample = int((cut_end - raw_start) * sr)
        sliced_audio = raw_audio[rel_start_sample:rel_end_sample]
        
        # Write final 16kHz mono WAV
        sf.write(str(final_wav), sliced_audio, sr, subtype="PCM_16")
        sf.write(str(praat_wav), sliced_audio, sr, subtype="PCM_16")
        
        if tmp_raw_wav.exists():
            tmp_raw_wav.unlink()
            
        print(f"[{cid}] Saved cleanly to {final_wav} ({len(sliced_audio)/sr:.2f}s)")
        
        results.append({
            "id": cid,
            "video_id": vid,
            "start": round(cut_start, 2),
            "end": round(cut_end, 2),
            "duration": round(dur, 2),
            "swear_count": cfg["swear_count"],
            "profs": cfg["profs"],
            "notes": cfg["notes"]
        })
        
    print("\n================== ALL 10 CLIPS SLICED SUCCESSFULLY ==================")
    for r in results:
        print(f"{r['id']} | {r['video_id']} | {r['start']}s - {r['end']}s ({r['duration']}s) | Count: {r['swear_count']} | {r['profs']}")

if __name__ == "__main__":
    process_all_clips()
