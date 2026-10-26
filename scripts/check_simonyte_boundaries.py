import json
import sys
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.systems import deserialize_words

CANDIDATE_IDS = [
    "karalius_prof_02",
    "karalius_prof_03",
    "karalius_prof_04",
    "karalius_prof_05",
    "karalius_prof_06",
    "karalius_silence_01",
    "karalius_silence_02",
    "karalius_silence_03",
    "karalius_silence_04",
    "karalius_silence_05",
]

def check_all():
    audio_dir = Path("benchmark/audio")
    hyp_dir = Path("benchmark/hyp/paprika_acoustic")
    
    print("\n================== BOUNDARY & CUT-OFF CHECK ==================")
    for cid in CANDIDATE_IDS:
        wav_path = audio_dir / f"{cid}.wav"
        hyp_path = hyp_dir / f"{cid}.json"
        
        if not wav_path.exists() or not hyp_path.exists():
            print(f"[{cid}] Waiting for generation...")
            continue
            
        data, sr = sf.read(str(wav_path))
        dur = len(data) / sr
        
        raw = json.loads(hyp_path.read_text(encoding="utf-8"))
        words = deserialize_words(raw)
        
        print(f"\n--- Clip: {cid} (duration: {dur:.3f}s) ---")
        if not words:
            print("  [SILENCE] 0 words transcribed.")
            continue
            
        first_w = words[0]
        last_w = words[-1]
        
        first_margin = first_w.start
        last_margin = dur - last_w.end
        
        full_text = " ".join(w.word for w in words)
        print(f"  Words ({len(words)}): \"{full_text}\"")
        print(f"  First word: '{first_w.word}' starts at {first_w.start:.3f}s (margin: {first_margin:.3f}s)")
        print(f"  Last word:  '{last_w.word}' ends at {last_w.end:.3f}s (margin: {last_margin:.3f}s)")
        
        # Diagnostics
        issues = []
        if first_margin < 0.05:
            issues.append(f"LEADING CUT RISK: '{first_w.word}' starts at {first_w.start:.3f}s (<50ms margin)")
        if last_margin < 0.05:
            issues.append(f"TRAILING CUT RISK: '{last_w.word}' ends at {last_w.end:.3f}s (<50ms margin)")
            
        if issues:
            for iss in issues:
                print(f"  [!] {iss}")
        else:
            print(f"  [OK] Clean acoustic margins on both boundaries.")

if __name__ == "__main__":
    check_all()
