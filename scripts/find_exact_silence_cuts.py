import sys
from pathlib import Path
import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def analyze_rms(wav_path: Path):
    audio, sr = sf.read(str(wav_path))
    # Frame size 20ms, hop 10ms
    frame_len = int(sr * 0.020)
    hop_len = int(sr * 0.010)
    
    rms = []
    times = []
    for i in range(0, len(audio) - frame_len, hop_len):
        chunk = audio[i : i + frame_len]
        r = np.sqrt(np.mean(chunk**2))
        rms.append(r)
        times.append(i / sr)
        
    rms = np.array(rms)
    times = np.array(times)
    # Threshold for silence: 5% of max RMS or fixed 0.01
    silence_thresh = max(0.005, np.percentile(rms, 15) * 1.5)
    return audio, sr, times, rms, silence_thresh

print("=== Analyzing karalius_prof_07_raw.wav ===")
wav7 = Path("benchmark/audio/raw_sections/karalius_prof_07_raw.wav")
audio7, sr7, times7, rms7, th7 = analyze_rms(wav7)

# Let's inspect intervals around 0.8s - 1.2s, 2.0s - 2.4s, and 7.4s - 8.5s
def print_energy_profile(times, rms, th, start_t, end_t):
    mask = (times >= start_t) & (times <= end_t)
    for t, r in zip(times[mask], rms[mask]):
        is_silence = "SILENCE" if r <= th else "SPEECH "
        bar = "#" * int(min(50, r * 200))
        print(f"  {t:5.3f}s: RMS={r:.4f} [{is_silence}] {bar}")

print("\n--- 07: Around 2.0s - 2.5s (before 'What the fuck') ---")
print_energy_profile(times7, rms7, th7, 2.0, 2.5)

print("\n--- 07: Around 7.5s - 8.5s (after 'ar aš tikrai...') ---")
print_energy_profile(times7, rms7, th7, 7.5, 8.5)

print("\n=== Analyzing karalius_prof_08_raw.wav ===")
wav8 = Path("benchmark/audio/raw_sections/karalius_prof_08_raw.wav")
audio8, sr8, times8, rms8, th8 = analyze_rms(wav8)

print("\n--- 08: Around 0.5s - 1.5s (before 'blet blet') ---")
print_energy_profile(times8, rms8, th8, 0.5, 1.5)

print("\n--- 08: Around 8.0s - 9.2s (after 'nereikia ir jaučiasi') ---")
print_energy_profile(times8, rms8, th8, 8.0, 9.2)
