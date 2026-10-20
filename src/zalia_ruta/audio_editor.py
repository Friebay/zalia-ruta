from pathlib import Path
from typing import List
import numpy as np
import soundfile as sf

from zalia_ruta.profanity import BleepInterval


def generate_sine_wave(
    duration_samples: int,
    sr: int,
    freq: float = 1000.0,
    volume: float = 0.35,
    fade_samples: int = 0,
) -> np.ndarray:
    """
    Generates a pure sine wave tone of specified frequency, volume, and sample duration,
    with optional linear fade-in and fade-out to prevent clicking.
    """
    t = np.arange(duration_samples, dtype=np.float32) / sr
    tone = (volume * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    if fade_samples > 0 and 2 * fade_samples <= duration_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

    return tone


def apply_bleep_intervals(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    intervals: List[BleepInterval],
    tone_freq: float = 1000.0,
    tone_volume: float = 0.35,
    fade_ms: float = 5.0,
) -> Path:
    """
    Replaces audio segments in `intervals` with a 1000 Hz sine wave tone,
    applying smooth 5 ms crossfading to eliminate audio click/pop artifacts.
    Preserves original sample rate, bit-depth, and channel count.
    """
    input_wav_path = Path(input_wav_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(input_wav_path), dtype="float32")
    is_multichannel = audio.ndim > 1
    num_channels = audio.shape[1] if is_multichannel else 1
    total_samples = len(audio)

    fade_samples = max(1, int((fade_ms / 1000.0) * sr))

    for iv in intervals:
        start_idx = max(0, int(iv.start_sec * sr))
        end_idx = min(total_samples, int(iv.end_sec * sr))
        n_samples = end_idx - start_idx

        if n_samples <= 0:
            continue

        curr_fade = min(fade_samples, n_samples // 2)
        tone = generate_sine_wave(
            duration_samples=n_samples,
            sr=sr,
            freq=tone_freq,
            volume=tone_volume,
            fade_samples=curr_fade,
        )

        # Crossfade background audio transition at edges
        if curr_fade > 0:
            audio_fade_out = np.linspace(1.0, 0.0, curr_fade, dtype=np.float32)
            audio_fade_in = np.linspace(0.0, 1.0, curr_fade, dtype=np.float32)

            if is_multichannel:
                audio[start_idx : start_idx + curr_fade] *= audio_fade_out[:, None]
                audio[end_idx - curr_fade : end_idx] *= audio_fade_in[:, None]
                # Insert tone across all channels
                audio[start_idx:end_idx] = tone[:, None]
            else:
                audio[start_idx : start_idx + curr_fade] *= audio_fade_out
                audio[end_idx - curr_fade : end_idx] *= audio_fade_in
                audio[start_idx:end_idx] = tone
        else:
            if is_multichannel:
                audio[start_idx:end_idx] = tone[:, None]
            else:
                audio[start_idx:end_idx] = tone

    # Save output master WAV
    sf.write(str(output_wav_path), audio, sr, subtype="PCM_16")
    return output_wav_path
