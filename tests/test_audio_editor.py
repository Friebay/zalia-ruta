from pathlib import Path
import numpy as np
import soundfile as sf

from zalia_ruta.audio_editor import generate_sine_wave, apply_bleep_intervals
from zalia_ruta.profanity import BleepInterval


def test_generate_sine_wave():
    sr = 16000
    duration_samples = 16000  # 1 second
    tone = generate_sine_wave(duration_samples, sr, freq=1000.0, volume=0.5, fade_samples=160)
    assert len(tone) == duration_samples
    # Check max volume limit
    assert np.max(np.abs(tone)) <= 0.5001
    # Check that fade-in starts near 0.0
    assert np.isclose(tone[0], 0.0, atol=1e-3)


def test_apply_bleep_intervals(tmp_path: Path):
    sr = 44100
    # Create 3 seconds of stereo white noise
    duration_s = 3.0
    total_samples = int(duration_s * sr)
    noise = np.random.uniform(-0.5, 0.5, size=(total_samples, 2)).astype(np.float32)

    in_wav = tmp_path / "test_input.wav"
    out_wav = tmp_path / "test_output.wav"
    sf.write(str(in_wav), noise, sr, subtype="PCM_16")

    # Bleep 1.0s to 2.0s
    intervals = [BleepInterval(start_sec=1.0, end_sec=2.0, matched_word="test", original_word="test")]
    apply_bleep_intervals(in_wav, out_wav, intervals, tone_freq=1000.0, fade_ms=5.0)

    # Read output and verify
    out_audio, out_sr = sf.read(str(out_wav), dtype="float32")
    assert out_sr == sr
    assert len(out_audio) == total_samples
    assert out_audio.shape == (total_samples, 2)

    # In [1.1s, 1.9s], audio should be pure 1000 Hz sine wave on both channels
    sub_chunk = out_audio[int(1.1 * sr) : int(1.9 * sr)]
    # Left and right channels should be identical sine wave
    diff = np.abs(sub_chunk[:, 0] - sub_chunk[:, 1])
    assert np.max(diff) < 1e-4

    # Before 1.0s and after 2.0s, audio should remain original noise (not bleep tone)
    before_chunk = out_audio[: int(0.9 * sr)]
    assert np.std(before_chunk) > 0.1
