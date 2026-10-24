import numpy as np
import pytest

from zalia_ruta.acoustic_refiner import compute_frame_rms, refine_word_boundaries
from zalia_ruta.asr import WordTimestamp


def test_compute_frame_rms_constant():
    sr = 16000
    win_samples = int(0.025 * sr)  # 25 ms = 400
    hop_samples = int(0.005 * sr)  # 5 ms = 80

    # Constant signal of 0.5
    audio = np.full(sr, 0.5, dtype=np.float32)
    rms = compute_frame_rms(audio, win_samples, hop_samples)

    assert len(rms) > 0
    np.testing.assert_allclose(rms, 0.5, atol=1e-5)


def test_compute_frame_rms_empty_or_short():
    rms_empty = compute_frame_rms(np.array([], dtype=np.float32), 400, 80)
    assert len(rms_empty) == 1
    assert rms_empty[0] == 0.0

    short_audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    rms_short = compute_frame_rms(short_audio, 400, 80)
    assert len(rms_short) == 1
    assert np.isclose(rms_short[0], 0.5)


def test_refine_empty_inputs():
    assert refine_word_boundaries(np.array([]), []) == []
    words = [WordTimestamp("test", 0.1, 0.5)]
    assert refine_word_boundaries(np.array([]), words) == words


def test_refine_forward_decay_extension():
    sr = 16000
    total_sec = 2.0
    audio = np.zeros(int(total_sec * sr), dtype=np.float32)

    # Word speech from 0.5s to 0.9s:
    # 0.5s - 0.7s: strong sine wave (amplitude 0.8)
    # 0.7s - 0.85s: decaying sine wave (amplitude 0.8 -> 0.1)
    # 0.85s - 2.0s: silence
    t_core = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    audio[int(0.5 * sr) : int(0.7 * sr)] = 0.8 * np.sin(2 * np.pi * 300 * t_core)

    decay_len = int(0.15 * sr)
    t_decay = np.linspace(0, 0.15, decay_len, endpoint=False)
    env = np.linspace(0.8, 0.15, decay_len)
    audio[int(0.7 * sr) : int(0.7 * sr) + decay_len] = env * np.sin(2 * np.pi * 300 * t_decay)

    # MMS alignment prematurely cut off at 0.70s (simulating typical negative end bias)
    mms_words = [WordTimestamp(word="žodis", start=0.50, end=0.70)]

    refined = refine_word_boundaries(audio, mms_words, sr=sr)
    assert len(refined) == 1
    w = refined[0]

    # Start should be near 0.50
    assert abs(w.start - 0.50) < 0.03
    # End should have extended beyond 0.70 into the decay (at least to 0.74 with 45ms cap)
    assert w.end >= 0.74, f"Expected end >= 0.74, got {w.end}"
    # But should NOT extend beyond max cap (0.70 + 0.045 = 0.745)
    assert w.end <= 0.75, f"Expected end <= 0.75, got {w.end}"


def test_refine_pause_protection_and_non_overlap():
    sr = 16000
    audio = np.zeros(3 * sr, dtype=np.float32)

    # Two bursts of speech:
    # Burst 1: 0.5s - 0.8s
    # Silence: 0.8s - 2.0s (1.2s pause)
    # Burst 2: 2.0s - 2.4s
    t1 = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    audio[int(0.5 * sr) : int(0.8 * sr)] = 0.6 * np.sin(2 * np.pi * 400 * t1)

    t2 = np.linspace(0, 0.4, int(0.4 * sr), endpoint=False)
    audio[int(2.0 * sr) : int(2.4 * sr)] = 0.6 * np.sin(2 * np.pi * 400 * t2)

    words = [
        WordTimestamp(word="pirmas", start=0.52, end=0.78),
        WordTimestamp(word="antras", start=2.02, end=2.38),
    ]

    refined = refine_word_boundaries(audio, words, sr=sr)
    assert len(refined) == 2

    # Words must be non-overlapping and preserve the pause
    assert refined[0].end < refined[1].start
    inter_word_gap = refined[1].start - refined[0].end
    # The pause is ~1.2s; refiner must preserve at least 800ms of silence
    assert inter_word_gap > 0.80, f"Expected large gap, got {inter_word_gap}s"


def test_refine_tight_consecutive_words():
    sr = 16000
    audio = np.zeros(2 * sr, dtype=np.float32)

    # Continuous audio from 0.5s to 1.1s
    t = np.linspace(0, 0.6, int(0.6 * sr), endpoint=False)
    audio[int(0.5 * sr) : int(1.1 * sr)] = 0.5 * np.sin(2 * np.pi * 300 * t)

    words = [
        WordTimestamp(word="vienas", start=0.50, end=0.75),
        WordTimestamp(word="du", start=0.77, end=1.05),
    ]

    refined = refine_word_boundaries(audio, words, sr=sr, min_pause_clearance_s=0.02)
    assert len(refined) == 2

    # Guaranteed non-overlap
    assert refined[0].end <= refined[1].start
