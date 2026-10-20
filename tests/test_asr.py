from zalia_ruta.asr import WordTimestamp


def test_word_timestamp_creation():
    wt = WordTimestamp(word="lietuva", start=1.23, end=1.85)
    assert wt.word == "lietuva"
    assert wt.start == 1.23
    assert wt.end == 1.85
    assert "lietuva" in repr(wt)


def test_acoustic_refine_words():
    import numpy as np
    from zalia_ruta.asr import acoustic_refine_words

    sr = 16000
    # Create 3 seconds of audio:
    # 0.0s - 0.4s: silence
    # 0.4s - 0.9s: speech burst (sine wave)
    # 0.9s - 1.8s: breath / silence (900 ms gap!)
    # 1.8s - 2.4s: speech burst (sine wave)
    # 2.4s - 3.0s: silence
    audio = np.zeros(sr * 3, dtype=np.float32)
    t1 = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    audio[int(0.4 * sr) : int(0.9 * sr)] = 0.5 * np.sin(2 * np.pi * 400 * t1)

    t2 = np.linspace(0, 0.6, int(0.6 * sr), endpoint=False)
    audio[int(1.8 * sr) : int(2.4 * sr)] = 0.5 * np.sin(2 * np.pi * 400 * t2)

    # Whisper naively gives contiguous tiling without gaps:
    raw_words = [
        WordTimestamp(word="ble", start=0.0, end=1.0),
        WordTimestamp(word="šiandien", start=1.0, end=2.8),
    ]

    refined = acoustic_refine_words(raw_words, audio, sr=sr, time_offset=0.0)
    assert len(refined) == 2

    # First word onset was at 0.4s, so start should NOT be 0.0s
    assert refined[0].start >= 0.35
    # First word ends near 0.9s
    assert refined[0].end <= 0.95

    # Second word begins near 1.8s, so start should NOT be 1.0s
    assert refined[1].start >= 1.75
    # Second word ends near 2.4s, so end should NOT be 2.8s
    assert refined[1].end <= 2.45

    # There MUST be a significant gap between word 0 and word 1
    gap = refined[1].start - refined[0].end
    assert gap > 0.60, f"Expected gap > 600ms, got {gap*1000:.0f}ms"

