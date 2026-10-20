import numpy as np
from zalia_ruta.chunker import find_pause_cuts


def test_find_pause_cuts_short():
    sr = 16000
    # 15 seconds of audio (shorter than min_chunk_s)
    audio = np.random.randn(15 * sr).astype(np.float32)
    chunks = find_pause_cuts(audio, sr=sr, min_chunk_s=20.0, max_chunk_s=29.0)
    assert len(chunks) == 1
    assert chunks[0].start_sec == 0.0
    assert chunks[0].duration_sec == 15.0


def test_find_pause_cuts_splits_at_silence():
    sr = 16000
    # 50 seconds total:
    # 0..24s: noise
    # 24..25s: near absolute silence (zeros)
    # 25..50s: noise
    audio = np.random.randn(50 * sr).astype(np.float32)
    silence_start = 24 * sr
    silence_end = 25 * sr
    audio[silence_start:silence_end] = 0.0

    chunks = find_pause_cuts(audio, sr=sr, min_chunk_s=20.0, max_chunk_s=29.0)
    assert len(chunks) == 2
    # The cut must happen inside the silence interval [24.0, 25.0]
    cut_sec = chunks[0].end_sec
    assert 24.0 <= cut_sec <= 25.0
    assert chunks[1].start_sec == cut_sec
    assert np.isclose(chunks[1].end_sec, 50.0)
