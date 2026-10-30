import numpy as np
import pytest
import torch

from zalia_ruta.asr import MMSWordAligner, WordTimestamp, acoustic_refine_words, clean_for_mms


def test_word_timestamp_creation():
    wt = WordTimestamp(word="lietuva", start=1.23, end=1.85)
    assert wt.word == "lietuva"
    assert wt.start == 1.23
    assert wt.end == 1.85
    assert "lietuva" in repr(wt)


def test_clean_for_mms():
    # Lithuanian diacritics mapping
    assert clean_for_mms("ąčęėįšųūž") == "aceeisuuz"
    assert clean_for_mms("ĄČĘĖĮŠŲŪŽ") == "aceeisuuz"
    assert clean_for_mms("Lietuvos") == "lietuvos"

    # Punctuation stripping
    assert clean_for_mms("labas,") == "labas"
    assert clean_for_mms("„žodis“!") == "zodis"
    assert clean_for_mms("-kaip-") == "kaip"
    assert clean_for_mms("gal-but") == "gal-but"

    # Non-letter / digits handling
    assert clean_for_mms("123") == ""
    assert clean_for_mms("---") == ""
    assert clean_for_mms("???") == ""


def test_mms_aligner_fallback_guarantee():
    aligner = MMSWordAligner(device="cpu")

    audio = np.zeros(16000 * 2, dtype=np.float32)

    # Word list containing a number that produces no CTC tokens
    words = [
        WordTimestamp(word="pirmas", start=0.1, end=0.6),
        WordTimestamp(word="12345", start=0.6, end=1.1),
        WordTimestamp(word="antras", start=1.1, end=1.7),
    ]

    aligned = aligner.align(audio, words, sr=16000, time_offset=0.0)

    # All 3 words must be preserved in exact order
    assert len(aligned) == 3
    assert [w.word for w in aligned] == ["pirmas", "12345", "antras"]

    # "12345" had no clean tokens, so its Whisper fallback timestamps MUST be preserved
    assert aligned[1].start == 0.6
    assert aligned[1].end == 1.1

    # Empty inputs guarantee
    assert aligner.align(np.array([], dtype=np.float32), words) == words
    assert aligner.align(audio, []) == []


def test_mms_aligner_synthetic_audio():
    aligner = MMSWordAligner(device="cpu")

    # Generate 1.5 seconds of synthetic audio
    sr = 16000
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    words = [
        WordTimestamp(word="labas", start=0.0, end=0.5),
        WordTimestamp(word="vakaras", start=0.5, end=1.2),
    ]

    offset = 5.0
    aligned = aligner.align(audio, words, sr=sr, time_offset=offset)

    assert len(aligned) == 2
    for w in aligned:
        assert w.start >= offset
        assert w.end > w.start
        assert w.end <= offset + 2.0


def test_acoustic_refine_words():
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


def test_lithuanian_asr_backend_resolution():
    from pathlib import Path
    from zalia_ruta.asr import LithuanianASR, DEFAULT_INITIAL_PROMPT

    ct2_path = Path("models/paprika-whisper-lt-v3-ct2-int8_float16")
    if ct2_path.exists():
        asr = LithuanianASR(model_id=str(ct2_path), device="cpu", backend="auto", use_mms_align=False)
        assert asr.effective_backend == "faster-whisper"
        assert asr.compute_type == "int8"
        assert asr.initial_prompt == DEFAULT_INITIAL_PROMPT

        # Custom prompt
        asr_custom = LithuanianASR(model_id=str(ct2_path), device="cpu", backend="auto", use_mms_align=False, initial_prompt="blet nachui")
        assert asr_custom.initial_prompt == "blet nachui"

        # Disabled prompt
        asr_disabled = LithuanianASR(model_id=str(ct2_path), device="cpu", backend="auto", use_mms_align=False, initial_prompt=None)
        assert asr_disabled.initial_prompt is None

