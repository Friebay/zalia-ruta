import pytest
from zalia_ruta.asr import WordTimestamp
from zalia_ruta.bench.score import calculate_pause_bleed, compute_alignment_matches, score_system
from zalia_ruta.bench.srt_io import Cue


def test_score_system_text_and_timing():
    ref_cues = [
        Cue(index=1, start=1.0, end=1.5, text="labas"),
        Cue(index=2, start=2.0, end=2.5, text="rytas"),
        Cue(index=3, start=3.0, end=3.5, text="Lietuva"),
    ]

    # Hypothesis with 100% text match and exact timing shifted +40ms
    hyp_words = [
        WordTimestamp(word="labas", start=1.04, end=1.54),
        WordTimestamp(word="rytas", start=2.04, end=2.54),
        WordTimestamp(word="Lietuva", start=3.04, end=3.54),
    ]

    res = score_system("test_sys", "clip1", ref_cues, hyp_words)
    assert res.wer == 0.0
    assert res.cer == 0.0
    assert res.timing is not None
    assert res.timing.matched_words == 3
    assert abs(res.timing.start_med_ms - 40.0) < 1.0
    assert abs(res.timing.end_med_ms - 40.0) < 1.0
    # Within 50ms should be 100%
    assert res.timing.within_50ms_pct == 100.0


def test_pause_bleed_calculation():
    # Gap between word 1 and 2 is 1.5s to 2.5s (1.0s total silence)
    ref_cues = [
        Cue(index=1, start=1.0, end=1.5, text="pirmas"),
        Cue(index=2, start=2.5, end=3.0, text="antras"),
    ]

    # Hypothesis word bleeds into gap from 1.5s to 1.8s (0.3s bleed into 1.0s gap = 30%)
    hyp_words = [
        WordTimestamp(word="pirmas", start=1.0, end=1.8),
        WordTimestamp(word="antras", start=2.5, end=3.0),
    ]

    bleed = calculate_pause_bleed(ref_cues, hyp_words, min_gap_s=0.15)
    assert bleed == 30.0


def test_special_tokens_excluded():
    ref_cues = [
        Cue(index=1, start=1.0, end=1.5, text="~ee"),
        Cue(index=2, start=2.0, end=2.5, text="tikras"),
        Cue(index=3, start=3.0, end=3.5, text="#SKIP"),
    ]
    hyp_words = [
        WordTimestamp(word="tikras", start=2.0, end=2.5),
    ]

    res = score_system("test_sys", "clip1", ref_cues, hyp_words)
    assert res.ref_words_count == 1
    assert res.wer == 0.0


def test_brand_name_normalization():
    from zalia_ruta.bench.score import normalize_word

    assert normalize_word("Facebook'o") == normalize_word("feisbuko")
    assert normalize_word("Facebook'e") == normalize_word("feisbuke")
    assert normalize_word("YouTube's") == normalize_word("jutubės")
    assert normalize_word("discord'ą") == normalize_word("diskordą")
    assert normalize_word("dicord'as") == normalize_word("diskordas")
    assert normalize_word("Edge'ą") == normalize_word("edžą")

