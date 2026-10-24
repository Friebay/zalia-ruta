import pytest
from zalia_ruta.asr import WordTimestamp
from zalia_ruta.profanity import ProfanityDetector
from zalia_ruta.highlights import find_pause_cut_points, find_profanity_highlight_windows, HighlightWindow


def test_find_pause_cut_points():
    words = [
        WordTimestamp("vienas", 1.0, 1.5),
        WordTimestamp("du", 2.0, 2.5),
        WordTimestamp("trys", 3.0, 3.5),
    ]
    cuts = find_pause_cut_points(words, total_duration_sec=5.0)

    # First cut at 0.0
    assert cuts[0][0] == 0.0
    assert cuts[0][1] == 1.0

    # Gap between 1.5 and 2.0 is 0.5, midpoint is 1.75
    assert cuts[1][0] == 1.75
    assert cuts[1][1] == 0.5

    # Gap between 2.5 and 3.0 is 0.5, midpoint is 2.75
    assert cuts[2][0] == 2.75
    assert cuts[2][1] == 0.5

    # Final cut at 5.0, gap from 3.5 to 5.0 is 1.5
    assert cuts[3][0] == 5.0
    assert cuts[3][1] == 1.5


def test_find_profanity_highlight_windows():
    detector = ProfanityDetector()

    # Generate synthetic sequence of 100 words over 120 seconds
    words = []
    # Clean section 0-40s
    t = 1.0
    for i in range(20):
        words.append(WordTimestamp(f"zodis{i}", round(t, 2), round(t + 0.8, 2)))
        t += 1.5

    # Profanity storm 40s - 70s
    t = 40.0
    for w in ["blet", "suka", "nachui", "kurva", "pyzda", "debilas", "blet"]:
        words.append(WordTimestamp(w, round(t, 2), round(t + 0.5, 2)))
        t += 1.2

    # More normal words 70s - 120s
    t = 70.0
    for i in range(20):
        words.append(WordTimestamp(f"ramus{i}", round(t, 2), round(t + 0.8, 2)))
        t += 1.5

    windows = find_profanity_highlight_windows(
        words=words,
        detector=detector,
        top_k=1,
        target_duration_sec=60.0,
        min_duration_sec=30.0,
        max_duration_sec=70.0,
        total_duration_sec=120.0,
    )

    assert len(windows) == 1
    top = windows[0]
    assert top.rank == 1
    assert top.profanity_count >= 7
    # Ensure window encompasses the profanity cluster around 40-50s
    assert top.start_sec <= 40.0
    assert top.end_sec >= 50.0
