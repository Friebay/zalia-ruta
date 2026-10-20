from zalia_ruta.asr import WordTimestamp
from zalia_ruta.profanity import ProfanityDetector, clean_token


def test_clean_token():
    assert clean_token("...Blet!") == "blet"
    assert clean_token('"Kurva",') == "kurva"
    assert clean_token("nachui?") == "nachui"


def test_profanity_matches():
    detector = ProfanityDetector()

    # Blet variations
    assert detector.is_profanity("blet")
    assert detector.is_profanity("blėt")
    assert detector.is_profanity("bleit")
    assert detector.is_profanity("buliot")
    assert detector.is_profanity("bliat")
    assert detector.is_profanity("bliad")
    assert detector.is_profanity("ble")
    assert detector.is_profanity("bleee")
    assert detector.is_profanity("bliacha")

    # Compound variations
    assert detector.is_profanity("sukanachų")
    assert detector.is_profanity("sukanachui")

    # Nahui variations
    assert detector.is_profanity("nahui")
    assert detector.is_profanity("nachui")
    assert detector.is_profanity("nachuj")
    assert detector.is_profanity("pochui")

    # Suka variations
    assert detector.is_profanity("suka")
    assert detector.is_profanity("sukos")
    assert detector.is_profanity("sukų")
    assert detector.is_profanity("suką")

    # Kurwa variations
    assert detector.is_profanity("kurva")
    assert detector.is_profanity("kurwa")
    assert detector.is_profanity("kurvos")
    assert detector.is_profanity("kurvų")
    assert detector.is_profanity("kurvynas")

    # Additional Slavic/Lithuanian roots
    assert detector.is_profanity("supistas")
    assert detector.is_profanity("atsipisk")
    assert detector.is_profanity("pyzda")

    # Clean non-profane words should NOT match (no false positives)
    assert not detector.is_profanity("labas")
    assert not detector.is_profanity("lietuva")
    assert not detector.is_profanity("dronas")
    assert not detector.is_profanity("sauka")
    assert not detector.is_profanity("kursa")
    assert not detector.is_profanity("blefas")
    assert not detector.is_profanity("sukasi")
    assert not detector.is_profanity("sukau")
    assert not detector.is_profanity("sukti")
    assert not detector.is_profanity("suktas")


def test_find_intervals_and_merging():
    detector = ProfanityDetector(pad_start_s=0.05, pad_end_s=0.05, merge_gap_s=0.10)

    words = [
        WordTimestamp("sveiki", 0.0, 0.5),
        WordTimestamp("suka", 1.0, 1.4),       # profanity: [0.95, 1.45]
        WordTimestamp("blet", 1.5, 1.9),       # profanity: [1.45, 1.95] -> gap is 0.0s <= 0.10s -> merge!
        WordTimestamp("draugai", 2.5, 3.0),
    ]

    intervals = detector.find_intervals(words)
    assert len(intervals) == 1
    assert intervals[0].start_sec == 0.95
    assert intervals[0].end_sec == 1.95
    assert "suka" in intervals[0].original_word
    assert "blet" in intervals[0].original_word
