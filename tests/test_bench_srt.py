import pytest
from zalia_ruta.bench.srt_io import Cue, parse_srt, parse_srt_timestamp, write_srt
from zalia_ruta.bench.validate import validate_cues


def test_parse_srt_timestamp():
    assert parse_srt_timestamp("00:00:01,500") == 1.5
    assert parse_srt_timestamp("00:01:00.000") == 60.0
    assert parse_srt_timestamp("01:00:00,000") == 3600.0


def test_srt_roundtrip(tmp_path):
    cues = [
        Cue(index=1, start=0.5, end=0.9, text="Labas"),
        Cue(index=2, start=1.2, end=1.8, text="rytas"),
    ]
    srt_file = tmp_path / "test.srt"
    write_srt(cues, srt_file)

    parsed = parse_srt(srt_file.read_text(encoding="utf-8"))
    assert len(parsed) == 2
    assert parsed[0].text == "Labas"
    assert parsed[0].start == 0.5
    assert parsed[0].end == 0.9
    assert parsed[1].text == "rytas"
    assert parsed[1].start == 1.2
    assert parsed[1].end == 1.8


def test_validator_detects_errors():
    # 1. Unresolved disagreement marker '|'
    cues = [Cue(index=1, start=0.0, end=0.5, text="labas|rytas")]
    errors = validate_cues(cues)
    assert any("unresolved disagreement marker" in e for e in errors)

    # 2. Overlap
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="pirmas"),
        Cue(index=2, start=0.8, end=1.5, text="antras"),
    ]
    errors = validate_cues(cues)
    assert any("overlaps" in e for e in errors)

    # 3. Start >= End
    cues = [Cue(index=1, start=1.5, end=1.0, text="klaida")]
    errors = validate_cues(cues)
    assert any("start" in e and ">= end" in e for e in errors)
