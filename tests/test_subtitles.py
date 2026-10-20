from pathlib import Path
from zalia_ruta.asr import WordTimestamp
from zalia_ruta.profanity import BleepInterval
from zalia_ruta.subtitles import format_srt_timestamp, generate_srt


def test_format_srt_timestamp():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(1.25) == "00:00:01,250"
    assert format_srt_timestamp(72.505) == "00:01:12,505"
    assert format_srt_timestamp(3665.1) == "01:01:05,100"


def test_generate_srt(tmp_path: Path):
    from zalia_ruta.profanity import ProfanityDetector

    words = [
        WordTimestamp("labas", 0.0, 0.5),
        WordTimestamp("blet", 1.0, 1.4),
        WordTimestamp("visiems", 2.0, 2.5),
    ]
    detector = ProfanityDetector()

    srt_file = tmp_path / "test.srt"
    content = generate_srt(words, profanity_detector=detector, output_srt_path=srt_file)

    assert "00:00:00,000 --> 00:00:00,500\nlabas" in content
    assert "00:00:01,000 --> 00:00:01,400\n[BLEEP: blet]" in content
    assert "00:00:02,000 --> 00:00:02,500\nvisiems" in content
    assert srt_file.exists()
