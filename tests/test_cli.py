from click.testing import CliRunner
from zalia_ruta.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Zalia-Ruta: Automated Lithuanian Video & Audio Profanity Bleeper" in result.output
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--transcribe-only" in result.output
    assert "--freq" in result.output
    assert "--volume" in result.output
    assert "--backend" in result.output
    assert "--initial-prompt" in result.output
    assert "--no-prompt" in result.output


def test_cli_missing_input():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_cli_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(main, ["-i", "non_existent_file.mp4"])
    assert result.exit_code != 0


def test_cli_transcribe_only(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from zalia_ruta.asr import WordTimestamp

    input_video = tmp_path / "test_input.mp4"
    input_video.write_bytes(b"dummy video content")
    out_srt = tmp_path / "test_input.srt"

    # Mock media probing and pipeline steps
    monkeypatch.setattr(
        "zalia_ruta.cli.probe_media",
        lambda p: {"streams": [{"codec_type": "audio"}, {"codec_type": "video"}]},
    )
    monkeypatch.setattr("zalia_ruta.cli.extract_audio_for_asr", lambda inp, out: None)
    monkeypatch.setattr("zalia_ruta.cli.chunk_audio_file", lambda p: ([(0.0, 5.0)], 16000))

    mock_asr_instance = MagicMock()
    mock_asr_instance.transcribe_chunks.return_value = [
        WordTimestamp(word="labas", start=0.5, end=1.0),
        WordTimestamp(word="kurva", start=1.2, end=1.8),
    ]
    monkeypatch.setattr("zalia_ruta.cli.LithuanianASR", lambda **kwargs: mock_asr_instance)

    mock_bleep = MagicMock()
    monkeypatch.setattr("zalia_ruta.cli.apply_bleep_intervals", mock_bleep)
    mock_remux = MagicMock()
    monkeypatch.setattr("zalia_ruta.cli.remux_video_with_audio_and_subtitles", mock_remux)
    mock_lossless = MagicMock()
    monkeypatch.setattr("zalia_ruta.cli.extract_audio_lossless", mock_lossless)

    runner = CliRunner()
    result = runner.invoke(main, ["-i", str(input_video), "-o", str(out_srt), "--transcribe-only"])
    assert result.exit_code == 0, result.output
    assert "Success! Word-level SRT subtitles saved" in result.output

    # Verbatim words preserved without [BLEEP: ...] tags
    content = out_srt.read_text(encoding="utf-8")
    assert "labas" in content
    assert "kurva" in content
    assert "[BLEEP" not in content

    # Ensure bleeping, lossless master audio extraction, and video remux were NOT invoked
    mock_bleep.assert_not_called()
    mock_lossless.assert_not_called()
    mock_remux.assert_not_called()

