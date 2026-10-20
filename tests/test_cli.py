from click.testing import CliRunner
from zalia_ruta.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Zalia-Ruta: Automated Lithuanian Video & Audio Profanity Bleeper" in result.output
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--freq" in result.output
    assert "--volume" in result.output


def test_cli_missing_input():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_cli_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(main, ["-i", "non_existent_file.mp4"])
    assert result.exit_code != 0
