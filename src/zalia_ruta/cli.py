import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click

from zalia_ruta.asr import LithuanianASR
from zalia_ruta.audio_editor import apply_bleep_intervals
from zalia_ruta.chunker import chunk_audio_file
from zalia_ruta.ffmpeg_utils import (
    extract_audio_for_asr,
    extract_audio_lossless,
    probe_media,
    remux_video_with_audio_and_subtitles,
)
from zalia_ruta.profanity import ProfanityDetector
from zalia_ruta.subtitles import generate_srt


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to input video or audio file.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to output file. Defaults to <input_stem>_bleeped<ext>.",
)
@click.option(
    "--custom-words",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional text file with additional profanity words or regex patterns (one per line).",
)
@click.option(
    "--freq",
    type=float,
    default=1000.0,
    show_default=True,
    help="Bleep sine wave tone frequency in Hz.",
)
@click.option(
    "--volume",
    type=float,
    default=0.35,
    show_default=True,
    help="Bleep sine wave tone volume (0.0 to 1.0).",
)
@click.option(
    "--pad-start",
    type=float,
    default=0.04,
    show_default=True,
    help="Safety padding in seconds added before profanity onset.",
)
@click.option(
    "--pad-end",
    type=float,
    default=0.04,
    show_default=True,
    help="Safety padding in seconds added after profanity offset.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default=None,
    help="Inference device for Whisper model (default: auto-detected cuda if available, else cpu).",
)
@click.option(
    "--export-srt/--no-export-srt",
    default=True,
    show_default=True,
    help="Export standalone .srt subtitles file alongside the output video.",
)
@click.option(
    "--keep-temp/--no-keep-temp",
    default=False,
    show_default=True,
    help="Keep intermediate extracted WAV files instead of cleaning them up.",
)
def main(
    input_path: Path,
    output_path: Optional[Path],
    custom_words: Optional[Path],
    freq: float,
    volume: float,
    pad_start: float,
    pad_end: float,
    device: Optional[str],
    export_srt: bool,
    keep_temp: bool,
) -> None:
    """
    Zalia-Ruta: Automated Lithuanian Video & Audio Profanity Bleeper.

    Losslessly preserves video stream quality (-c:v copy), transcribes with local
    Lithuanian Whisper ASR, detects inflected profanities, applies crossfaded 1000 Hz
    bleeps, and embeds soft synchronized subtitles.
    """
    click.echo("=" * 60)
    click.echo("🌿 Zalia-Ruta: Lithuanian Profanity Bleeper")
    click.echo("=" * 60)
    click.echo(f"Input: {input_path}")

    # Determine default output path if not provided
    if output_path is None:
        target_dir = Path("output") / input_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{input_path.stem}_bleeped{input_path.suffix}"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output: {output_path}")

    # Verify input has audio streams
    click.echo("\n[1/6] Probing media file...")
    try:
        media_info = probe_media(input_path)
    except Exception as e:
        click.secho(f"Error reading media file with ffprobe: {e}", fg="red", err=True)
        sys.exit(1)

    streams = media_info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)

    if not has_audio:
        click.secho("Error: Input file contains no audio stream to process.", fg="red", err=True)
        sys.exit(1)

    click.echo(f"  Detected streams: Video={has_video}, Audio={has_audio}")

    # Setup temporary directory for working WAV files
    temp_dir_obj = None
    if keep_temp:
        temp_dir = input_path.parent / f"{input_path.stem}_zalia_ruta_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"  Working directory (persisted): {temp_dir}")
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="zalia_ruta_")
        temp_dir = Path(temp_dir_obj.name)

    try:
        master_wav = temp_dir / "master.wav"
        asr_wav = temp_dir / "asr_16k.wav"
        clean_wav = temp_dir / "clean.wav"
        srt_path = output_path.with_suffix(".srt")

        # Step 1: Lossless Audio Extraction
        click.echo("\n[2/6] Extracting lossless master audio & 16kHz ASR audio...")
        extract_audio_lossless(input_path, master_wav)
        extract_audio_for_asr(input_path, asr_wav)

        # Step 2: Pause-Aligned Audio Chunking
        click.echo("\n[3/6] Slicing audio into pause-aligned speech chunks...")
        chunks, sr = chunk_audio_file(asr_wav)
        click.echo(f"  Created {len(chunks)} speech chunk(s) (20s-29s pause boundaries).")

        # Step 3: Transcription & Acoustic Refinement
        click.echo(f"\n[4/6] Transcribing Lithuanian speech on {device or 'auto'}...")
        asr = LithuanianASR(device=device)
        words = asr.transcribe_chunks(chunks, sr=sr)
        click.echo(f"  Transcribed {len(words)} words total.")

        # Step 4: Profanity Detection
        click.echo("\n[5/6] Scanning transcript for profanities & inflections...")
        detector = ProfanityDetector(
            custom_words_file=custom_words,
            pad_start_s=pad_start,
            pad_end_s=pad_end,
        )
        intervals = detector.find_intervals(words)

        if not intervals:
            click.secho("\n✨ No profanities detected! The audio is clean.", fg="green")
            click.echo("Creating output copy without modifications...")
            if has_video:
                shutil.copyfile(input_path, output_path)
            else:
                shutil.copyfile(input_path, output_path)
            if export_srt:
                generate_srt(words, profanity_detector=detector, output_srt_path=srt_path)
                click.echo(f"Saved subtitles: {srt_path}")
            click.secho(f"\nCompleted successfully: {output_path}", fg="green")
            return

        click.secho(f"  Found {len(intervals)} profanity interval(s) to bleep:", fg="yellow")
        for inv in intervals:
            click.echo(f"    - {inv.start_sec:6.2f}s -> {inv.end_sec:6.2f}s: {inv.matched_word} (token: '{inv.original_word}')")

        # Step 5: Bleep Tone Application
        click.echo("\n[6/6] Applying 1000 Hz bleep tone with 5 ms crossfading...")
        apply_bleep_intervals(
            input_wav_path=master_wav,
            output_wav_path=clean_wav,
            intervals=intervals,
            tone_freq=freq,
            tone_volume=volume,
        )

        # Step 6: Generate Subtitles & Remux
        if export_srt or has_video:
            generate_srt(
                words=words,
                profanity_detector=detector,
                output_srt_path=srt_path,
                mark_bleeps=True,
            )

        if has_video:
            click.echo("  Remuxing video stream losslessly (-c:v copy) with bleeped audio and subtitles...")
            remux_video_with_audio_and_subtitles(
                video_path=input_path,
                clean_audio_path=clean_wav,
                output_video_path=output_path,
                subtitles_path=srt_path,
            )
        else:
            # Audio-only file
            click.echo("  Saving cleaned audio file...")
            shutil.copyfile(clean_wav, output_path)

        if not export_srt and srt_path.exists():
            srt_path.unlink()

        click.secho("\n" + "=" * 60, fg="green")
        click.secho(f"✅ Success! Censored video saved to: {output_path}", fg="green")
        if export_srt:
            click.secho(f"📄 Word-level SRT subtitles saved to: {srt_path}", fg="green")
        click.secho("=" * 60, fg="green")

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
