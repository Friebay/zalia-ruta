import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from zalia_ruta.highlights import export_highlight_clip, find_profanity_highlight_windows
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
    "--mms-align/--no-mms-align",
    default=True,
    show_default=True,
    help="Use MMS forced alignment for word boundary precision (replaces raw Whisper timestamps).",
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
@click.option(
    "--transcribe-only",
    is_flag=True,
    default=False,
    help="Only perform transcription and export verbatim .srt subtitles without bleeping or video remuxing.",
)
@click.option(
    "--backend",
    type=click.Choice(["transformers", "faster-whisper", "auto"], case_sensitive=False),
    default="transformers",
    show_default=True,
    help="ASR backend engine (transformers for highest accuracy 13.4% WER, or faster-whisper for 5x speed and low VRAM).",
)
@click.option(
    "--model",
    "model_id",
    type=str,
    default=None,
    help="Custom model path or Hugging Face ID (defaults to paprika-whisper-lt-v3).",
)
@click.option(
    "--highlight-clips",
    type=int,
    default=None,
    help="Extract Top N ~1-minute pause-aligned clips with highest profanity density (raw MP4 + 16kHz WAV + verbatim SRT).",
)
@click.option(
    "--initial-prompt",
    type=str,
    default=None,
    help="Custom initial prompt to prime Whisper decoder (defaults to built-in Lithuanian streamer vocabulary).",
)
@click.option(
    "--no-prompt",
    is_flag=True,
    default=False,
    help="Disable Whisper initial prompt conditioning completely.",
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
    backend: str,
    model_id: Optional[str],
    mms_align: bool,
    export_srt: bool,
    keep_temp: bool,
    transcribe_only: bool,
    highlight_clips: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    no_prompt: bool = False,
) -> None:
    """
    Zalia-Ruta: Automated Lithuanian Video & Audio Profanity Bleeper.

    Losslessly preserves video stream quality (-c:v copy), transcribes with local
    Lithuanian Whisper ASR, detects inflected profanities, applies crossfaded 1000 Hz
    bleeps, and embeds soft synchronized subtitles.
    """
    click.echo("=" * 60)
    title = "🌿 Zalia-Ruta: Lithuanian ASR Transcriber" if transcribe_only else "🌿 Zalia-Ruta: Lithuanian Profanity Bleeper"
    click.echo(title)
    click.echo("=" * 60)
    click.echo(f"Input: {input_path}")

    # Determine default output path if not provided
    if output_path is None:
        target_dir = Path("output") / input_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        if transcribe_only:
            output_path = target_dir / f"{input_path.stem}.srt"
        else:
            output_path = target_dir / f"{input_path.stem}_bleeped{input_path.suffix}"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output: {output_path}")

    # Verify input has audio streams
    total_steps = 4 if transcribe_only else 6
    click.echo(f"\n[1/{total_steps}] Probing media file...")
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
        srt_path = output_path if output_path.suffix.lower() == ".srt" else output_path.with_suffix(".srt")

        # Step 1: Audio Extraction
        if not transcribe_only:
            click.echo(f"\n[2/{total_steps}] Extracting lossless master audio & 16kHz ASR audio...")
            extract_audio_lossless(input_path, master_wav)
        else:
            click.echo(f"\n[2/{total_steps}] Extracting 16kHz ASR audio...")
        extract_audio_for_asr(input_path, asr_wav)

        # Step 2: Pause-Aligned Audio Chunking
        click.echo(f"\n[3/{total_steps}] Slicing audio into pause-aligned speech chunks...")
        chunks, sr = chunk_audio_file(asr_wav)
        click.echo(f"  Created {len(chunks)} speech chunk(s) (20s-29s pause boundaries).")

        # Step 3: Transcription & MMS Forced Alignment
        align_desc = "with MMS forced alignment" if mms_align else "(raw Whisper timestamps)"
        backend_desc = f" [{backend}]" if backend != "auto" else ""
        click.echo(f"\n[4/{total_steps}] Transcribing Lithuanian speech {align_desc} on {device or 'auto'}{backend_desc}...")

        if model_id is None:
            if backend == "faster-whisper":
                local_ct2 = Path("models/paprika-whisper-lt-v3-ct2-int8_float16")
                resolved_model = str(local_ct2) if local_ct2.exists() else "RobertasTa/paprika-whisper-lt-v3-ct2-int8"
            else:
                resolved_model = "kristijonas/paprika-whisper-lt-v3"
        else:
            resolved_model = model_id

        if device != "cpu" and backend == "transformers":
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory <= 4.5 * (1024 ** 3):
                    click.secho("  Tip: Detected <=4GB VRAM GPU. If transcription feels slow, use '--backend faster-whisper' for 5x speed.", fg="cyan")
            except Exception:
                pass

        from zalia_ruta.asr import DEFAULT_INITIAL_PROMPT
        asr_prompt = "" if no_prompt else (initial_prompt if initial_prompt is not None else DEFAULT_INITIAL_PROMPT)

        asr = LithuanianASR(
            model_id=resolved_model,
            device=device,
            use_mms_align=mms_align,
            backend=backend,
            initial_prompt=asr_prompt,
        )
        words = asr.transcribe_chunks(chunks, sr=sr)
        click.echo(f"  Transcribed {len(words)} words total.")

        detector = ProfanityDetector(
            custom_words_file=custom_words,
            pad_start_s=pad_start,
            pad_end_s=pad_end,
        )

        if highlight_clips and highlight_clips > 0:
            click.echo(f"\n[Highlights] Finding Top {highlight_clips} pause-aligned profanity highlight windows...")
            windows = find_profanity_highlight_windows(
                words=words,
                detector=detector,
                top_k=highlight_clips,
                target_duration_sec=60.0,
                min_duration_sec=45.0,
                max_duration_sec=75.0,
            )
            if not windows:
                click.secho("  No profanity highlight windows found.", fg="yellow")
            else:
                highlights_dir = output_path.parent / "highlights"
                highlights_dir.mkdir(parents=True, exist_ok=True)
                click.echo(f"  Exporting {len(windows)} raw highlight clips to: {highlights_dir}")
                
                report_lines = [
                    f"# Profanity Highlights Report: {input_path.name}",
                    f"- Total words transcribed: {len(words)}",
                    f"- Highlight clips extracted: {len(windows)}\n",
                    "| Rank | Start | End | Duration | Profanities | Profane Words |",
                    "|------|-------|-----|----------|-------------|---------------|",
                ]

                for win in windows:
                    click.echo(f"  -> Clip #{win.rank}: [{win.start_sec:.1f}s - {win.end_sec:.1f}s] ({win.duration_sec:.1f}s) - {win.profanity_count} profanities")
                    mp4_p, wav_p, srt_p = export_highlight_clip(
                        input_media_path=input_path,
                        window=win,
                        words=words,
                        output_dir=highlights_dir,
                        clip_prefix=f"{input_path.stem[:25]}_highlight",
                    )
                    words_summary = ", ".join(win.profane_words[:8])
                    if len(win.profane_words) > 8:
                        words_summary += f" (+{len(win.profane_words)-8} more)"
                    report_lines.append(
                        f"| {win.rank} | {win.start_sec:.2f}s | {win.end_sec:.2f}s | {win.duration_sec:.2f}s | {win.profanity_count} | {words_summary} |"
                    )

                report_path = highlights_dir / "highlights_report.md"
                report_path.write_text("\n".join(report_lines), encoding="utf-8")
                click.secho(f"  Report written to: {report_path}", fg="green")

        if transcribe_only:
            generate_srt(
                words=words,
                profanity_detector=None,
                output_srt_path=srt_path,
                mark_bleeps=False,
            )
            click.secho("\n" + "=" * 60, fg="green")
            click.secho(f"✅ Success! Word-level SRT subtitles saved to: {srt_path}", fg="green")
            click.secho("=" * 60, fg="green")
            return

        # Step 4: Profanity Detection
        click.echo("\n[5/6] Scanning transcript for profanities & inflections...")
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
