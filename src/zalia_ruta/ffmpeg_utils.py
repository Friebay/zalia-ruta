import subprocess
import json
from pathlib import Path


def probe_media(file_path: str | Path) -> dict:
    """Probes media file streams and format using ffprobe."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_name,codec_type,sample_rate,channels,bit_rate,duration",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(file_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return json.loads(res.stdout)


def extract_audio_lossless(video_path: str | Path, output_wav_path: str | Path) -> Path:
    """
    Extracts the master audio stream to uncompressed 16-bit PCM WAV.
    This performs lossless decoding of the source audio stream (preserving full quality,
    original sample rate, and channels) so it can be edited sample-by-sample without recompression artifacts.
    """
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "pcm_s16le",
        str(output_wav_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return output_wav_path


def extract_audio_for_asr(video_path: str | Path, output_wav_path: str | Path) -> Path:
    """
    Extracts audio resampled to 16 kHz mono 16-bit PCM WAV,
    the exact standard input format required by Whisper / ASR models.
    """
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_wav_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return output_wav_path


def demux_audio_stream_copy(video_path: str | Path, output_path: str | Path) -> Path:
    """
    Bit-for-bit demuxes the compressed audio stream directly without any decoding (-c:a copy).
    Useful for extracting the raw original bitstream.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "copy",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return output_path


def remux_video_with_audio_and_subtitles(
    video_path: str | Path,
    clean_audio_path: str | Path,
    output_video_path: str | Path,
    subtitles_path: str | Path | None = None,
    audio_bitrate: str = "320k",
) -> Path:
    """
    Combines the original lossless video stream (-c:v copy) with the cleaned audio
    and embeds the word-level subtitles as a soft subtitle track (-c:s mov_text).
    """
    video_path = Path(video_path)
    clean_audio_path = Path(clean_audio_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(clean_audio_path),
    ]

    if subtitles_path is not None and Path(subtitles_path).exists():
        cmd.extend(["-i", str(subtitles_path)])
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:s:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=lit",
            "-metadata:s:s:0", "title=Lietuviški subtitrai (Cenzūruota)",
        ])
    else:
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
        ])

    cmd.append(str(output_video_path))
    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return output_video_path


