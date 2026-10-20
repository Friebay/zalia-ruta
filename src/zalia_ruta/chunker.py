from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    index: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float
    audio: np.ndarray


def find_pause_cuts(
    audio: np.ndarray,
    sr: int = 16000,
    min_chunk_s: float = 20.0,
    max_chunk_s: float = 29.0,
    window_s: float = 0.10,
    hop_s: float = 0.02,
) -> list[AudioChunk]:
    """
    Slices audio into pause-aligned chunks between `min_chunk_s` and `max_chunk_s`.
    Within the search window [min_chunk_s, max_chunk_s], it identifies the point of lowest
    acoustic energy (quietest dip/pause/breath) and cuts there to avoid cutting mid-word.

    Args:
        audio: 1D float numpy array (mono audio samples).
        sr: Sample rate in Hz (default: 16000).
        min_chunk_s: Minimum chunk duration before looking for a pause (default: 20.0s).
        max_chunk_s: Maximum chunk duration allowed (default: 29.0s).
        window_s: Energy measurement window in seconds (default: 100 ms).
        hop_s: Step size in seconds for energy calculation (default: 20 ms).

    Returns:
        List of AudioChunk objects containing sample ranges, timestamps, and audio arrays.
    """
    if audio.ndim > 1:
        # Convert to mono if multi-channel
        audio = audio.mean(axis=1)

    total_samples = len(audio)
    min_samples = int(min_chunk_s * sr)
    max_samples = int(max_chunk_s * sr)
    win_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)

    chunks: list[AudioChunk] = []
    start = 0
    chunk_idx = 0

    while (total_samples - start) > max_samples:
        search_start = start + min_samples
        search_end = start + max_samples

        # Calculate RMS energy across sliding windows in [search_start, search_end]
        min_energy = float("inf")
        best_cut = start + max_samples  # fallback to max if search window is degenerate

        for i in range(search_start, search_end - win_samples + 1, hop_samples):
            window = audio[i : i + win_samples]
            energy = np.mean(window**2)
            if energy < min_energy:
                min_energy = energy
                best_cut = i + win_samples // 2

        end = best_cut
        chunk_audio = audio[start:end]
        chunks.append(
            AudioChunk(
                index=chunk_idx,
                start_sample=start,
                end_sample=end,
                start_sec=start / sr,
                end_sec=end / sr,
                duration_sec=(end - start) / sr,
                audio=chunk_audio,
            )
        )
        start = end
        chunk_idx += 1

    # Remaining tail chunk
    if start < total_samples:
        chunks.append(
            AudioChunk(
                index=chunk_idx,
                start_sample=start,
                end_sample=total_samples,
                start_sec=start / sr,
                end_sec=total_samples / sr,
                duration_sec=(total_samples - start) / sr,
                audio=audio[start:],
            )
        )

    return chunks


def chunk_audio_file(
    audio_path: str | Path,
    min_chunk_s: float = 20.0,
    max_chunk_s: float = 29.0,
) -> tuple[list[AudioChunk], int]:
    """
    Reads an audio file and divides it into pause-aligned chunks.

    Returns:
        tuple (chunks, sample_rate)
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = sf.read(str(audio_path), dtype="float32")
    chunks = find_pause_cuts(
        audio,
        sr=sr,
        min_chunk_s=min_chunk_s,
        max_chunk_s=max_chunk_s,
    )
    return chunks, sr
