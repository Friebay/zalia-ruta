from typing import List, Optional
import numpy as np

from zalia_ruta.asr import WordTimestamp


def compute_frame_rms(
    audio: np.ndarray,
    win_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """
    Computes root-mean-square (RMS) energy across sliding windows using fast cumulative sums.
    """
    if len(audio) < win_samples:
        return np.array([np.sqrt(np.mean(audio**2))] if len(audio) > 0 else [0.0], dtype=np.float32)

    sq = audio.astype(np.float32) ** 2
    csum = np.pad(np.cumsum(sq), (1, 0), mode="constant")
    window_sums = csum[win_samples:] - csum[:-win_samples]
    window_means = np.maximum(0.0, window_sums[::hop_samples] / win_samples)
    return np.sqrt(window_means)


def refine_word_boundaries(
    audio: np.ndarray,
    words: List[WordTimestamp],
    sr: int = 16000,
    win_ms: float = 25.0,
    hop_ms: float = 5.0,
    energy_threshold_ratio: float = 0.12,
    max_onset_extend_s: float = 0.025,
    max_offset_extend_s: float = 0.045,
    min_pause_clearance_s: float = 0.030,
) -> List[WordTimestamp]:
    """
    Acoustically refines MMS-aligned word boundaries to eliminate negative end bias
    and onset truncation without bleeding into inter-word pauses.

    Args:
        audio: 1D float numpy array (mono audio samples).
        words: List of WordTimestamp objects from MMS forced alignment.
        sr: Sample rate in Hz (default: 16000).
        win_ms: RMS measurement window in milliseconds (default: 25 ms).
        hop_ms: RMS hop step in milliseconds (default: 5 ms).
        energy_threshold_ratio: Fraction of local word peak RMS considered speech (default: 0.12 = 12%).
        max_onset_extend_s: Maximum backward extension for word onset (default: 25 ms).
        max_offset_extend_s: Maximum forward extension for vocalic decay (default: 45 ms).
        min_pause_clearance_s: Safety margin to prevent adjacent words from touching (default: 30 ms).

    Returns:
        New List of WordTimestamp objects with acoustically refined start and end times.
    """
    if not words or len(audio) == 0:
        return words

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    win_samples = max(1, int((win_ms / 1000.0) * sr))
    hop_samples = max(1, int((hop_ms / 1000.0) * sr))

    rms = compute_frame_rms(audio, win_samples, hop_samples)
    num_frames = len(rms)
    if num_frames == 0:
        return words

    # Estimate background noise floor from lower 10th percentile of frames
    noise_floor = float(np.percentile(rms, 10))
    total_audio_s = len(audio) / sr

    def time_to_frame(t: float) -> int:
        return int(round(t * sr / hop_samples))

    refined: List[WordTimestamp] = []
    prev_end = 0.0

    for i, w in enumerate(words):
        orig_start = max(0.0, w.start)
        orig_end = min(total_audio_s, max(orig_start + 0.02, w.end))

        # Next word's start constraint
        if i + 1 < len(words):
            next_start = max(orig_end, words[i + 1].start)
            initial_gap = next_start - orig_end
        else:
            next_start = total_audio_s
            initial_gap = total_audio_s - orig_end

        # Find peak RMS within the MMS word span
        s_frame = min(num_frames - 1, max(0, time_to_frame(orig_start)))
        e_frame = min(num_frames - 1, max(s_frame + 1, time_to_frame(orig_end)))

        word_rms_slice = rms[s_frame:e_frame]
        peak_rms = float(np.max(word_rms_slice)) if len(word_rms_slice) > 0 else 0.0

        if peak_rms <= noise_floor * 1.2:
            # Word span has virtually no energy above noise; preserve original bounds safely
            s_clamped = max(prev_end + min_pause_clearance_s if i > 0 else 0.0, orig_start)
            e_clamped = min(next_start - min_pause_clearance_s if i + 1 < len(words) else total_audio_s, orig_end)
            refined.append(WordTimestamp(word=w.word, start=round(s_clamped, 3), end=round(max(s_clamped + 0.03, e_clamped), 3), score=w.score))
            prev_end = refined[-1].end
            continue

        threshold = max(noise_floor * 1.25, peak_rms * energy_threshold_ratio)

        # --- 1. Backward Onset Search ---
        # Bound backward search so it does not collide with previous word
        left_bound_s = prev_end + min_pause_clearance_s if i > 0 else 0.0
        earliest_allowed_s = max(left_bound_s, orig_start - max_onset_extend_s)
        earliest_frame = time_to_frame(earliest_allowed_s)

        curr_frame = s_frame
        while curr_frame > earliest_frame and rms[curr_frame - 1] >= threshold:
            curr_frame -= 1

        new_start = round(curr_frame * (hop_samples / sr), 3)
        new_start = max(left_bound_s, min(new_start, orig_start))

        # --- 2. Forward Offset / Decay Search ---
        # If there was a distinct pause gap (>150 ms) to the next word, don't bridge more than 40% of it
        if initial_gap > 0.150:
            allowed_forward_s = min(max_offset_extend_s, initial_gap * 0.40)
        else:
            allowed_forward_s = min(max_offset_extend_s, max(0.0, initial_gap - min_pause_clearance_s))

        right_bound_s = next_start - min_pause_clearance_s if i + 1 < len(words) else total_audio_s
        latest_allowed_s = min(right_bound_s, orig_end + allowed_forward_s)
        latest_frame = min(num_frames - 1, time_to_frame(latest_allowed_s))

        curr_frame = e_frame
        while curr_frame < latest_frame and rms[curr_frame] >= threshold:
            curr_frame += 1

        new_end = round(curr_frame * (hop_samples / sr), 3)
        new_end = min(right_bound_s, max(new_end, orig_end))
        if new_end <= new_start:
            new_end = new_start + 0.030

        refined.append(
            WordTimestamp(
                word=w.word,
                start=round(new_start, 3),
                end=round(new_end, 3),
                score=w.score,
            )
        )
        prev_end = new_end

    return refined
