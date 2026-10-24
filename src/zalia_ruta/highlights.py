import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from zalia_ruta.asr import WordTimestamp
from zalia_ruta.profanity import ProfanityDetector, clean_token
from zalia_ruta.subtitles import generate_srt


@dataclass
class HighlightWindow:
    rank: int
    start_sec: float
    end_sec: float
    duration_sec: float
    profanity_count: int
    profane_words: List[str]
    total_words: int
    start_pause_duration: float
    end_pause_duration: float


def find_pause_cut_points(
    words: List[WordTimestamp],
    total_duration_sec: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Finds valid cut points between words.
    Returns a list of (timestamp_sec, pause_gap_sec).
    Guarantees cuts will never occur inside a word.
    """
    if not words:
        return []

    cut_points: List[Tuple[float, float]] = []

    # Beginning of audio before first word
    first_gap = max(0.0, words[0].start)
    cut_points.append((0.0, first_gap))

    # Gaps between consecutive words
    for i in range(len(words) - 1):
        prev_end = words[i].end
        next_start = words[i + 1].start
        gap = max(0.0, next_start - prev_end)
        # Midpoint of the gap
        cut_time = round(prev_end + gap / 2.0, 3)
        cut_points.append((cut_time, gap))

    # End of audio after last word
    last_word_end = words[-1].end
    end_time = total_duration_sec if total_duration_sec and total_duration_sec > last_word_end else last_word_end
    last_gap = max(0.0, end_time - last_word_end)
    cut_points.append((round(end_time, 3), last_gap))

    return cut_points


def find_profanity_highlight_windows(
    words: List[WordTimestamp],
    detector: ProfanityDetector,
    top_k: int = 5,
    target_duration_sec: float = 60.0,
    min_duration_sec: float = 45.0,
    max_duration_sec: float = 75.0,
    min_gap_between_clips_sec: float = 10.0,
    total_duration_sec: Optional[float] = None,
) -> List[HighlightWindow]:
    """
    Scans word transcript using candidate pause cut points to find top non-overlapping
    windows with the highest density and count of profanities.
    """
    if not words:
        return []

    # Tag profane words and their indices
    profane_indices = set()
    for i, w in enumerate(words):
        if detector.is_profanity(w.word):
            profane_indices.add(i)

    cut_points = find_pause_cut_points(words, total_duration_sec=total_duration_sec)
    if len(cut_points) < 2:
        return []

    candidates: List[dict] = []

    # Evaluate candidate window pairs (start_cut, end_cut)
    right_idx = 0
    num_cuts = len(cut_points)

    import bisect

    word_starts = [w.start for w in words]
    word_ends = [w.end for w in words]

    for left_idx in range(num_cuts):
        start_cut, start_gap = cut_points[left_idx]

        while right_idx < num_cuts and (cut_points[right_idx][0] - start_cut) < min_duration_sec:
            right_idx += 1

        curr_r = right_idx
        while curr_r < num_cuts:
            end_cut, end_gap = cut_points[curr_r]
            dur = end_cut - start_cut
            if dur > max_duration_sec:
                break

            # Fast binary search slice
            w_start_idx = bisect.bisect_left(word_starts, start_cut - 0.05)
            w_end_idx = bisect.bisect_right(word_ends, end_cut + 0.05)

            matching_profanities = []
            matching_words_count = 0
            for idx in range(w_start_idx, min(len(words), w_end_idx)):
                w = words[idx]
                if w.start >= start_cut - 0.05 and w.end <= end_cut + 0.05:
                    matching_words_count += 1
                    if idx in profane_indices:
                        matching_profanities.append(w.word)

            prof_count = len(matching_profanities)
            if prof_count > 0:
                # Score prioritizes profanity count, then pause quality, then proximity to target_duration_sec
                pause_score = min(start_gap, 1.0) + min(end_gap, 1.0)
                dur_penalty = abs(dur - target_duration_sec) * 0.05
                score = (prof_count * 100.0) + (pause_score * 5.0) - dur_penalty

                candidates.append({
                    "start_sec": start_cut,
                    "end_sec": end_cut,
                    "duration_sec": dur,
                    "profanity_count": prof_count,
                    "profane_words": matching_profanities,
                    "total_words": matching_words_count,
                    "start_pause_duration": start_gap,
                    "end_pause_duration": end_gap,
                    "score": score,
                })

            curr_r += 1

    # Sort candidates by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Greedy Non-Maximum Suppression (NMS) to avoid overlapping highlight clips
    selected: List[HighlightWindow] = []
    for cand in candidates:
        if len(selected) >= top_k:
            break

        # Check overlap with already selected windows
        overlaps = False
        for s in selected:
            # Overlap occurs if candidate starts before (s.end + min_gap) and ends after (s.start - min_gap)
            if not (cand["end_sec"] + min_gap_between_clips_sec <= s.start_sec or
                    cand["start_sec"] >= s.end_sec + min_gap_between_clips_sec):
                overlaps = True
                break

        if not overlaps:
            selected.append(
                HighlightWindow(
                    rank=len(selected) + 1,
                    start_sec=round(cand["start_sec"], 3),
                    end_sec=round(cand["end_sec"], 3),
                    duration_sec=round(cand["duration_sec"], 3),
                    profanity_count=cand["profanity_count"],
                    profane_words=cand["profane_words"],
                    total_words=cand["total_words"],
                    start_pause_duration=round(cand["start_pause_duration"], 3),
                    end_pause_duration=round(cand["end_pause_duration"], 3),
                )
            )

    return selected


def export_highlight_clip(
    input_media_path: Path | str,
    window: HighlightWindow,
    words: List[WordTimestamp],
    output_dir: Path | str,
    clip_prefix: str = "highlight",
) -> Tuple[Path, Path, Path]:
    """
    Extracts raw uncompressed 16kHz WAV, high-quality H.264 MP4, and normalized verbatim SRT.
    Returns (mp4_path, wav_path, srt_path).
    """
    input_media_path = Path(input_media_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{clip_prefix}_{window.rank:02d}_{int(window.start_sec)}s_{int(window.end_sec)}s"
    mp4_path = output_dir / f"{stem}.mp4"
    wav_path = output_dir / f"{stem}.wav"
    srt_path = output_dir / f"{stem}.srt"

    start_str = f"{window.start_sec:.3f}"
    dur_str = f"{window.duration_sec:.3f}"

    # 1. Cut video losslessly/frame-accurately with high quality (crf 18, audio untouched/high quality)
    cmd_video = [
        "ffmpeg",
        "-y",
        "-ss", start_str,
        "-i", str(input_media_path),
        "-t", dur_str,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        str(mp4_path),
    ]
    subprocess.run(cmd_video, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

    # 2. Extract 16 kHz mono WAV for Subtitle Edit / Benchmark
    cmd_wav = [
        "ffmpeg",
        "-y",
        "-ss", start_str,
        "-i", str(input_media_path),
        "-t", dur_str,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd_wav, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

    # 3. Create normalized word timestamps for SRT
    clip_words: List[WordTimestamp] = []
    for w in words:
        if w.start >= window.start_sec - 0.05 and w.end <= window.end_sec + 0.05:
            rel_start = max(0.0, round(w.start - window.start_sec, 3))
            rel_end = max(rel_start + 0.05, round(w.end - window.start_sec, 3))
            clip_words.append(
                WordTimestamp(
                    word=w.word,
                    start=rel_start,
                    end=rel_end,
                )
            )

    generate_srt(
        words=clip_words,
        profanity_detector=None,
        output_srt_path=srt_path,
        mark_bleeps=False,
    )

    return mp4_path, wav_path, srt_path
