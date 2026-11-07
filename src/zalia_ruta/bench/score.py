import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import jiwer
import numpy as np

from zalia_ruta.asr import WordTimestamp
from zalia_ruta.bench.mms_align import _LT_DIACRITICS
from zalia_ruta.bench.srt_io import Cue, parse_srt


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_word(w: str) -> str:
    """Lowercases, strips punctuation, and normalizes approved brand name variants."""
    w = w.lower().strip()
    w = _PUNCT_RE.sub("", w)

    # Approved Brand Name Equivalence Normalization
    if w.startswith("facebook"):
        w = "feisbuk" + w[8:]
    elif w in ("youtubes", "youtubės", "jutubes"):
        w = "jutubės"
    elif w.startswith("youtube"):
        w = "jutub" + w[7:]
    elif w.startswith("youtub"):
        w = "jutub" + w[6:]
    elif w.startswith("discord"):
        w = "diskord" + w[7:]
    elif w.startswith("dicord"):
        w = "diskord" + w[6:]
    elif w.startswith("edge"):
        w = "edž" + w[4:]

    return w


def strip_lt_diacritics(w: str) -> str:
    """Maps Lithuanian diacritic characters to plain Latin."""
    return w.translate(_LT_DIACRITICS)


@dataclass
class TimingMetrics:
    matched_words: int
    start_med_ms: float
    start_mean_ms: float
    start_p90_ms: float
    end_med_ms: float
    end_mean_ms: float
    end_p90_ms: float
    within_25ms_pct: float
    within_50ms_pct: float
    within_100ms_pct: float
    start_bias_ms: float
    end_bias_ms: float
    pause_bleed_pct: float


@dataclass
class ScoreResult:
    system_id: str
    clip_id: str
    ref_words_count: int
    hyp_words_count: int
    wer: float
    cer: float
    wer_no_diacritics: float
    timing: Optional[TimingMetrics] = None


def is_filler_or_special(word: str) -> bool:
    w = word.strip().lower()
    return w.startswith("~") or w.endswith("-") or w == "???" or w == "#skip"


def filter_cues_and_regions(cues: List[Cue]) -> Tuple[List[Cue], List[Tuple[float, float]]]:
    """
    Separates cues into evaluable words and excluded #SKIP time intervals.
    """
    skip_intervals: List[Tuple[float, float]] = []
    eval_cues: List[Cue] = []

    for c in cues:
        raw_text = c.text.strip()
        if raw_text == "#SKIP":
            skip_intervals.append((c.start, c.end))
        elif not is_filler_or_special(raw_text):
            norm = normalize_word(raw_text)
            if norm:
                eval_cues.append(Cue(index=c.index, start=c.start, end=c.end, text=norm))

    return eval_cues, skip_intervals


def is_in_skip(start: float, end: float, skip_intervals: List[Tuple[float, float]]) -> bool:
    for s_skip, e_skip in skip_intervals:
        if max(start, s_skip) < min(end, e_skip):
            return True
    return False


def calculate_pause_bleed(
    ref_cues: List[Cue],
    hyp_words: List[WordTimestamp],
    min_gap_s: float = 0.15,
) -> float:
    """
    Calculates pause bleed percentage:
    Fraction of reference silence gaps (>= 150ms) covered by hypothesis words.
    """
    if len(ref_cues) < 2:
        return 0.0

    # 1. Identify reference gaps >= min_gap_s
    gaps: List[Tuple[float, float]] = []
    for i in range(len(ref_cues) - 1):
        g_start = ref_cues[i].end
        g_end = ref_cues[i + 1].start
        if g_end - g_start >= min_gap_s:
            gaps.append((g_start, g_end))

    total_gap_duration = sum(g_end - g_start for g_start, g_end in gaps)
    if total_gap_duration <= 0:
        return 0.0

    # 2. Measure hypothesis overlap in gaps
    bleed_duration = 0.0
    for g_start, g_end in gaps:
        for hw in hyp_words:
            # Overlap between hw and gap
            overlap_start = max(g_start, hw.start)
            overlap_end = min(g_end, hw.end)
            if overlap_end > overlap_start:
                bleed_duration += (overlap_end - overlap_start)

    bleed_pct = (bleed_duration / total_gap_duration) * 100.0
    return round(float(bleed_pct), 2)


def compute_alignment_matches(
    ref_words: List[Cue],
    hyp_words: List[WordTimestamp],
) -> List[Tuple[Cue, WordTimestamp]]:
    """
    Finds exact matching words between ref and hyp in monotonic order using edit distance.
    """
    ref_seq = [c.text for c in ref_words]
    hyp_seq = [normalize_word(h.word) for h in hyp_words]

    # Standard Wagner-Fischer DP for word alignment
    n = len(ref_seq)
    m = len(hyp_seq)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_seq[i - 1] == hyp_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrace
    matches: List[Tuple[Cue, WordTimestamp]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if ref_seq[i - 1] == hyp_seq[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            matches.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches


def score_system(
    system_id: str,
    clip_id: str,
    ref_cues: List[Cue],
    hyp_words: List[WordTimestamp],
    has_end_times: bool = True,
) -> ScoreResult:
    """
    Scores a single hypothesis against reference cues.
    """
    # 1. Clean and filter ref and hyp
    clean_refs, skip_intervals = filter_cues_and_regions(ref_cues)
    clean_hyps = [
        h for h in hyp_words
        if not is_filler_or_special(h.word)
        and normalize_word(h.word)
        and not is_in_skip(h.start, h.end, skip_intervals)
    ]

    ref_str = " ".join([c.text for c in clean_refs])
    hyp_str = " ".join([normalize_word(h.word) for h in clean_hyps])

    wer = float(jiwer.wer(ref_str, hyp_str)) if ref_str else 0.0
    cer = float(jiwer.cer(ref_str, hyp_str)) if ref_str else 0.0

    ref_no_dia = strip_lt_diacritics(ref_str)
    hyp_no_dia = strip_lt_diacritics(hyp_str)
    wer_no_dia = float(jiwer.wer(ref_no_dia, hyp_no_dia)) if ref_no_dia else 0.0

    # 2. Timing metrics
    matches = compute_alignment_matches(clean_refs, clean_hyps)
    if not matches:
        return ScoreResult(
            system_id=system_id,
            clip_id=clip_id,
            ref_words_count=len(clean_refs),
            hyp_words_count=len(clean_hyps),
            wer=round(wer * 100, 2),
            cer=round(cer * 100, 2),
            wer_no_diacritics=round(wer_no_dia * 100, 2),
            timing=None,
        )

    start_diffs = np.array([(m[1].start - m[0].start) * 1000.0 for m in matches])
    start_abs = np.abs(start_diffs)

    if has_end_times:
        end_diffs = np.array([(m[1].end - m[0].end) * 1000.0 for m in matches])
        end_abs = np.abs(end_diffs)
        both_25 = np.mean((start_abs <= 25) & (end_abs <= 25)) * 100.0
        both_50 = np.mean((start_abs <= 50) & (end_abs <= 50)) * 100.0
        both_100 = np.mean((start_abs <= 100) & (end_abs <= 100)) * 100.0
        pause_bleed = calculate_pause_bleed(clean_refs, clean_hyps)
        end_med = float(np.median(end_abs))
        end_mean = float(np.mean(end_abs))
        end_p90 = float(np.percentile(end_abs, 90))
        end_bias = float(np.mean(end_diffs))
    else:
        both_25 = np.mean(start_abs <= 25) * 100.0
        both_50 = np.mean(start_abs <= 50) * 100.0
        both_100 = np.mean(start_abs <= 100) * 100.0
        pause_bleed = 0.0
        end_med = 0.0
        end_mean = 0.0
        end_p90 = 0.0
        end_bias = 0.0

    timing = TimingMetrics(
        matched_words=len(matches),
        start_med_ms=round(float(np.median(start_abs)), 1),
        start_mean_ms=round(float(np.mean(start_abs)), 1),
        start_p90_ms=round(float(np.percentile(start_abs, 90)), 1),
        end_med_ms=round(end_med, 1),
        end_mean_ms=round(end_mean, 1),
        end_p90_ms=round(end_p90, 1),
        within_25ms_pct=round(float(both_25), 1),
        within_50ms_pct=round(float(both_50), 1),
        within_100ms_pct=round(float(both_100), 1),
        start_bias_ms=round(float(np.mean(start_diffs)), 1),
        end_bias_ms=round(end_bias, 1),
        pause_bleed_pct=pause_bleed,
    )

    return ScoreResult(
        system_id=system_id,
        clip_id=clip_id,
        ref_words_count=len(clean_refs),
        hyp_words_count=len(clean_hyps),
        wer=round(wer * 100, 2),
        cer=round(cer * 100, 2),
        wer_no_diacritics=round(wer_no_dia * 100, 2),
        timing=timing,
    )


@dataclass
class ProfanityMetrics:
    total_gt: int
    caught: int
    missed: int
    false_alarms: int
    recall_pct: float
    precision_pct: float
    f1_pct: float


def compute_profanity_metrics_for_system(
    manifest_clips: List[dict],
    bench_dir: Any,
    runner: Any,
    system_id: str,
    detector: Optional[Any] = None,
) -> ProfanityMetrics:
    from pathlib import Path
    from zalia_ruta.profanity import ProfanityDetector

    if detector is None:
        detector = ProfanityDetector()

    bench_path = Path(bench_dir)
    refs_dir = bench_path / "references"

    total_gt = 0
    tp = 0
    fn = 0
    fp = 0

    for c in manifest_clips:
        cid = c["id"]
        ref_file = refs_dir / f"{cid}.srt"
        if not ref_file.exists():
            continue

        cues = parse_srt(ref_file.read_text(encoding="utf-8"))
        eval_cues, _ = filter_cues_and_regions(cues)
        gt_profanities = [
            cue for cue in eval_cues
            if detector.is_profanity(cue.text.strip())
        ]
        total_gt += len(gt_profanities)

        hyp_words = runner.load_cached(system_id, cid) or []
        hyp_profanities = [
            w for w in hyp_words
            if detector.is_profanity(w.word)
        ]

        for gt in gt_profanities:
            if any(hw.start < gt.end + 0.35 and hw.end > gt.start - 0.35 for hw in hyp_profanities):
                tp += 1
            else:
                fn += 1

        for hw in hyp_profanities:
            if not any(hw.start < gt.end + 0.35 and hw.end > gt.start - 0.35 for gt in gt_profanities):
                fp += 1

    recall = (tp / total_gt) * 100.0 if total_gt > 0 else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0

    return ProfanityMetrics(
        total_gt=total_gt,
        caught=tp,
        missed=fn,
        false_alarms=fp,
        recall_pct=round(recall, 2),
        precision_pct=round(precision, 2),
        f1_pct=round(f1, 2),
    )

