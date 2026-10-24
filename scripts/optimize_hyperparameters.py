import json
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.acoustic_refiner import refine_word_boundaries
from zalia_ruta.bench.score import score_system
from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.bench.systems import BenchmarkRunner


BENCH_DIR = Path("benchmark")
MANIFEST_PATH = BENCH_DIR / "manifest.json"


@dataclass
class ClipData:
    cid: str
    audio: np.ndarray
    sr: int
    ref_cues: list
    mms_words: list


def load_all_benchmark_data() -> List[ClipData]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    runner = BenchmarkRunner(BENCH_DIR)
    refs_dir = BENCH_DIR / "references"
    audio_dir = BENCH_DIR / "audio"

    data = []
    for c in manifest.get("clips", []):
        cid = c["id"]
        ref_file = refs_dir / f"{cid}.srt"
        wav_file = audio_dir / f"{cid}.wav"

        if not ref_file.exists() or not wav_file.exists():
            continue

        ref_cues = parse_srt(ref_file.read_text(encoding="utf-8"))
        mms_words = runner.load_cached("paprika_refined", cid)
        if not mms_words:
            continue

        audio, sr = sf.read(str(wav_file))
        data.append(ClipData(cid=cid, audio=audio, sr=sr, ref_cues=ref_cues, mms_words=mms_words))

    return data


def evaluate_params(
    dataset: List[ClipData],
    energy_thresh: float,
    max_offset: float,
    max_onset: float,
    min_clearance: float,
) -> Dict[str, float]:
    accuracies = []
    end_biases = []
    pause_bleeds = []
    start_meds = []
    end_meds = []

    for d in dataset:
        refined = refine_word_boundaries(
            d.audio,
            d.mms_words,
            sr=d.sr,
            energy_threshold_ratio=energy_thresh,
            max_offset_extend_s=max_offset,
            max_onset_extend_s=max_onset,
            min_pause_clearance_s=min_clearance,
        )

        res = score_system("opt_trial", d.cid, d.ref_cues, refined, has_end_times=True)
        t = res.timing
        if t:
            accuracies.append(t.within_50ms_pct)
            end_biases.append(t.end_bias_ms)
            pause_bleeds.append(t.pause_bleed_pct)
            start_meds.append(t.start_med_ms)
            end_meds.append(t.end_med_ms)

    avg_acc = float(np.mean(accuracies))
    avg_bias = float(np.mean(end_biases))
    avg_abs_bias = float(np.mean(np.abs(end_biases)))
    avg_bleed = float(np.mean(pause_bleeds))
    avg_start_med = float(np.mean(start_meds))
    avg_end_med = float(np.mean(end_meds))

    # Composite score: reward high accuracy, penalize large bias and pause bleed
    score = avg_acc - (0.35 * avg_abs_bias) - (0.40 * avg_bleed)

    return {
        "score": score,
        "avg_acc": avg_acc,
        "avg_bias": avg_bias,
        "avg_abs_bias": avg_abs_bias,
        "avg_bleed": avg_bleed,
        "avg_start_med": avg_start_med,
        "avg_end_med": avg_end_med,
    }


def main():
    print("Loading all validated benchmark clips into memory...")
    dataset = load_all_benchmark_data()
    print(f"Loaded {len(dataset)} clips with validated ground truth.")

    # 1. Baseline: evaluate paprika_refined (pure MMS)
    mms_accs, mms_biases, mms_bleeds, mms_start_meds, mms_end_meds = [], [], [], [], []
    for d in dataset:
        res = score_system("mms", d.cid, d.ref_cues, d.mms_words, has_end_times=True)
        t = res.timing
        if t:
            mms_accs.append(t.within_50ms_pct)
            mms_biases.append(t.end_bias_ms)
            mms_bleeds.append(t.pause_bleed_pct)
            mms_start_meds.append(t.start_med_ms)
            mms_end_meds.append(t.end_med_ms)

    print("\n--- BASELINE (paprika_refined / pure MMS) ---")
    print(f"  Accuracy (<=50ms): {np.mean(mms_accs):.2f}%")
    print(f"  End Bias:         {np.mean(mms_biases):+.2f} ms")
    print(f"  Start Med:        {np.mean(mms_start_meds):.2f} ms")
    print(f"  End Med:          {np.mean(mms_end_meds):.2f} ms")
    print(f"  Pause Bleed:      {np.mean(mms_bleeds):.2f}%")

    # Current defaults
    curr = evaluate_params(dataset, energy_thresh=0.16, max_offset=0.060, max_onset=0.030, min_clearance=0.030)
    print("\n--- CURRENT (paprika_acoustic defaults) ---")
    print(f"  Accuracy (<=50ms): {curr['avg_acc']:.2f}%")
    print(f"  End Bias:         {curr['avg_bias']:+.2f} ms")
    print(f"  Start Med:        {curr['avg_start_med']:.2f} ms")
    print(f"  End Med:          {curr['avg_end_med']:.2f} ms")
    print(f"  Pause Bleed:      {curr['avg_bleed']:.2f}%")

    # 2. Search Grid
    grid_energy_thresh = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24]
    grid_max_offset = [0.030, 0.045, 0.060, 0.075, 0.090]
    grid_max_onset = [0.015, 0.025, 0.035, 0.045]
    grid_min_clearance = [0.020, 0.025, 0.030]

    combinations = list(product(grid_energy_thresh, grid_max_offset, grid_max_onset, grid_min_clearance))
    print(f"\nRunning Grid Search over {len(combinations)} hyperparameter combinations...")

    t0 = time.time()
    results = []
    for eth, moff, mons, mclr in combinations:
        metrics = evaluate_params(dataset, eth, moff, mons, mclr)
        results.append({
            "energy_thresh": eth,
            "max_offset": moff,
            "max_onset": mons,
            "min_clearance": mclr,
            **metrics,
        })

    elapsed = time.time() - t0
    print(f"Evaluated {len(combinations)} configurations in {elapsed:.2f}s ({len(combinations)/elapsed:.1f} trials/sec).")

    # Filter out configurations where Pause Bleed is unreasonably high (> 8%)
    valid_results = [r for r in results if r["avg_bleed"] <= 8.5]
    valid_results.sort(key=lambda r: r["score"], reverse=True)

    print("\n=== TOP 5 OPTIMAL CONFIGURATIONS ===")
    for rank, r in enumerate(valid_results[:5], 1):
        print(f"\n#{rank} (Composite Score: {r['score']:.2f}):")
        print(f"  Parameters: energy_threshold_ratio={r['energy_thresh']}, max_offset={r['max_offset']*1000:.0f}ms, max_onset={r['max_onset']*1000:.0f}ms, clearance={r['min_clearance']*1000:.0f}ms")
        print(f"  Accuracy (<=50ms): {r['avg_acc']:.2f}% (vs MMS {np.mean(mms_accs):.2f}%, curr {curr['avg_acc']:.2f}%)")
        print(f"  End Bias:         {r['avg_bias']:+.2f} ms (vs MMS {np.mean(mms_biases):+.2f} ms)")
        print(f"  Start Med:        {r['avg_start_med']:.2f} ms | End Med: {r['avg_end_med']:.2f} ms")
        print(f"  Pause Bleed:      {r['avg_bleed']:.2f}%")


if __name__ == "__main__":
    main()
