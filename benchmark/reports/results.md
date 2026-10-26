# Lithuanian ASR & Word-Boundary Benchmark Results

## Executive Summary

| System | Clips | Macro WER% | Macro CER% | ≤50ms Precision | Pause Bleed% | RTF | Speed | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paprika_acoustic | 1 | 0.00% | 0.00% | 78.3% | 13.8% | N/A | N/A | N/A |
| whisper_small_lt_acoustic | 1 | 34.78% | 6.98% | 73.3% | 13.8% | N/A | N/A | N/A |
| azuolas_acoustic | 1 | 8.70% | 1.74% | 81.0% | 11.8% | N/A | N/A | N/A |
| turbo_acoustic | 1 | 13.04% | 2.91% | 75.0% | 13.8% | 1.894 | 0.5x | 5946 MB |
| paprika_refined | 1 | 0.00% | 0.00% | 60.9% | 10.8% | N/A | N/A | N/A |
| whisper_small_lt_refined | 1 | 34.78% | 6.98% | 46.7% | 10.8% | N/A | N/A | N/A |
| azuolas_refined | 1 | 8.70% | 1.74% | 61.9% | 7.5% | N/A | N/A | N/A |
| turbo_refined | 1 | 13.04% | 2.91% | 55.0% | 10.8% | 1.893 | 0.5x | 5946 MB |

## Detailed Per-Clip Results

| System | Clip | WER% | CER% | Start Med (ms) | End Med (ms) | ≤50ms% | End Bias (ms) | Pause Bleed% | RTF | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paprika_acoustic | stubbas | 0.0 | 0.0 | 24.0 | 17.0 | 78.3 | +4.5 | 13.8% | N/A | N/A |
| whisper_small_lt_acoustic | stubbas | 34.8 | 7.0 | 35.0 | 19.0 | 73.3 | +0.3 | 13.8% | N/A | N/A |
| azuolas_acoustic | stubbas | 8.7 | 1.7 | 24.0 | 19.0 | 81.0 | +5.8 | 11.8% | N/A | N/A |
| turbo_acoustic | stubbas | 13.0 | 2.9 | 35.5 | 13.5 | 75.0 | +2.4 | 13.8% | 1.894 | 21.47s |
| paprika_refined | stubbas | 0.0 | 0.0 | 33.0 | 0.0 | 60.9 | -18.0 | 10.8% | N/A | N/A |
| whisper_small_lt_refined | stubbas | 34.8 | 7.0 | 36.0 | 0.0 | 46.7 | -27.3 | 10.8% | N/A | N/A |
| azuolas_refined | stubbas | 8.7 | 1.7 | 33.0 | 0.0 | 61.9 | -17.5 | 7.5% | N/A | N/A |
| turbo_refined | stubbas | 13.0 | 2.9 | 36.0 | 0.0 | 55.0 | -20.5 | 10.8% | 1.893 | 21.45s |