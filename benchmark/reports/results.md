# Lithuanian ASR & Word-Boundary Benchmark Results

## Executive Summary

| System | Clips | Macro WER% | Macro CER% | ≤50ms Precision | Pause Bleed% |
| --- | --- | --- | --- | --- | --- |
| paprika_acoustic | 1 | 0.00% | 0.00% | 78.3% | 13.8% |
| whisper_small_lt_acoustic | 1 | 34.78% | 6.98% | 73.3% | 13.8% |
| azuolas_acoustic | 1 | 8.70% | 1.74% | 81.0% | 11.8% |
| turbo_acoustic | 1 | 13.04% | 2.91% | 75.0% | 13.8% |
| paprika_refined | 1 | 0.00% | 0.00% | 60.9% | 10.8% |
| whisper_small_lt_refined | 1 | 34.78% | 6.98% | 46.7% | 10.8% |
| azuolas_refined | 1 | 8.70% | 1.74% | 61.9% | 7.5% |
| turbo_refined | 1 | 13.04% | 2.91% | 55.0% | 10.8% |
| paprika_raw | 1 | 0.00% | 0.00% | 39.1% | 35.1% |
| whisper_small_lt_raw | 1 | 34.78% | 6.98% | 13.3% | 100.0% |
| azuolas_raw | 1 | 8.70% | 1.74% | 9.5% | 100.0% |
| turbo_raw | 1 | 13.04% | 2.91% | 20.0% | 80.7% |

## Detailed Per-Clip Results

| System | Clip | WER% | CER% | Start Med (ms) | End Med (ms) | ≤50ms% | End Bias (ms) | Pause Bleed% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paprika_acoustic | stubbas | 0.0 | 0.0 | 24.0 | 17.0 | 78.3 | +4.5 | 13.8% |
| whisper_small_lt_acoustic | stubbas | 34.8 | 7.0 | 35.0 | 19.0 | 73.3 | +0.3 | 13.8% |
| azuolas_acoustic | stubbas | 8.7 | 1.7 | 24.0 | 19.0 | 81.0 | +5.8 | 11.8% |
| turbo_acoustic | stubbas | 13.0 | 2.9 | 35.5 | 13.5 | 75.0 | +2.4 | 13.8% |
| paprika_refined | stubbas | 0.0 | 0.0 | 33.0 | 0.0 | 60.9 | -18.0 | 10.8% |
| whisper_small_lt_refined | stubbas | 34.8 | 7.0 | 36.0 | 0.0 | 46.7 | -27.3 | 10.8% |
| azuolas_refined | stubbas | 8.7 | 1.7 | 33.0 | 0.0 | 61.9 | -17.5 | 7.5% |
| turbo_refined | stubbas | 13.0 | 2.9 | 36.0 | 0.0 | 55.0 | -20.5 | 10.8% |
| paprika_raw | stubbas | 0.0 | 0.0 | 26.0 | 39.0 | 39.1 | -53.6 | 35.1% |
| whisper_small_lt_raw | stubbas | 34.8 | 7.0 | 109.0 | 22.0 | 13.3 | +18.5 | 100.0% |
| azuolas_raw | stubbas | 8.7 | 1.7 | 133.0 | 68.0 | 9.5 | -66.1 | 100.0% |
| turbo_raw | stubbas | 13.0 | 2.9 | 91.5 | 29.0 | 20.0 | +16.6 | 80.7% |