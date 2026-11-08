# Lithuanian ASR & Word-Boundary Benchmark Results

## Executive Summary

| System | Clips | Macro WER% | Macro CER% | ≤50ms Precision* | Pause Bleed%* | RTF | Speed | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paprika_refined | 133 | 12.54% | 4.85% | 69.9% | 7.5% | 0.269 | 3.7x | 7437 MB |

> \*Timing precision and pause bleed metrics are evaluated exclusively on human-audited clips (`timing_audited: true`). Clips marked `timing_audited: false` are evaluated for WER%/CER% only to prevent circular ground-truth bias.

## Profanity Censorship Leaderboard (Swear Word Catch Rate)

| System | Caught / Total | Missed | False Alarm | Recall% (Catch Rate) | Precision% | F1% |
| --- | --- | --- | --- | --- | --- | --- |
| paprika_refined | 230/255 | 25 | 0 | 90.20% | 100.00% | 94.85% |

> Evaluates how reliably each ASR model catches swear words across human-audited ground-truth profanities. Essential for automated censorship and bleeping pipelines.

## Detailed Per-Clip Results

| System | Clip | WER% | CER% | Start Med (ms) | End Med (ms) | ≤50ms% | End Bias (ms) | Pause Bleed% | RTF | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paprika_refined | sinkevicius | 5.7 | 2.7 | 41.0 | 21.0 | 51.5 | -6.3 | 11.4% | 0.212 | 10.24s |
| paprika_refined | stubbas | 0.0 | 0.0 | 33.0 | 0.0 | 60.9 | -18.0 | 10.8% | 0.315 | 3.57s |
| paprika_refined | jav_kariai | 8.0 | 2.0 | 14.0 | 15.0 | 78.3 | -14.3 | 1.8% | 0.298 | 3.79s |
| paprika_refined | paramedikas | 2.9 | 0.5 | 22.0 | 0.0 | 69.7 | -13.8 | 0.0% | 0.320 | 4.65s |
| paprika_refined | sinica | 3.9 | 2.2 | 0.0 | 0.0 | 84.0 | -7.6 | 19.6% | 0.267 | 4.49s |
| paprika_refined | gecas | 0.0 | 0.0 | 4.0 | 5.0 | 100.0 | -5.8 | 0.9% | 0.235 | 9.19s |
| paprika_refined | tubelis_taiklumas | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | -0.0 | 0.0% | 0.179 | 3.46s |
| paprika_refined | tubelis_komentaras | 26.3 | 10.0 | 0.0 | 0.0 | 61.1 | +30.1 | 2.5% | 0.164 | 3.42s |
| paprika_refined | kaimo_reikalai | 2.6 | 0.4 | 0.0 | 1.5 | 68.4 | -22.9 | 1.7% | 0.225 | 4.85s |
| paprika_refined | karalius_short_01 | 13.9 | 5.0 | 16.0 | 22.0 | 59.4 | -9.9 | 13.2% | 0.253 | 8.04s |
| paprika_refined | karalius_short_04 | 6.4 | 0.7 | 13.0 | 20.0 | 71.1 | -31.3 | 0.8% | 0.240 | 4.89s |
| paprika_refined | karalius_10s_01 | 3.0 | 1.1 | 13.0 | 10.0 | 90.6 | -11.0 | 0.0% | 0.394 | 3.67s |
| paprika_refined | karalius_10s_02 | 18.2 | 7.1 | 9.5 | 26.5 | 67.9 | -27.3 | 4.0% | 0.345 | 3.56s |
| paprika_refined | karalius_10s_03 | 10.0 | 2.9 | 15.0 | 29.0 | 66.7 | -39.3 | 3.3% | 0.357 | 3.65s |
| paprika_refined | karalius_10s_04 | 5.6 | 1.9 | 31.0 | 61.0 | 41.2 | -98.4 | 0.6% | 0.352 | 3.46s |
| paprika_refined | karalius_10s_05 | 10.5 | 1.6 | 17.0 | 35.0 | 52.9 | -65.5 | 5.2% | 0.313 | 3.32s |
| paprika_refined | karalius_prof_01 | 0.0 | 0.0 | 19.0 | 39.0 | 54.5 | -43.7 | 2.0% | 0.490 | 2.95s |
| paprika_refined | orlauskas_prof_02 | 14.3 | 5.0 | 11.5 | 15.5 | 91.7 | +0.1 | 0.0% | 0.970 | 3.09s |
| paprika_refined | orlauskas_prof_03 | 20.0 | 3.5 | 6.0 | 24.0 | 83.3 | -37.5 | 1.6% | 0.675 | 3.38s |
| paprika_refined | orlauskas_prof_04 | 20.0 | 6.7 | 26.0 | 23.0 | 77.8 | -31.2 | 0.0% | 0.789 | 2.89s |
| paprika_refined | orlauskas_prof_05 | 0.0 | 0.0 | 22.5 | 64.5 | 25.0 | -81.6 | 0.0% | 0.965 | 2.82s |
| paprika_refined | karalius_prof_02 | 13.3 | 1.9 | 2.0 | 24.0 | 84.6 | -8.9 | 0.0% | 0.593 | 3.11s |
| paprika_refined | karalius_prof_03 | 6.2 | 1.2 | 10.0 | 8.0 | 100.0 | -12.6 | 1.9% | 0.628 | 3.55s |
| paprika_refined | karalius_prof_04 | 0.0 | 0.0 | 0.0 | 10.0 | 100.0 | -11.0 | 0.0% | 0.668 | 3.01s |
| paprika_refined | karalius_prof_05 | 7.1 | 2.1 | 38.0 | 31.0 | 53.8 | -15.5 | 0.0% | 0.498 | 3.09s |
| paprika_refined | karalius_prof_06 | 0.0 | 0.0 | 0.0 | 7.0 | 100.0 | -7.9 | 0.0% | 0.408 | 3.23s |
| paprika_refined | karalius_silence_01 | 0.0 | 0.0 | 0.0 | 20.0 | 100.0 | +6.9 | 0.0% | 0.481 | 3.39s |
| paprika_refined | karalius_silence_02 | 20.0 | 12.9 | 10.0 | 24.0 | 80.0 | -80.8 | 0.2% | 0.377 | 2.77s |
| paprika_refined | karalius_silence_03 | 0.0 | 0.0 | 2397.0 | 204.0 | 25.0 | -1366.8 | 100.0% | 0.387 | 2.67s |
| paprika_refined | karalius_silence_04 | 0.0 | 0.0 | 35.0 | 40.0 | 66.7 | -51.3 | 0.0% | 0.400 | 2.72s |
| paprika_refined | karalius_silence_05 | 21.4 | 5.6 | 13.0 | 22.0 | 90.9 | -22.2 | 92.6% | 0.467 | 3.62s |
| paprika_refined | vaziuok_nachuj | 104.8 | 48.1 | 79.0 | 242.0 | 50.0 | +1985.8 | 43.0% | 0.222 | 4.10s |
| paprika_refined | karalius_prof_07 | 42.1 | 15.9 | 17.0 | 15.0 | 92.3 | -18.5 | 23.2% | 0.448 | 3.14s |
| paprika_refined | karalius_prof_08 | 30.4 | 13.2 | 14.5 | 20.5 | 93.8 | -17.0 | 0.0% | 0.473 | 3.34s |
| paprika_refined | karalius_prof_09 | 0.0 | 0.0 | 0.0 | 20.5 | 100.0 | -17.1 | 0.0% | 0.599 | 3.84s |
| paprika_refined | karalius_prof_10 | 0.0 | 0.0 | 24.5 | 29.0 | 60.0 | -49.1 | 0.6% | 0.484 | 2.89s |
| paprika_refined | karalius_prof_11 | 0.0 | 0.0 | 0.0 | 10.0 | 100.0 | -9.4 | 0.0% | 0.464 | 3.14s |
| paprika_refined | karalius_silence_06 | 0.0 | 0.0 | 0.0 | 102.0 | 0.0 | +102.0 | 0.0% | 0.346 | 2.60s |
| paprika_refined | karalius_silence_07 | 16.7 | 2.8 | 18.0 | 10.0 | 60.0 | -27.0 | 0.0% | 0.353 | 3.11s |
| paprika_refined | karalius_silence_08 | 0.0 | 0.0 | 155.0 | 3589.0 | 0.0 | +3589.0 | 0.0% | 0.547 | 2.79s |
| paprika_refined | karalius_silence_09 | 28.6 | 5.6 | 46.0 | 84.0 | 40.0 | -140.6 | 0.0% | 0.496 | 2.73s |
| paprika_refined | karalius_silence_10 | 40.0 | 20.8 | 11.0 | 19.0 | 66.7 | -31.0 | 0.0% | 0.491 | 2.70s |
| paprika_refined | karalius_broken_01 | 22.2 | 20.4 | 20.5 | 18.0 | 85.7 | -19.4 | 0.0% | 0.581 | 3.27s |
| paprika_refined | karalius_broken_02 | 40.0 | 13.6 | 10.0 | 15.0 | 71.4 | -59.9 | 7.5% | 0.694 | 3.24s |
| paprika_refined | karalius_broken_05 | 28.6 | 4.9 | 8.0 | 28.0 | 60.0 | -39.4 | 1.3% | 0.583 | 3.07s |
| paprika_refined | karalius_broken_07 | 13.3 | 4.0 | 9.0 | 31.0 | 76.9 | -50.5 | 0.0% | 0.623 | 2.94s |
| paprika_refined | karalius_broken_08 | 30.0 | 32.1 | 25.0 | 22.0 | 71.4 | -19.0 | 3.6% | 0.540 | 2.90s |
| paprika_refined | karalius_prof_12 | 20.9 | 7.6 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.245 | 5.54s |
| paprika_refined | karalius_prof_13 | 4.3 | 1.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.215 | 5.10s |
| paprika_refined | karalius_prof_14 | 3.9 | 0.6 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.265 | 4.94s |
| paprika_refined | karalius_prof_15 | 11.8 | 7.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.221 | 4.42s |
| paprika_refined | karalius_prof_16 | 11.1 | 5.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.255 | 4.80s |
| paprika_refined | karalius_prof_17 | 4.3 | 1.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.234 | 4.84s |
| paprika_refined | karalius_prof_18 | 9.5 | 1.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.211 | 4.34s |
| paprika_refined | karalius_prof_19 | 9.1 | 3.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.199 | 4.25s |
| paprika_refined | karalius_prof_20 | 3.0 | 1.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.237 | 5.67s |
| paprika_refined | karalius_prof_21 | 5.7 | 1.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.229 | 4.76s |
| paprika_refined | karalius_prof_45 | 8.0 | 4.6 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.242 | 4.95s |
| paprika_refined | karalius_prof_46 | 5.0 | 1.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.203 | 4.33s |
| paprika_refined | karalius_prof_47 | 4.2 | 0.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.207 | 4.79s |
| paprika_refined | karalius_prof_48 | 4.8 | 1.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.248 | 5.11s |
| paprika_refined | karalius_prof_49 | 15.0 | 4.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.215 | 4.88s |
| paprika_refined | karalius_prof_50 | 20.4 | 7.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.216 | 4.15s |
| paprika_refined | karalius_prof_51 | 7.8 | 2.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.258 | 5.31s |
| paprika_refined | karalius_prof_52 | 11.6 | 3.0 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.217 | 4.41s |
| paprika_refined | karalius_prof_53 | 6.2 | 1.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.220 | 5.00s |
| paprika_refined | karalius_prof_54 | 20.0 | 10.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.248 | 5.39s |
| paprika_refined | karalius_prof_55 | 15.8 | 4.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.198 | 4.22s |
| paprika_refined | karalius_prof_56 | 9.7 | 5.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.261 | 5.38s |
| paprika_refined | karalius_prof_58 | 11.7 | 5.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.239 | 4.97s |
| paprika_refined | karalius_prof_59 | 8.8 | 1.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.219 | 4.65s |
| paprika_refined | karalius_prof_60 | 6.8 | 1.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.219 | 5.25s |
| paprika_refined | karalius_prof_61 | 27.0 | 10.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.210 | 4.11s |
| paprika_refined | karalius_prof_62 | 7.5 | 2.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.219 | 4.27s |
| paprika_refined | karalius_prof_63 | 13.5 | 4.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.248 | 4.99s |
| paprika_refined | karalius_prof_64 | 3.7 | 1.6 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.237 | 4.58s |
| paprika_refined | karalius_prof_65 | 5.0 | 4.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.348 | 3.65s |
| paprika_refined | karalius_prof_66 | 15.5 | 8.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.251 | 8.71s |
| paprika_refined | karalius_prof_67 | 14.0 | 3.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.245 | 4.91s |
| paprika_refined | karalius_prof_68 | 0.0 | 0.0 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.254 | 3.04s |
| paprika_refined | karalius_prof_69 | 34.0 | 6.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.231 | 4.64s |
| paprika_refined | karalius_prof_70 | 13.0 | 4.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.225 | 4.37s |
| paprika_refined | karalius_prof_71 | 10.0 | 2.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.223 | 4.96s |
| paprika_refined | karalius_prof_72 | 0.0 | 0.0 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.157 | 3.60s |
| paprika_refined | karalius_prof_74 | 12.1 | 4.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.178 | 4.01s |
| paprika_refined | karalius_prof_75 | 6.7 | 3.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.232 | 4.82s |
| paprika_refined | karalius_prof_76 | 7.8 | 2.0 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.213 | 4.93s |
| paprika_refined | karalius_prof_78 | 15.9 | 6.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.286 | 5.66s |
| paprika_refined | karalius_prof_80 | 1.7 | 0.6 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.218 | 4.67s |
| paprika_refined | karalius_prof_81 | 21.7 | 6.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.212 | 4.49s |
| paprika_refined | karalius_prof_82 | 16.7 | 6.1 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.247 | 5.04s |
| paprika_refined | karalius_prof_83 | 10.7 | 2.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.222 | 4.71s |
| paprika_refined | karalius_prof_84 | 11.4 | 3.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.223 | 4.21s |
| paprika_refined | karalius_prof_85 | 11.4 | 3.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.213 | 4.86s |
| paprika_refined | karalius_prof_86 | 7.7 | 2.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.221 | 4.51s |
| paprika_refined | karalius_prof_87 | 7.1 | 2.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.246 | 4.51s |
| paprika_refined | karalius_prof_88 | 10.9 | 2.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.229 | 4.61s |
| paprika_refined | karalius_prof_89 | 16.7 | 9.1 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.211 | 4.01s |
| paprika_refined | karalius_prof_90 | 12.1 | 4.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.197 | 4.33s |
| paprika_refined | karalius_prof_91 | 8.7 | 3.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.217 | 4.81s |
| paprika_refined | karalius_prof_92 | 9.6 | 3.2 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.220 | 4.36s |
| paprika_refined | karalius_prof_93 | 11.7 | 2.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.251 | 5.48s |
| paprika_refined | karalius_prof_94 | 9.4 | 5.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.186 | 3.56s |
| paprika_refined | karalius_prof_95 | 18.8 | 5.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.178 | 3.83s |
| paprika_refined | karalius_prof_96 | 23.7 | 5.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.222 | 4.74s |
| paprika_refined | karalius_prof_97 | 4.5 | 1.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.200 | 4.24s |
| paprika_refined | karalius_prof_98 | 0.0 | 0.0 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.214 | 4.55s |
| paprika_refined | karalius_prof_99 | 2.4 | 0.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.227 | 5.06s |
| paprika_refined | karalius_prof_100 | 9.8 | 2.8 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.220 | 4.32s |
| paprika_refined | karalius_prof_101 | 15.4 | 5.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.231 | 4.47s |
| paprika_refined | karalius_prof_102 | 13.7 | 5.1 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.269 | 5.54s |
| paprika_refined | karalius_prof_103 | 1.7 | 0.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.232 | 4.47s |
| paprika_refined | karalius_prof_104 | 22.2 | 8.1 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.197 | 4.13s |
| paprika_refined | karalius_prof_105 | 8.0 | 3.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.221 | 4.93s |
| paprika_refined | karalius_prof_106 | 19.4 | 4.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.197 | 4.20s |
| paprika_refined | karalius_prof_107 | 14.9 | 7.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.265 | 4.81s |
| paprika_refined | karalius_prof_108 | 19.5 | 5.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.209 | 4.49s |
| paprika_refined | karalius_prof_109 | 18.9 | 7.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.203 | 4.51s |
| paprika_refined | karalius_prof_110 | 8.3 | 3.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.246 | 5.83s |
| paprika_refined | karalius_prof_111 | 4.7 | 1.9 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.213 | 4.21s |
| paprika_refined | karalius_prof_112 | 3.6 | 1.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.219 | 4.86s |
| paprika_refined | karalius_prof_113 | 10.9 | 2.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.233 | 4.75s |
| paprika_refined | karalius_prof_114 | 12.8 | 5.1 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.212 | 4.31s |
| paprika_refined | karalius_prof_115 | 6.9 | 2.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.227 | 4.53s |
| paprika_refined | karalius_prof_116 | 40.0 | 17.0 | N/A* | N/A* | N/A* | N/A* | N/A* | 1.845 | 34.95s |
| paprika_refined | karalius_prof_117 | 27.3 | 2.7 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.204 | 4.03s |
| paprika_refined | karalius_prof_118 | 8.2 | 4.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.236 | 4.82s |
| paprika_refined | karalius_prof_119 | 24.2 | 11.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.624 | 12.70s |
| paprika_refined | karalius_prof_120 | 16.3 | 4.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.218 | 4.18s |
| paprika_refined | karalius_prof_121 | 7.9 | 2.5 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.245 | 4.97s |
| paprika_refined | karalius_prof_122 | 5.8 | 1.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.233 | 4.57s |
| paprika_refined | karalius_prof_123 | 20.0 | 10.3 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.210 | 4.44s |
| paprika_refined | karalius_prof_124 | 48.6 | 36.4 | N/A* | N/A* | N/A* | N/A* | N/A* | 0.254 | 5.66s |