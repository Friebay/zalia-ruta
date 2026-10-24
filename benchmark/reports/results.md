# Lithuanian ASR & Word-Boundary Benchmark Results

## Executive Summary

| System | Clips | Macro WER% | Macro CER% | ≤50ms Precision | Pause Bleed% |
| --- | --- | --- | --- | --- | --- |
| paprika_acoustic | 21 | 8.55% | 2.63% | 80.2% | 6.3% |
| azuolas_acoustic | 21 | 15.87% | 10.16% | 78.7% | 8.8% |
| whisper_small_lt_acoustic | 21 | 58.30% | 18.77% | 79.4% | 7.2% |
| paprika_refined | 21 | 8.55% | 2.63% | 69.6% | 3.8% |
| azuolas_refined | 21 | 15.87% | 10.16% | 67.7% | 6.7% |
| whisper_small_lt_refined | 21 | 58.30% | 18.77% | 68.2% | 5.5% |

## Detailed Per-Clip Results

| System | Clip | WER% | CER% | Start Med (ms) | End Med (ms) | ≤50ms% | End Bias (ms) | Pause Bleed% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paprika_acoustic | sinkevicius | 5.7 | 2.7 | 38.0 | 29.0 | 52.4 | +12.5 | 13.7% |
| azuolas_acoustic | sinkevicius | 6.7 | 2.9 | 38.0 | 28.5 | 52.0 | +12.1 | 13.7% |
| whisper_small_lt_acoustic | sinkevicius | 37.1 | 8.7 | 39.0 | 30.0 | 50.7 | +13.3 | 5.8% |
| paprika_refined | sinkevicius | 5.7 | 2.7 | 41.0 | 21.0 | 51.5 | -6.3 | 11.4% |
| azuolas_refined | sinkevicius | 6.7 | 2.9 | 41.0 | 21.0 | 51.0 | -6.3 | 11.4% |
| whisper_small_lt_refined | sinkevicius | 37.1 | 8.7 | 42.0 | 25.0 | 50.7 | -7.9 | 4.5% |
| paprika_acoustic | stubbas | 0.0 | 0.0 | 24.0 | 17.0 | 78.3 | +4.5 | 13.8% |
| azuolas_acoustic | stubbas | 8.7 | 1.7 | 24.0 | 19.0 | 81.0 | +5.8 | 11.8% |
| whisper_small_lt_acoustic | stubbas | 34.8 | 7.0 | 35.0 | 19.0 | 73.3 | +0.3 | 13.8% |
| paprika_refined | stubbas | 0.0 | 0.0 | 33.0 | 0.0 | 60.9 | -18.0 | 10.8% |
| azuolas_refined | stubbas | 8.7 | 1.7 | 33.0 | 0.0 | 61.9 | -17.5 | 7.5% |
| whisper_small_lt_refined | stubbas | 34.8 | 7.0 | 36.0 | 0.0 | 46.7 | -27.3 | 10.8% |
| paprika_acoustic | jav_kariai | 8.0 | 2.0 | 20.0 | 12.0 | 87.0 | +12.5 | 5.2% |
| azuolas_acoustic | jav_kariai | 4.0 | 0.5 | 18.0 | 12.5 | 87.5 | +11.0 | 5.2% |
| whisper_small_lt_acoustic | jav_kariai | 44.0 | 9.5 | 13.5 | 12.0 | 78.6 | +4.4 | 5.2% |
| paprika_refined | jav_kariai | 8.0 | 2.0 | 14.0 | 15.0 | 78.3 | -14.3 | 1.8% |
| azuolas_refined | jav_kariai | 4.0 | 0.5 | 16.5 | 15.5 | 75.0 | -15.9 | 1.8% |
| whisper_small_lt_refined | jav_kariai | 44.0 | 9.5 | 7.0 | 20.0 | 64.3 | -24.9 | 1.8% |
| paprika_acoustic | paramedikas | 0.0 | 0.0 | 14.5 | 10.5 | 73.5 | +5.4 | 4.7% |
| azuolas_acoustic | paramedikas | 11.8 | 4.4 | 12.0 | 10.0 | 70.0 | +3.0 | 4.7% |
| whisper_small_lt_acoustic | paramedikas | 79.4 | 21.3 | 13.0 | 18.0 | 75.0 | -19.9 | 4.7% |
| paprika_refined | paramedikas | 0.0 | 0.0 | 20.5 | 0.0 | 70.6 | -13.4 | 0.0% |
| azuolas_refined | paramedikas | 11.8 | 4.4 | 17.5 | 4.5 | 66.7 | -16.6 | 0.0% |
| whisper_small_lt_refined | paramedikas | 79.4 | 21.3 | 29.5 | 4.5 | 37.5 | -45.2 | 0.0% |
| paprika_acoustic | sinica | 3.9 | 2.2 | 9.0 | 10.0 | 88.0 | +8.9 | 33.0% |
| azuolas_acoustic | sinica | 7.8 | 4.8 | 8.0 | 10.0 | 86.3 | +7.2 | 12.1% |
| whisper_small_lt_acoustic | sinica | 56.9 | 16.2 | 13.0 | 24.0 | 83.3 | +12.2 | 7.2% |
| paprika_refined | sinica | 3.9 | 2.2 | 0.0 | 0.0 | 84.0 | -7.6 | 19.6% |
| azuolas_refined | sinica | 7.8 | 4.8 | 0.0 | 0.0 | 82.4 | -9.0 | 3.8% |
| whisper_small_lt_refined | sinica | 56.9 | 16.2 | 0.0 | 0.0 | 75.0 | -9.8 | 3.8% |
| paprika_acoustic | gecas | 0.0 | 0.0 | 13.0 | 27.0 | 98.7 | +20.8 | 9.9% |
| azuolas_acoustic | gecas | 0.0 | 0.0 | 13.0 | 27.0 | 98.7 | +20.8 | 9.9% |
| whisper_small_lt_acoustic | gecas | 40.8 | 14.5 | 15.0 | 27.0 | 97.8 | +24.0 | 30.9% |
| paprika_refined | gecas | 0.0 | 0.0 | 4.0 | 5.0 | 100.0 | -5.8 | 0.9% |
| azuolas_refined | gecas | 0.0 | 0.0 | 4.0 | 5.0 | 100.0 | -5.8 | 0.9% |
| whisper_small_lt_refined | gecas | 40.8 | 14.5 | 4.0 | 5.0 | 100.0 | -5.1 | 23.3% |
| paprika_acoustic | tubelis_taiklumas | 0.0 | 0.0 | 1.0 | 28.0 | 100.0 | +23.1 | 3.2% |
| azuolas_acoustic | tubelis_taiklumas | 30.0 | 30.4 | 1.0 | 28.0 | 95.0 | +0.3 | 7.5% |
| whisper_small_lt_acoustic | tubelis_taiklumas | 70.0 | 21.6 | 0.5 | 19.5 | 100.0 | +20.6 | 7.8% |
| paprika_refined | tubelis_taiklumas | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | -0.0 | 0.0% |
| azuolas_refined | tubelis_taiklumas | 30.0 | 30.4 | 0.0 | 0.0 | 95.0 | -19.0 | 4.0% |
| whisper_small_lt_refined | tubelis_taiklumas | 70.0 | 21.6 | 0.0 | 0.0 | 100.0 | +0.0 | 5.7% |
| paprika_acoustic | tubelis_komentaras | 26.3 | 10.0 | 20.0 | 24.5 | 61.1 | +41.8 | 4.9% |
| azuolas_acoustic | tubelis_komentaras | 142.1 | 135.4 | 27.0 | 45.0 | 46.7 | +535.9 | 76.0% |
| whisper_small_lt_acoustic | tubelis_komentaras | 79.0 | 38.2 | 26.0 | 11.0 | 75.0 | +19.0 | 1.0% |
| paprika_refined | tubelis_komentaras | 26.3 | 10.0 | 0.0 | 0.0 | 61.1 | +30.1 | 2.5% |
| azuolas_refined | tubelis_komentaras | 142.1 | 135.4 | 29.0 | 0.0 | 46.7 | +532.4 | 79.0% |
| whisper_small_lt_refined | tubelis_komentaras | 79.0 | 38.2 | 0.0 | 0.0 | 75.0 | +0.0 | 0.0% |
| paprika_acoustic | kaimo_reikalai | 2.6 | 0.4 | 18.5 | 20.5 | 84.2 | +3.2 | 5.7% |
| azuolas_acoustic | kaimo_reikalai | 0.0 | 0.0 | 18.0 | 20.0 | 84.6 | +2.9 | 5.7% |
| whisper_small_lt_acoustic | kaimo_reikalai | 43.6 | 11.1 | 14.0 | 19.0 | 86.4 | -1.3 | 5.7% |
| paprika_refined | kaimo_reikalai | 2.6 | 0.4 | 0.0 | 1.5 | 68.4 | -22.9 | 1.7% |
| azuolas_refined | kaimo_reikalai | 0.0 | 0.0 | 0.0 | 1.0 | 69.2 | -22.3 | 1.7% |
| whisper_small_lt_refined | kaimo_reikalai | 43.6 | 11.1 | 0.0 | 0.0 | 72.7 | -27.7 | 1.7% |
| paprika_acoustic | karalius_short_01 | 12.7 | 4.8 | 14.0 | 15.5 | 62.9 | +5.5 | 14.7% |
| azuolas_acoustic | karalius_short_01 | 11.4 | 6.0 | 14.0 | 14.5 | 62.9 | +7.4 | 14.7% |
| whisper_small_lt_acoustic | karalius_short_01 | 64.6 | 27.4 | 14.0 | 20.5 | 64.3 | +4.6 | 32.1% |
| paprika_refined | karalius_short_01 | 12.7 | 4.8 | 16.0 | 20.5 | 60.0 | -9.7 | 13.2% |
| azuolas_refined | karalius_short_01 | 11.4 | 6.0 | 15.5 | 18.5 | 62.9 | -7.4 | 13.2% |
| whisper_small_lt_refined | karalius_short_01 | 64.6 | 27.4 | 7.0 | 23.5 | 57.1 | -7.0 | 31.3% |
| paprika_acoustic | karalius_short_04 | 6.4 | 0.7 | 14.0 | 16.0 | 82.2 | -15.7 | 1.5% |
| azuolas_acoustic | karalius_short_04 | 14.9 | 3.1 | 15.0 | 16.0 | 81.0 | -16.2 | 1.3% |
| whisper_small_lt_acoustic | karalius_short_04 | 57.5 | 13.4 | 17.5 | 16.0 | 75.0 | -17.3 | 18.5% |
| paprika_refined | karalius_short_04 | 6.4 | 0.7 | 13.0 | 20.0 | 71.1 | -31.3 | 0.8% |
| azuolas_refined | karalius_short_04 | 14.9 | 3.1 | 13.0 | 23.5 | 69.0 | -33.1 | 0.8% |
| whisper_small_lt_refined | karalius_short_04 | 57.5 | 13.4 | 20.0 | 9.5 | 65.0 | -36.7 | 17.9% |
| paprika_acoustic | karalius_10s_01 | 3.0 | 1.1 | 13.0 | 6.5 | 93.8 | -2.6 | 0.0% |
| azuolas_acoustic | karalius_10s_01 | 9.1 | 2.6 | 14.0 | 7.0 | 93.5 | -1.7 | 0.0% |
| whisper_small_lt_acoustic | karalius_10s_01 | 72.7 | 26.7 | 14.0 | 7.0 | 100.0 | +4.6 | 0.0% |
| paprika_refined | karalius_10s_01 | 3.0 | 1.1 | 13.0 | 10.0 | 90.6 | -11.0 | 0.0% |
| azuolas_refined | karalius_10s_01 | 9.1 | 2.6 | 14.0 | 10.0 | 90.3 | -10.7 | 0.0% |
| whisper_small_lt_refined | karalius_10s_01 | 72.7 | 26.7 | 12.0 | 10.0 | 100.0 | -3.7 | 0.0% |
| paprika_acoustic | karalius_10s_02 | 18.2 | 7.1 | 8.0 | 16.5 | 89.3 | -15.6 | 8.2% |
| azuolas_acoustic | karalius_10s_02 | 9.1 | 3.1 | 10.0 | 16.0 | 87.1 | -15.0 | 8.2% |
| whisper_small_lt_acoustic | karalius_10s_02 | 75.8 | 24.5 | 14.5 | 16.0 | 100.0 | -14.6 | 8.2% |
| paprika_refined | karalius_10s_02 | 18.2 | 7.1 | 9.5 | 26.5 | 67.9 | -27.3 | 4.0% |
| azuolas_refined | karalius_10s_02 | 9.1 | 3.1 | 10.0 | 23.0 | 67.7 | -26.2 | 4.0% |
| whisper_small_lt_refined | karalius_10s_02 | 75.8 | 24.5 | 10.5 | 27.0 | 75.0 | -29.0 | 4.0% |
| paprika_acoustic | karalius_10s_03 | 10.0 | 2.9 | 17.0 | 20.0 | 81.5 | -23.7 | 3.4% |
| azuolas_acoustic | karalius_10s_03 | 6.7 | 1.2 | 16.5 | 19.0 | 82.1 | -23.0 | 3.4% |
| whisper_small_lt_acoustic | karalius_10s_03 | 63.3 | 19.3 | 21.0 | 22.0 | 63.6 | -37.4 | 3.4% |
| paprika_refined | karalius_10s_03 | 10.0 | 2.9 | 15.0 | 29.0 | 66.7 | -39.3 | 3.3% |
| azuolas_refined | karalius_10s_03 | 6.7 | 1.2 | 14.5 | 29.0 | 67.9 | -39.7 | 3.3% |
| whisper_small_lt_refined | karalius_10s_03 | 63.3 | 19.3 | 20.0 | 35.0 | 54.5 | -49.5 | 3.3% |
| paprika_acoustic | karalius_10s_04 | 5.6 | 1.9 | 18.0 | 49.0 | 47.1 | -76.0 | 0.8% |
| azuolas_acoustic | karalius_10s_04 | 5.6 | 1.9 | 18.0 | 49.0 | 47.1 | -76.0 | 0.8% |
| whisper_small_lt_acoustic | karalius_10s_04 | 66.7 | 23.3 | 18.5 | 80.5 | 33.3 | -84.7 | 0.8% |
| paprika_refined | karalius_10s_04 | 5.6 | 1.9 | 31.0 | 61.0 | 41.2 | -98.4 | 0.6% |
| azuolas_refined | karalius_10s_04 | 5.6 | 1.9 | 31.0 | 61.0 | 41.2 | -98.4 | 0.6% |
| whisper_small_lt_refined | karalius_10s_04 | 66.7 | 23.3 | 18.0 | 103.5 | 33.3 | -112.5 | 0.6% |
| paprika_acoustic | karalius_10s_05 | 10.5 | 1.6 | 15.0 | 26.0 | 64.7 | -53.2 | 5.6% |
| azuolas_acoustic | karalius_10s_05 | 0.0 | 0.0 | 15.0 | 26.0 | 63.2 | -42.4 | 5.6% |
| whisper_small_lt_acoustic | karalius_10s_05 | 31.6 | 9.6 | 12.0 | 26.0 | 69.2 | -35.1 | 2.5% |
| paprika_refined | karalius_10s_05 | 10.5 | 1.6 | 17.0 | 35.0 | 52.9 | -65.5 | 5.2% |
| azuolas_refined | karalius_10s_05 | 0.0 | 0.0 | 17.0 | 35.0 | 52.6 | -54.1 | 5.2% |
| whisper_small_lt_refined | karalius_10s_05 | 31.6 | 9.6 | 18.0 | 28.0 | 61.5 | -45.0 | 2.1% |
| paprika_acoustic | karalius_prof_01 | 0.0 | 0.0 | 17.0 | 28.0 | 90.9 | -27.9 | 2.0% |
| azuolas_acoustic | karalius_prof_01 | 9.1 | 2.7 | 18.0 | 27.0 | 90.0 | -26.8 | 2.0% |
| whisper_small_lt_acoustic | karalius_prof_01 | 63.6 | 17.8 | 13.0 | 18.5 | 75.0 | -31.2 | 2.0% |
| paprika_refined | karalius_prof_01 | 0.0 | 0.0 | 19.0 | 39.0 | 54.5 | -43.7 | 2.0% |
| azuolas_refined | karalius_prof_01 | 9.1 | 2.7 | 19.0 | 46.5 | 50.0 | -44.2 | 2.0% |
| whisper_small_lt_refined | karalius_prof_01 | 63.6 | 17.8 | 14.5 | 44.5 | 50.0 | -47.5 | 2.0% |
| paprika_acoustic | orlauskas_prof_02 | 14.3 | 5.0 | 11.0 | 9.0 | 100.0 | -0.5 | 0.0% |
| azuolas_acoustic | orlauskas_prof_02 | 7.1 | 2.5 | 11.0 | 10.0 | 100.0 | -1.2 | 0.0% |
| whisper_small_lt_acoustic | orlauskas_prof_02 | 71.4 | 22.5 | 11.0 | 10.0 | 100.0 | +6.3 | 0.0% |
| paprika_refined | orlauskas_prof_02 | 14.3 | 5.0 | 11.5 | 15.5 | 91.7 | +0.1 | 0.0% |
| azuolas_refined | orlauskas_prof_02 | 7.1 | 2.5 | 13.0 | 17.0 | 92.3 | -1.5 | 0.0% |
| whisper_small_lt_refined | orlauskas_prof_02 | 71.4 | 22.5 | 13.0 | 18.5 | 100.0 | +4.3 | 0.0% |
| paprika_acoustic | orlauskas_prof_03 | 20.0 | 3.5 | 4.0 | 15.0 | 91.7 | -28.4 | 1.7% |
| azuolas_acoustic | orlauskas_prof_03 | 6.7 | 2.3 | 8.0 | 14.0 | 86.7 | -21.3 | 1.7% |
| whisper_small_lt_acoustic | orlauskas_prof_03 | 66.7 | 18.4 | 14.0 | 12.0 | 100.0 | -13.2 | 1.7% |
| paprika_refined | orlauskas_prof_03 | 20.0 | 3.5 | 6.0 | 24.0 | 83.3 | -37.5 | 1.6% |
| azuolas_refined | orlauskas_prof_03 | 6.7 | 2.3 | 8.0 | 21.0 | 80.0 | -29.1 | 1.6% |
| whisper_small_lt_refined | orlauskas_prof_03 | 66.7 | 18.4 | 13.0 | 15.0 | 80.0 | -13.8 | 1.6% |
| paprika_acoustic | orlauskas_prof_04 | 20.0 | 6.7 | 9.0 | 12.0 | 100.0 | -6.9 | 0.0% |
| azuolas_acoustic | orlauskas_prof_04 | 30.0 | 5.0 | 9.0 | 12.0 | 100.0 | -8.3 | 0.0% |
| whisper_small_lt_acoustic | orlauskas_prof_04 | 80.0 | 35.0 | 20.0 | 8.5 | 100.0 | +3.5 | 0.0% |
| paprika_refined | orlauskas_prof_04 | 20.0 | 6.7 | 26.0 | 23.0 | 77.8 | -31.2 | 0.0% |
| azuolas_refined | orlauskas_prof_04 | 30.0 | 5.0 | 24.0 | 23.0 | 71.4 | -33.4 | 0.0% |
| whisper_small_lt_refined | orlauskas_prof_04 | 80.0 | 35.0 | 19.5 | 25.0 | 100.0 | -25.0 | 0.0% |
| paprika_acoustic | orlauskas_prof_05 | 12.5 | 2.7 | 12.0 | 37.0 | 57.1 | -48.4 | 0.0% |
| azuolas_acoustic | orlauskas_prof_05 | 12.5 | 2.7 | 12.0 | 37.0 | 57.1 | -48.4 | 0.0% |
| whisper_small_lt_acoustic | orlauskas_prof_05 | 25.0 | 8.1 | 15.5 | 22.5 | 66.7 | -40.7 | 0.0% |
| paprika_refined | orlauskas_prof_05 | 12.5 | 2.7 | 19.0 | 67.0 | 28.6 | -84.4 | 0.0% |
| azuolas_refined | orlauskas_prof_05 | 12.5 | 2.7 | 19.0 | 67.0 | 28.6 | -84.4 | 0.0% |
| whisper_small_lt_refined | orlauskas_prof_05 | 25.0 | 8.1 | 15.5 | 55.5 | 33.3 | -75.3 | 0.0% |