# Next Phase Plan: Accuracy Optimization, Architecture Evaluation, & 100-Clip Dataset

This document details the active roadmap for `zalia-ruta`, transitioning from inference speed optimization to driving down Word Error Rate (WER%), evaluating alternative speech architectures, and expanding the benchmark dataset.

---

## 1. Completed Milestones (Baseline Status)

- **Phase 1: Multi-Model Benchmark Leaderboard (47 Clips):**
  - Evaluated `paprika`, `azuolas`, `whisper_small_lt`, and `turbo` on 47 frozen, human-audited clips.
  - Winner: **`paprika_acoustic`** (Macro WER: **13.43%**, Macro CER: **5.50%**, ≤50ms Precision: **77.7%**).
- **Phase 2: Hugging Face Release:**
  - Published 47 audio clips and verbatim transcripts to Hugging Face Hub: [`Frararo/zalia-ruta-bench-lt`](https://huggingface.co/datasets/Frararo/zalia-ruta-bench-lt).
- **Phase 3: High-Speed CTranslate2 / INT8 Inference:**
  - Converted `paprika-whisper-lt-v3` to CTranslate2 INT8 (`models/paprika-whisper-lt-v3-ct2-int8_float16` / `RobertasTa/paprika-whisper-lt-v3-ct2-int8`).
  - Achieved **5x–8x faster inference** and **~1.4 GB–2.1 GB VRAM** (completely avoiding Windows PCIe RAM paging on 4 GB GPUs).
  - Empirically measured INT8 accuracy trade-off: **16.37% WER** (+2.94% vs. PyTorch FP16 13.43% WER) with **78.7% ≤50ms precision** (+1.0% tighter alignment).
  - Maintained `transformers` (PyTorch FP16) as default for minimum WER, providing `--backend faster-whisper` as an opt-in acceleration flag.

---

## 2. Phase 4: Prompt Conditioning (`initial_prompt`) — Completed & Evaluated

### Implementation
- Added `DEFAULT_INITIAL_PROMPT` containing inflected Lithuanian profanities (*blet, nachui, kurva, pize, pisau, jobtvaimat*), streamer slang (*seni, davai, davadzia, streameris*), and common English loanwords (*actually, fuck this, mind fucked, I guess*).
- Wired `initial_prompt` support into `LithuanianASR` for both `transformers` (pre-tokenized prompt ID tensor) and `faster-whisper` (CTranslate2 `initial_prompt` parameter).
- Added CLI flags `--initial-prompt <str>` and `--no-prompt` to `zalia-ruta`.
- Registered `paprika_ct2_prompt` and `paprika_prompt` in `BenchmarkRunner` (`src/zalia_ruta/bench/systems.py`).

### Empirical Benchmark Findings (47 Clips)
- **Massive Gains on Livestream / Profanity Clips (15 Clips Improved):**
  - `vaziuok_nachuj`: **114.3% $\rightarrow$ 66.7%** (saved **47.6%** WER; eliminates phonetic evasions like `"nakvaizdžiai"` $\rightarrow$ `"nachui"`).
  - `karalius_broken_02`: **40.0% $\rightarrow$ 10.0%** (saved **30.0%** WER).
  - `karalius_broken_08`: **30.0% $\rightarrow$ 10.0%** (saved **20.0%** WER).
  - `karalius_broken_07`: **13.3% $\rightarrow$ 0.0%** (saved **13.3%** WER).
  - `tubelis_komentaras`: **47.4% $\rightarrow$ 36.8%** (saved **10.5%** WER).
  - `tubelis_taiklumas`: **25.0% $\rightarrow$ 15.0%** (saved **10.0%** WER).
- **Degradations on Silent / Hesitant Pauses (14 Clips Degraded):**
  - Unconditional prompt conditioning causes Whisper to hallucinate, drop soft syllables, or mis-classify low-amplitude speech during long pauses (e.g. `karalius_silence_03`: 0.0% $\rightarrow$ 100.0%, `karalius_silence_01`: 12.5% $\rightarrow$ 50.0%).
  - **Macro WER Impact:** **16.37%** (unprompted) vs. **17.61%** (unconditionally prompted) across the 47-clip aggregate.
- **Architectural Conclusion:** Prompt conditioning is highly effective for heavy slang and profanity clips, but applying it unconditionally across dead-air chunks introduces false insertions. The baseline is preserved as unprompted default, with prompted execution available via `--initial-prompt`.

---

## 3. Phase 5: Architecture Evaluation — NVIDIA Parakeet-TDT (`RobertasTa/lt-ausys`)

Recently published [`RobertasTa/lt-ausys`](https://huggingface.co/RobertasTa/lt-ausys) packages Kristijonas Jakubsonas's `kmynas-parakeet-lt-v4` (NVIDIA Parakeet-TDT 0.6B RNN-Transducer fine-tuned on 3,087h LIEPA-3) in ONNX INT8 format.

### Objectives & Status:
1. Add `onnx-asr` to optional benchmark dependencies (`uv add --optional bench onnx-asr`) — **DONE**.
2. Register `lt_ausys` in `BenchmarkRunner` (`src/zalia_ruta/bench/systems.py`) with adapter `LtAusysASR` — **DONE**.
3. Benchmark across the dataset and compare latency and accuracy — **DONE**.

### Empirical Findings:
- **Inference Speed & Latency:**
  - **RTF ~0.19 (5.2x speedup on CPU):** Parakeet-TDT processes exact audio duration without Whisper's 30s padding.
  - On a 9.6s audio clip (`karalius_10s_01`), raw CPU inference took only ~0.35s (1.80s total including MMS forced alignment).
- **Recognition Accuracy (Catastrophic Collapse on Streamer Speech):**
  - **Mean WER across test clips: 91.62%** (CER 63.3%).
  - Example clips:
    - `karalius_10s_01`: **78.79% WER** (vs. Paprika Whisper **3.03% WER**).
    - `gecas`: **96.05% WER**.
    - `jav_kariai`: **96.00% WER**.
    - `kaimo_reikalai`: **89.74% WER**.
    - `stubbas`: **104.35% WER**.
    - `sinkevicius`: **84.76% WER**.
- **Root Cause Analysis:**
  - **Training Domain Mismatch:** `kmynas-parakeet-lt-v4` was trained strictly on LIEPA-3 (studio read speech, formal radio, telephone audio). It has zero exposure to internet streaming, conversational dialogue, colloquial clipping, slang, background music, or code-switching.
  - **Acoustic / Language Model Fragility:** The RNN-Transducer architecture lacks Whisper's massive cross-attention encoder-decoder capacity. Under natural conversational acoustics, the greedy decoder falls into consonant repetition loops (`nešnukstu`, `šaukšti krūstė`, `stovst`, `spėst`).
- **Architectural Conclusion & Decision:**
  - **Parakeet-TDT is rejected for production transcription and censorship.**
  - Whisper's encoder-decoder architecture with Paprika fine-tuning remains far superior in acoustic robustness and contextual language modeling (13.43% WER vs 91.62% WER).
  - The adapter `LtAusysASR` and registration remain in `src/zalia_ruta/bench/systems.py` as an optional benchmark baseline.

---

## 4. Phase 6: Post-ASR Slang & Code-Switching Normalizer

Our error analysis showed Paprika Whisper consistently makes identical phonetic transcription errors on recurrent English loanwords:
- `"seks šilį mastą"` / `"akščia"` $\rightarrow$ `actually`
- `"mindfakt"` $\rightarrow$ `mind fucked`
- `"davadze"` $\rightarrow$ `davadzia`

### Objectives:
1. Implement a lightweight, rule-based / fuzzy replacement pass in `src/zalia_ruta/asr.py` for verified streamer loanword patterns.
2. Evaluate impact on the benchmark to ensure zero false positives on clean Lithuanian vocabulary.

---

## 5. Phase 7: Dataset Expansion to 100 Clips (Verbatim Transcripts Only)

Expand the current 47-clip benchmark to **100 clips** to increase statistical diversity and uncover further edge cases.

### Strategy (Fast-Track Labeling):
- **Skip Word-Level Timestamping in Praat:** Auditing cues in Praat takes ~15–20 minutes per clip. Writing verbatim text takes only ~1–2 minutes per clip.
- **Harvest 53 Additional Clips:**
  - Slices from active YouTube sources (10s–30s duration).
  - Mix of streamer slang, conversational speech, fast dialogue, and background music.
- **Update Manifest & Hugging Face:**
  - Register new clips in `benchmark/manifest.json`.
  - Push updated 100-clip dataset to `Frararo/zalia-ruta-bench-lt` (v2).

---

## Quick Resume Instructions for Next Chat

To begin immediately in a fresh chat, provide this prompt:
> *"Read `next_phase_plan.md` and start with Phase 4: implement `initial_prompt` conditioning in `asr.py` and benchmark on the 47 clips to lower WER."*
