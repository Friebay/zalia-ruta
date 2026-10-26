# Next Phase Plan: Leaderboard, Hugging Face Release, & Speed Optimization

This document outlines the exact execution plan for the next phase of `zalia-ruta`, transitioning from benchmark dataset construction (47 audited clips) to multi-model evaluation, Hugging Face publication, and local inference acceleration.

---

## 1. Current Benchmark Status (Baseline)

- **Total Active Clips:** **47 clips** registered in [`benchmark/manifest.json`](file:///c:/Users/zabit/Documents/GitHub/zalia-ruta/benchmark/manifest.json).
- **Human-Audited Clips:** 20 clips audited in Praat (including extreme streamer slang, English loanwords, and long silence gaps).
- **Current Baseline Numbers (`paprika` on 47 clips):**
  - **`paprika_acoustic`:** Macro WER: **13.43%** | Macro CER: **5.50%** | ≤50ms Precision: **77.7%** | Pause Bleed: **8.9%**
  - **`paprika_refined` (MMS FA only):** Macro WER: **13.43%** | Macro CER: **5.50%** | ≤50ms Precision: **69.9%** | Pause Bleed: **7.5%**

---

## 2. Phase 1: Establish the Multi-Model Leaderboard

Before publishing or optimizing, run all competitor models on the frozen 47-clip dataset.

### Models to Benchmark:
1. `kristijonas/paprika-whisper-lt-v3` (Whisper Large v3 fine-tune — current champion)
2. `akisviete/azuolas-whisper-lt` (Lithuanian ASR model)
3. `DeividasM/whisper-small-lt` (Lightweight Lithuanian Whisper)
4. `openai/whisper-large-v3-turbo` (OpenAI's high-speed pruned large model)

### Execution Commands:
```powershell
# 1. Run inference across all candidate models
uv run zalia-bench run --model azuolas
uv run zalia-bench run --model whisper_small_lt
uv run zalia-bench run --model turbo

# 2. Score all models and generate the comprehensive leaderboard
uv run zalia-bench score --all-systems
```

### Deliverable:
Updated [`benchmark/reports/results.md`](file:///c:/Users/zabit/Documents/GitHub/zalia-ruta/benchmark/reports/results.md) containing the full cross-model comparison across:
- **Macro WER / CER** (transcription accuracy on slang and clean speech).
- **≤50ms Precision** (word boundary alignment accuracy).
- **Pause Bleed %** (hallucination into dead silence gaps).

---

## 3. Phase 2: Package & Publish to Hugging Face

Publish the benchmark as `zalia-ruta-bench-lt` on Hugging Face to make Lithuanian speech & alignment benchmarking publicly reproducible.

### Key Considerations & Safeguards:
- **Copyright Handling:** Because clips are sourced from YouTube, the dataset card must state fair-use research purposes (or provide a Python dataset script that downloads/slices from YouTube on-the-fly, or host audio under a gated Hugging Face repository).
- **Dataset Structure:**
  ```text
  data/
  ├── audio/
  │   └── <clip_id>.wav        (16 kHz mono PCM WAV)
  ├── references/
  │   └── <clip_id>.srt        (Audited word cues, 1 word per cue)
  └── metadata.jsonl           (clip_id, duration, speaker, category, notes)
  ```
- **Hugging Face `datasets` Schema:**
  - `id`: string (clip ID)
  - `audio`: `datasets.Audio(sampling_rate=16000)`
  - `text`: string (verbatim transcript)
  - `words`: list of `{"word": str, "start": float, "end": float}`
  - `category`: string (`profanity`, `silence`, `speech`, `reaction`)

### Execution:
Create [`scripts/export_hf_dataset.py`](file:///c:/Users/zabit/Documents/GitHub/zalia-ruta/scripts/) to bundle the manifest and push to the Hugging Face Hub using `huggingface_hub`.

---

## 4. Phase 3: Speed Optimization on Local PC (`faster-whisper` / CTranslate2)

Right now, `zalia-ruta` uses standard Hugging Face PyTorch inference (`transformers.pipeline`), which has significant VRAM and latency overhead.

### Optimization Steps:
1. **Migrate Engine to `faster-whisper` (CTranslate2):**
   - 3x–4x faster inference on NVIDIA GPUs.
   - Converts `kristijonas/paprika-whisper-lt-v3` weights to CTranslate2 format (`ct2-transformers-converter`).
2. **Quantization Benchmark:**
   - Test `float16` vs `int8_float16`.
   - Verify that `int8_float16` does not degrade WER or boundary alignment on the 47 benchmark clips.
3. **Measure Throughput:**
   - Target: >5x real-time speed for 1-minute audio on local GPU.

---

## 5. Phase 4: Accuracy & Code-Switching Improvements

Address the real failure modes uncovered by the benchmark:
1. **English Loanwords & Code-Switching:**
   - Words like *actually*, *fuck this*, *mind fucked*, *I guess* were phonetically mangled by Paprika Whisper (`seks šilį mastą`, `akščia`, `mindfakt`).
   - Add prompt conditioning or a secondary correction pass for common Lithuanian streaming loanwords.
2. **MMS Forced Alignment on Slang:**
   - MMS CTC aligner strips diacritics and assumes standard Lithuanian phonology. Slang and English words often lose character alignment.
   - Refine fallback alignment rules when MMS CTC confidence drops below threshold.

---

## Quick Resume Instructions for Next Chat

To begin immediately in a fresh chat, provide this prompt:
> *"Read `next_phase_plan.md` and start with Phase 1: run all competitor models on the 47-clip benchmark and score the leaderboard."*
