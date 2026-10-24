# AGENTS.md — AI Agent & LLM Operating Instructions

This document provides direct, deterministic instructions for AI assistants, autonomous coding agents, and LLMs interacting with the `zalia-ruta` repository.

---

## 1. Primary Workflows: Processing a Video or Audio File

### Workflow A: Full Bleep & Censorship Pipeline
When asked to "censor", "bleep", or "process" a video or audio file:

```powershell
uv run zalia-ruta -i "<input_path>"
```

### Workflow B: Transcription-Only (Verbatim SRT Generation)
When asked to "transcribe", "generate subtitles", or "create SRT" without bleeping:

```powershell
uv run zalia-ruta -i "<input_path>" --transcribe-only
```
- Skips profanity bleeping, sine generation, audio modification, and video remuxing.
- Directly exports clean, verbatim word-level `.srt` subtitles without `[BLEEP: word]` tags.

### Critical Rules for Path Handling
1. **Always double-quote the input path (`"<path>"`)**:
   - Windows file paths frequently contain spaces, parentheses, or Lithuanian diacritics (e.g. `"original_videos\Obage velniu (atiduok karve2).mp4"`).
   - In PowerShell, unquoted parentheses trigger syntax errors (`The term 'atiduok' is not recognized...`).
2. **Always execute through `uv run`**:
   - `uv run zalia-ruta ...` ensures the isolated virtualenv with CUDA-enabled PyTorch (`cu124`) and `torchaudio` is used.
   - Never run raw `python` or `pip install` without `uv`.

---

## 2. Default Output Structure (VLC Auto-Pairing)

When `-o` / `--output` is omitted, the CLI automatically generates dedicated output files under `output/<stem>/`:

- **Full Bleep Mode (`zalia-ruta -i ...`):**
  ```text
  output/
  └── <video_stem>/
      ├── <video_stem>_bleeped.mp4   <-- Censored video (lossless video copy, -c:v copy)
      └── <video_stem>_bleeped.srt   <-- Aligned word-level subtitles with [BLEEP: word] annotations
  ```

- **Transcribe-Only Mode (`zalia-ruta -i ... --transcribe-only`):**
  ```text
  output/
  └── <video_stem>/
      └── <video_stem>.srt           <-- Aligned word-level verbatim subtitles (clean, unbleeped)
  ```

> **VLC Auto-Pairing:** Because the `.mp4` and `.srt` share the exact same stem and folder, VLC and other media players automatically load the subtitles when the video is opened. In addition, the subtitles are embedded into the `.mp4` container as a soft-subtitle track during full bleep mode.

### Custom Output Location
To specify a custom output directory or filename:
```powershell
# Bleeped video + SRT:
uv run zalia-ruta -i "<input_path>" -o "<output_dir_or_file>"

# Verbatim SRT only:
uv run zalia-ruta -i "<input_path>" --transcribe-only -o "<output_file.srt>"
```

---

## 3. How the Pipeline Works

1. **Audio Extraction:** Lossless master audio + 16 kHz mono WAV via FFmpeg.
2. **Pause-Aligned Chunking:** Slices audio along acoustic energy dips into 20s–29s chunks.
3. **ASR & Precision Alignment:**
   - Transcribes Lithuanian speech using `kristijonas/paprika-whisper-lt-v3` on CUDA.
   - Applies **MMS Forced Alignment** (`torchaudio.pipelines.MMS_FA`) to eliminate pause bleed (down to 0%–2%) and align word boundaries within 50 ms of speech onsets/offsets.
4. **Profanity Detection:** Scans for inflected Lithuanian/Slavic roots, slang, and compound words.
5. **Click-Free Bleeping:** Applies a 1000 Hz sine wave tone with 5 ms crossfading over detected profanity intervals.
6. **Subtitles & Remux:** Saves word-level `.srt` and remuxes video stream losslessly (`-c:v copy`).

---

## 4. Benchmark Operations (`zalia-bench`)

If asked to test, validate, or score the ASR benchmark:

```powershell
# 1. Validate reference SRT formatting
uv run zalia-bench validate

# 2. Run the production system (paprika) on clips
uv run zalia-bench run

# 3. Run an alternative model (e.g. azuolas, whisper_small_lt, turbo)
uv run zalia-bench run --model azuolas
uv run zalia-bench run --model whisper_small_lt

# 4. Score against human ground-truth references (outputs Executive Summary + per-clip breakdown)
uv run zalia-bench score

# Optional: score specific clip, system, or model
uv run zalia-bench score --clip <clip_id>
uv run zalia-bench score --model azuolas
uv run zalia-bench score --all-systems

# 5. Importing Audited Praat TextGrids
When human adjustments are saved to `praat_samples/<clip_id>_adjusted.TextGrid`:
```powershell
uv run python scripts/import_praat_textgrid.py --clip <clip_id> --tier Acoustic_Refined
```

- **Manifest:** [benchmark/manifest.json](benchmark/manifest.json) lists all active test clips.
- **Audio Dataset:** `benchmark/audio/*.wav` contains the self-contained 16 kHz audio files.
- **References:** [benchmark/references/*.srt](benchmark/references/) stores human-audited ground-truth word cues (1 word per cue).
- **Reports:** Results with executive summary are written to [benchmark/reports/results.md](benchmark/reports/results.md).

---

## 5. Running Tests

```powershell
uv run pytest
```
All unit tests in `tests/` must pass (exit code 0) before submitting or concluding changes.
