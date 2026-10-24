# 🌿 Žalia-Rūta (zalia-ruta)

Automated video and audio profanity bleeper tailored for the Lithuanian language.

It losslessly preserves video streams (`-c:v copy`), transcribes speech with local Lithuanian Whisper (`kristijonas/paprika-whisper-lt-v3`), refines word boundaries with **MMS Forced Alignment** (`torchaudio.pipelines.MMS_FA`), detects inflected Lithuanian/Slavic profanities, applies click-free crossfaded 1000 Hz bleep tones, and exports synchronized word-level `.srt` subtitles.

---

## Key Features

1. **0% Video Quality Loss**: Uses FFmpeg stream copying (`-c:v copy`) to leave video streams untouched without re-encoding delays or quality degradation.
2. **Specialized Lithuanian ASR**: Powered by `kristijonas/paprika-whisper-lt-v3` on NVIDIA CUDA (`float16`), achieving ~3.5% average WER on Lithuanian speech.
3. **MMS Forced Alignment Precision**: Uses `torchaudio.pipelines.MMS_FA` to eliminate pause bleed (reducing silent pause bleed from 35%–100% down to **0%–2%**), ensuring bleeps cut off cleanly when speech ends.
4. **Smart Pause-Aligned Chunking**: Slices long audio along acoustic energy dips (20.0s – 29.0s window), resetting CTC alignment drift and keeping VRAM under 50 MB for videos of any length.
5. **Inflection & Slang Engine**:
   - Diacritic-insensitive matching (`NFKD` normalization) catching roots like `blėt`, `blet`, `bliat`, `buliot`, `šūdas`, etc.
   - Compound word and suffix detection (e.g. `sukanachų`).
   - Core roots: `blet*`, `nachui*`, `suka*`, `kurva*`, `pis*`, `pyzd*`, `jobt*`, `byb*`, `šikna*`, `debil*`.
6. **Click-Free Bleeping**: Replaces profanities with a 1000 Hz sine tone with 5 ms crossfading at boundaries to avoid audible audio pops.
7. **Synchronized Subtitles**: Emits aligned word-level `.srt` subtitles with `[BLEEP: word]` tags for censored speech.

---

## Installation

### Prerequisites
- **Python 3.11+**
- **FFmpeg** installed and accessible on `PATH`.
- **NVIDIA GPU** with CUDA support (recommended).

### Setup with `uv` (Recommended)

```powershell
# Clone repository
git clone https://github.com/Friebay/zalia-ruta.git
cd zalia-ruta

# Sync environment with locked CUDA PyTorch
uv sync

# Optional: include benchmark dependencies (jiwer, rapidfuzz)
uv sync --extra bench
```

---

## Usage

### Basic CLI Command

```powershell
# Run with uv
uv run zalia-ruta -i "original_videos/mano_video.mp4"

# Or using the Windows batch launcher
.\zalia-ruta.bat -i "original_videos/mano_video.mp4"
```

By default, output is placed in:
```text
output/
└── mano_video/
    ├── mano_video_bleeped.mp4
    └── mano_video_bleeped.srt
```

### Custom Output Destination

```powershell
uv run zalia-ruta -i "original_videos/mano_video.mp4" -o "output/mano_video/mano_cenzuruotas.mp4"
```

### Options

```text
Usage: zalia-ruta [OPTIONS]

Options:
  -i, --input FILE                Path to input video or audio file. [required]
  -o, --output FILE               Path to output file. Defaults to output/<stem>/<stem>_bleeped<ext>.
  --custom-words FILE             Optional text file with additional profanity words or regex patterns.
  --freq FLOAT                    Bleep sine wave tone frequency in Hz. [default: 1000.0]
  --volume FLOAT                  Bleep sine wave tone volume (0.0 to 1.0). [default: 0.35]
  --pad-start FLOAT               Safety padding in seconds before profanity onset. [default: 0.04]
  --pad-end FLOAT                 Safety padding in seconds after profanity offset. [default: 0.04]
  --device [cuda|cpu]             Inference device for Whisper model. [default: auto-detected]
  --mms-align / --no-mms-align    Use MMS forced alignment for boundary precision. [default: mms-align]
  --export-srt / --no-export-srt  Export standalone .srt subtitles file. [default: export-srt]
  --keep-temp / --no-keep-temp    Keep intermediate extracted WAV files. [default: no-keep-temp]
  -h, --help                      Show this message and exit.
```

### Custom Swear Words File

Create a text file (e.g. `custom_words.txt`):
```text
# Exact or wildcard roots
durnelis
durne*
# Regex patterns
^kvail.*$
```
Run with:
```powershell
uv run zalia-ruta -i "video.mp4" --custom-words "custom_words.txt"
```

---

## Lithuanian Word-Level Benchmark (`zalia-bench`)

The repository includes an empirical benchmark evaluating Lithuanian ASR accuracy and acoustic word-boundary precision against **21 human-audited ground truth references** across news broadcasts, reaction streams, colloquial slang, and heavy profanity:

```powershell
# 1. Validate reference SRT formatting
uv run zalia-bench validate

# 2. Run the production model (paprika)
uv run zalia-bench run

# 3. Run alternative models (e.g. azuolas, whisper_small_lt, turbo)
uv run zalia-bench run --model azuolas
uv run zalia-bench run --model whisper_small_lt

# 4. Score systems against human references
uv run zalia-bench score

# Optional: score specific clip, system, or include legacy baselines
uv run zalia-bench score --clip <clip_id>
uv run zalia-bench score --model azuolas
uv run zalia-bench score --all-systems
```

### Model Comparison Executive Summary (21 Clips)

| System | Architecture / Model | Macro WER% | Macro CER% | ≤50ms Precision | Pause Bleed% |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`paprika_acoustic`** (Production) | `whisper-large-v3-turbo` (809M) + MMS + Energy Refiner | **8.55%** | **2.63%** | **80.2%** | **6.3%** |
| `paprika_refined` | `whisper-large-v3-turbo` (809M) + MMS Forced Alignment | 8.55% | 2.63% | 69.6% | 3.8% |
| `azuolas_acoustic` | `whisper-large-v3` (1.55B) + MMS + Energy Refiner | 15.87% | 10.16% | 78.7% | 8.8% |
| `whisper_small_lt_acoustic` | `whisper-small` (244M) + MMS + Energy Refiner | 58.30% | 18.77% | 79.4% | 7.2% |

Full per-clip breakdown is saved at [benchmark/reports/results.md](benchmark/reports/results.md).

---

## Running Tests

```powershell
uv run pytest
```
The test suite covers:
* MMS forced alignment tokenization, diacritic normalization, and fallback guarantees
* Media probing, audio extraction & FFmpeg remuxing
* RMS energy dip pause search & chunking
* Profanity pattern matching & compound detection
* Sine wave generation, linear crossfading & boundary padding
* Subtitle formatting & SRT validation
* Benchmark scoring metrics (WER, CER, pause bleed, and median offset timing)
