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

The repository includes an empirical speech benchmark evaluating Lithuanian ASR accuracy and acoustic word-boundary precision against **133 human-audited ground truth references** (5,146 reference words) across news broadcasts, political debates, colloquial streamer reactions, and heavy profanity.

The dataset is publicly hosted on Hugging Face: [**Frararo/zalia-ruta-bench-lt**](https://huggingface.co/datasets/Frararo/zalia-ruta-bench-lt).

### Official Leaderboard (133 Clips)

| Rank | Model | Model Type | Micro WER% (Corpus) | Macro WER% | Micro CER% | WER (No Diacritics) | RTF | Speed |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | Whisper Large v3 (CT2 INT8) | **11.08%** | **12.86%** | **4.28%** | **10.30%** | **0.28** | **3.6x** |
| 🥈 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | Whisper Large v3 (CT2 INT8) | **11.60%** | **13.81%** | **4.96%** | **10.88%** | **1.30** | **0.8x** |
| 🥉 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | Whisper Large v3 Turbo (OpenAI) | **29.93%** | **33.82%** | **14.34%** | **27.40%** | **1.89** | **0.5x** |
| 4 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | Whisper Small (DeividasM) | **74.66%** | **104.44%** | **36.51%** | **71.73%** | **0.72** | **1.4x** |

* **Real-Time Factor (RTF):** Ratio of total transcription latency to total audio duration (`latency / duration`). Lower is faster; RTF < 1.0 means faster than real-time.
* **Speed:** Speedup factor (`duration / latency` = `1 / RTF`). Evaluated on NVIDIA GTX 1650 (4 GB VRAM).

### Profanity Censorship Leaderboard (Swear Word Catch Rate)

Evaluates how reliably each model catches Lithuanian and Slavic swear words across **262 human-audited ground-truth profanities** across the benchmark dataset. Critical for automated censorship and bleeping pipelines:

| Rank | Model | Caught / Total | Missed | False Alarms | Recall% (Catch Rate) | Precision% | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | **240 / 262** | 22 | 0 | **91.60%** | **100.00%** | **95.62%** |
| 🥈 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | **236 / 262** | 26 | 1 | **90.08%** | **99.58%** | **94.59%** |
| 🥉 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | **214 / 262** | 48 | 0 | **81.68%** | **100.00%** | **89.92%** |
| 4 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | **166 / 262** | 96 | 11 | **63.36%** | **93.79%** | **75.63%** |

* **Recall% (Catch Rate):** $\frac{\text{Profanities Caught}}{\text{Total Ground-Truth Profanities}} \times 100\%$. The primary metric for censorship safety.
* **Precision%:** $\frac{\text{Profanities Caught}}{\text{Profanities Caught + False Alarms}} \times 100\%$. Measures false-positive bleeping rate.

*Detailed per-clip breakdown is saved at [benchmark/reports/results.md](benchmark/reports/results.md).*

---

### Running the Benchmark Locally

```powershell
# 1. Validate reference SRT formatting
uv run zalia-bench validate

# 2. Run the production model (paprika) with live tqdm progress bar
uv run zalia-bench run

# 3. Run alternative models (e.g. whisper_small_lt, turbo, azuolas)
uv run zalia-bench run --model whisper_small_lt
uv run zalia-bench run --model turbo

# 4. Score systems against human references
uv run zalia-bench score

# Optional: score specific clip, system, or include legacy baselines
uv run zalia-bench score --clip <clip_id>
uv run zalia-bench score --system paprika_ct2_acoustic
uv run zalia-bench score --all-systems
```

---

### Evaluating Third-Party Models

Anyone can benchmark their own ASR model against this dataset using Python:

```python
import re
import jiwer
from datasets import load_dataset

# 1. Load the benchmark dataset
ds = load_dataset("Frararo/zalia-ruta-bench-lt", split="test")

# 2. Official benchmark normalizer
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize(text: str) -> str:
    words = []
    for w in _PUNCT_RE.sub("", text.lower()).split():
        if w.startswith("facebook"): w = "feisbuk" + w[8:]
        elif w in ("youtubes", "youtubės", "jutubes"): w = "jutubės"
        elif w.startswith("youtube") or w.startswith("youtub"): w = "jutub" + w[7:]
        elif w.startswith("discord") or w.startswith("dicord"): w = "diskord" + w[7:]
        elif w.startswith("edge"): w = "edž" + w[4:]
        words.append(w)
    return " ".join(words)

# 3. Transcribe with your model
refs, hyps = [], []
for sample in ds:
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    
    predicted_text = my_model.transcribe(audio_array, sampling_rate) # <-- Put model here
    
    refs.append(normalize(sample["text"]))
    hyps.append(normalize(predicted_text))

# 4. Official Leaderboard Score
micro_wer = jiwer.wer(refs, hyps) * 100
print(f"Official Micro WER: {micro_wer:.2f}%")
```

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
