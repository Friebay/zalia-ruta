# 🌿 Žalia-Rūta (zalia-ruta)

Automated video and audio profanity bleeper tailored for the Lithuanian language.

It losslessly preserves video quality (`-c:v copy`), runs local Lithuanian ASR (`kristijonas/paprika-whisper-lt-v3`) on NVIDIA CUDA, detects inflected Lithuanian/Slavic profanities, replaces them with a 1000 Hz sine tone with smooth 5 ms crossfading, and embeds synchronized word-level `.srt` subtitles.

---

## Key Features

1. **0% Video Quality Loss**: Uses FFmpeg stream copying (`-c:v copy`) to leave video streams completely untouched and losslessly preserved.
2. **Specialized Lithuanian ASR**: Powered by `kristijonas/paprika-whisper-lt-v3` running on local GPU (`CUDA float16`).
3. **Smart Pause-Aligned Chunking**: Slices long audio along 100 ms RMS acoustic energy dips (20.0s – 29.0s window), preventing words from being cut in half.
4. **Inflection & Slang Engine**:
   - Diacritic-insensitive matching (`NFKD` normalization) to catch `blėt`, `blet`, `bleit`, `bliat`, `buliot`, `šūdas`, etc.
   - Compound word detection (e.g. `sukanachų`).
   - Core Lithuanian/Slavic stems: `blet*`, `nachui*`, `suka*`, `kurva*`, `pis*`, `pyzd*`, `jobt*`, `byb*`, `šikna*`, `debil*`.
5. **Click-Free Bleeping**: Replaces swear words with a 1000 Hz sine tone applying 5 ms linear crossfading at boundaries to avoid audible popping.
6. **Synchronized Subtitles**: Produces word-level `.srt` subtitle tracks with acoustic silence trimming and embedded `[BLEEP: word]` annotations.

---

## Installation

Requires **Python 3.11+**, **FFmpeg**, and an **NVIDIA GPU** (recommended).

```powershell
# Clone the repository
git clone https://github.com/Friebay/zalia-ruta.git
cd zalia-ruta

# Create virtual environment and install dependencies with uv
uv venv
.venv\Scripts\activate
uv pip install -e .
```

---

## Usage

### Basic Command

**In PowerShell / CMD (without activating virtualenv):**
```powershell
.\zalia-ruta.bat -i "original_videos/mano_video.mp4"
```
Or with direct virtualenv path:
```powershell
.venv\Scripts\zalia-ruta -i "original_videos/mano_video.mp4"
```
Or after activating your venv (`.venv\Scripts\activate`):
```powershell
zalia-ruta -i "original_videos/mano_video.mp4"
```

By default, this automatically creates:
```text
output/
└── mano_video/
    ├── mano_video_bleeped.mp4
    └── mano_video_bleeped.srt
```

### Specify Custom Output Path

```powershell
.\zalia-ruta.bat -i "original_videos/mano_video.mp4" -o "output/mano_video/mano_cenzuruotas.mp4"
```

### Full Options

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
  --device [cuda|cpu]             Inference device for Whisper model [default: auto].
  --export-srt / --no-export-srt  Export standalone .srt subtitles file. [default: export-srt]
  --keep-temp / --no-keep-temp    Keep intermediate extracted WAV files. [default: no-keep-temp]
  -h, --help                      Show this message and exit.
```

### Custom Words File

Create a text file (e.g. `my_words.txt`) with additional words or regular expressions:
```text
# Exact or wildcard roots
durnelis
durne*
# Regex pattern
^kvail.*$
```
Run with:
```powershell
zalia-ruta -i "video.mp4" --custom-words "my_words.txt"
```

---

## Running Tests

```powershell
.venv\Scripts\pytest
```
All 14 unit tests cover:
* Audio extraction & stream probing
* RMS energy dip pause search
* ASR word timestamp generation & acoustic silence trimming
* Profanity pattern matching & compound detection
* Sine wave generation, crossfading & drift verification
* Subtitle timestamp formatting
* CLI argument parsing & validation
