# Lithuanian Speech Annotation Guidelines

## 1. Goal & Tool
- The goal is to establish ground truth word-level transcriptions and start/end boundaries for measuring Lithuanian ASR accuracy.
- Tool: **Subtitle Edit** (Nikse.dk).
- Open the draft `.srt` and corresponding 16 kHz `.wav` file (*Video -> Open video file* to load the audio waveform/spectrogram).
- Target waveform zoom: 1-2 seconds per view. Target boundary accuracy: ±20 ms.

## 2. Text Transcription Rules
- **Verbatim text**: Transcribe exactly what is spoken, not "corrected" formal grammar.
- **Slang and dialect**: Spell as pronounced (e.g. `blėt`, `davai`, `bleit`).
- **Numbers**: Write out numbers as spoken words (e.g. `penki šimtai`, not `500`).
- **Acronyms**: Write standard capitalized letters or phonetic if read as a word (e.g. `NATO`, `VSD`).
- **Casing & Punctuation**: Standard capitalization at sentence starts and proper nouns is fine, but scoring normalizes case and strips punctuation.
- **Disagreement markers**: The draft marks differing words across systems as `paprika|turbo` or `w1|w2|w3`. **You must resolve every `|`** to the single correct word (or edit it if both were wrong). The benchmark validator rejects any remaining `|`.

## 3. Boundary Rules (Start & End)
- **Word Start**:
  - The first audible acoustic energy of the word.
  - For voiceless stop consonants (*p, t, k*) and voiced stops (*b, d, g*) following silence or a pause, place the start boundary at the **burst release** (the silent closure phase before the burst is not part of the audible word).
  - For fricatives (*s, š, z, ž, f, v*) and nasals/vowels, place the boundary where spectral energy begins above the noise floor.
- **Word End**:
  - The last audible acoustic energy of the word.
  - Include trailing aspiration, frication (*-s, -š, -t*), and vowel decay before silence.
- **Continuous Speech / Connected Words (No Pause)**:
  - Place the boundary at the clearest acoustic transition point (spectral change or formant discontinuity).
  - **Do NOT invent fake gaps** between connected words.
  - **Do NOT bridge real pauses**: If there is a silence between words, neither word should bleed into it.

## 4. Special Markers
Mark special speech events using plain text tokens safe for SRT:
- `~ee`, `~mm`: Hesitation / filler sounds. Excluded from WER and timing evaluation.
- `pir-`: False starts or truncated words cut off mid-syllable (dash at the end). Excluded from WER.
- `???`: Truly unintelligible speech. Timed, but excluded from WER.
- `#SKIP`: Insert as cue text for sections that must be excluded completely from all metrics (overlapping speech, heavy background music drowning voice, unparseable laughter/coughs).

## 5. Subtitle Edit Warning
- **NEVER run *Tools -> Fix common errors*** in Subtitle Edit. It merges short cues and extends cue durations to meet standard TV reading speed minimums, which destroys word-level timing!
