#!/usr/bin/env python3
"""
Package and publish the 133-clip Lithuanian ASR Benchmark (zalia-ruta-bench-lt)
to Hugging Face Datasets.

Exports:
- 16 kHz mono WAV audio files (embedded into Parquet).
- Verbatim human-audited Lithuanian transcripts.
- Non-commercial research dataset card (CC BY-NC 4.0).
- Pure audio + text schema (no word timestamps, no internal model run data).

Usage:
    # 1. Export locally to inspect:
    uv run python scripts/export_hf_dataset.py

    # 2. Export and push to Hugging Face Hub (public, non-commercial):
    uv run python scripts/export_hf_dataset.py --push --repo-id Frararo/zalia-ruta-bench-lt
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import soundfile as sf

from zalia_ruta.bench.srt_io import parse_srt


BENCH_DIR = Path("benchmark")
DEFAULT_REPO_ID = "Frararo/zalia-ruta-bench-lt"
DEFAULT_EXPORT_DIR = Path("hf_export/zalia-ruta-bench-lt")


DATASET_CARD_TEMPLATE = """---
language:
- lt
license: cc-by-nc-4.0
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
pretty_name: "Žalia Rūta Lithuanian Speech Benchmark"
tags:
- speech
- audio
- lithuanian
- asr
- benchmark
configs:
- config_name: default
  data_files:
  - split: test
    path: "data/test-*"
---

# Žalia Rūta: Lithuanian Speech Benchmark (`zalia-ruta-bench-lt`)

A curated evaluation dataset consisting of **{total_clips} Lithuanian audio clips** and verbatim human-verified transcripts for speech recognition benchmarking.

---

## Dataset Overview

- **Clips:** **{total_clips} unique clips** (16 kHz mono WAV, 16-bit PCM).
- **Transcripts:** Human-audited verbatim Lithuanian orthographic transcripts (~95% estimated accuracy; rapidly transcribed and audited to expand clip coverage, so minor colloquial phrasing, background noise mishearings, or non-standard slang spellings may be present).
- **Coverage & Speech Diversity:**
  - Conversational speech, political debates, and news commentary (Alexander Stubb, Virginijus Sinkevičius, Vytautas Sinica, Ingrida Šimonytė).
  - Streamer commentary, gaming reactions, heavy profanities, and compound slang / loanwords (~120 Karalius clips, including 97 profanity clips).
  - Natural pauses, background audio, overlapping audio artifacts, and acoustic noise challenges.
- **Alignment:** Word-level alignment timestamps will be provided in a future update.

---

## Dataset Schema

Each sample in `test` contains:

```python
{
    "id": "stubbas",
    "audio": {
        "path": "data/audio/stubbas.wav",
        "array": [...],
        "sampling_rate": 16000
    },
    "text": "Stubb apie Suomijos ir NATO saugumo politiką...",
    "duration": 11.33,
    "category": "speech",
    "youtube_id": "AYzmpjtX-90",
    "notes": "Alexander Stubb news short (11.3s)"
}
```

---

## Quick Usage

```python
from datasets import load_dataset

dataset = load_dataset("Frararo/zalia-ruta-bench-lt", split="test")

sample = dataset[0]
print("Clip ID:", sample["id"])
print("Transcript:", sample["text"])
```

---

## Official Benchmark Leaderboard

Evaluated across all **{total_clips} clips** (5,146 reference words), spanning formal broadcast speech, debates, and challenging colloquial streamer reactions with heavy slang, background audio, and profanities.

| Rank | Model | Model Type | Micro WER% (Corpus) | Macro WER% | Micro CER% | WER (No Diacritics) | RTF | Speed |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | Whisper Large v3 (CT2 INT8) | **11.08%** | **12.86%** | **4.28%** | **10.30%** | **0.28** | **3.6x** |
| 🥈 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | Whisper Large v3 (CT2 INT8) | **11.60%** | **13.81%** | **4.96%** | **10.88%** | **1.30** | **0.8x** |
| 🥉 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | Whisper Large v3 Turbo (OpenAI) | **29.93%** | **33.82%** | **14.34%** | **27.40%** | **1.89** | **0.5x** |
| 4 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | Whisper Small (DeividasM) | **74.66%** | **104.44%** | **36.51%** | **71.73%** | **0.72** | **1.4x** |

* **Micro WER% (Corpus WER):** Total word errors (substitutions + deletions + insertions) divided by total reference words across the entire dataset. Primary benchmark metric.
* **Macro WER%:** Arithmetic mean of per-clip WERs.
* **WER (No Diacritics):** Scored after normalizing Lithuanian diacritic characters (`ą, č, ę, ė, į, š, ų, ū, ž` → `a, c, e, e, i, s, u, u, z`).

---

### Profanity Censorship Leaderboard (Swear Word Catch Rate)

Evaluates how reliably each model catches Lithuanian and Slavic swear words across **262 human-audited ground-truth profanities** across the benchmark dataset:

| Rank | Model | Caught / Total | Missed | False Alarms | Recall% (Catch Rate) | Precision% | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | **240 / 262** | 22 | 0 | **91.60%** | **100.00%** | **95.62%** |
| 🥈 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | **236 / 262** | 26 | 1 | **90.08%** | **99.58%** | **94.59%** |
| 🥉 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | **214 / 262** | 48 | 0 | **81.68%** | **100.00%** | **89.92%** |
| 4 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | **166 / 262** | 96 | 11 | **63.36%** | **93.79%** | **75.63%** |

* **Recall% (Catch Rate):** `(Profanities Caught / Total Ground-Truth Profanities) × 100%`. The primary metric for censorship safety.
* **Precision%:** `(Profanities Caught / (Profanities Caught + False Alarms)) × 100%`. Measures false-positive bleeping rate.

---

## How to Evaluate Your Model

You can benchmark any ASR model against this dataset using the following self-contained evaluation script:

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
        if w.startswith("facebook"):
            w = "feisbuk" + w[8:]
        elif w in ("youtubes", "youtubės", "jutubes"):
            w = "jutubės"
        elif w.startswith("youtube") or w.startswith("youtub"):
            w = "jutub" + w[7:]
        elif w.startswith("discord") or w.startswith("dicord"):
            w = "diskord" + w[7:]
        elif w.startswith("edge"):
            w = "edž" + w[4:]
        words.append(w)
    return " ".join(words)

# 3. Transcribe each sample with your model
refs, hyps = [], []
for sample in ds:
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    
    # Replace this with your model's transcription call:
    # predicted_text = my_model.transcribe(audio_array, sampling_rate)
    predicted_text = sample["text"] # placeholder
    
    refs.append(normalize(sample["text"]))
    hyps.append(normalize(predicted_text))

# 4. Compute official benchmark scores
micro_wer = jiwer.wer(refs, hyps) * 100
micro_cer = jiwer.cer(refs, hyps) * 100
print(f"Official Micro WER: {micro_wer:.2f}%")
print(f"Official Micro CER: {micro_cer:.2f}%")
```

### Submitting to the Leaderboard
To submit your model to the leaderboard table:
1. Open a **Discussion** or **Pull Request** on this dataset repo.
2. Provide your model name and Hugging Face link.
3. Provide your predictions in JSON format (`{"clip_id": "predicted text", ...}`) so the score can be reproduced and verified.

---

## Terms of Use & License

- **License:** [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
- **Intended Use:** Non-commercial research, academic evaluation, and private benchmarking. Not licensed for commercial deployment or commercial model training without explicit authorization.
- **Source Project:** Curated as part of the [`zalia-ruta`](https://github.com/Friebay/zalia-ruta) Lithuanian speech processing project.
"""


def load_and_validate_records(include_timestamps: bool = False) -> List[Dict[str, Any]]:
    """Loads manifest and cross-verifies audio files and SRT references."""
    manifest_file = BENCH_DIR / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    clips = manifest_data.get("clips", [])
    records = []

    for c in clips:
        cid = c["id"]
        wav_path = BENCH_DIR / "audio" / f"{cid}.wav"
        srt_path = BENCH_DIR / "references" / f"{cid}.srt"

        if not wav_path.exists():
            raise FileNotFoundError(f"[{cid}] Audio missing: {wav_path}")
        if not srt_path.exists():
            raise FileNotFoundError(f"[{cid}] Reference SRT missing: {srt_path}")

        # Audio duration & properties
        info = sf.info(str(wav_path))
        duration = round(info.duration, 3)

        # Parse cues for verbatim transcript
        cues = parse_srt(srt_path.read_text(encoding="utf-8"))
        words = []
        spoken_tokens = []
        for cue in cues:
            clean_word = cue.text.strip()
            if not clean_word:
                continue
            if include_timestamps:
                words.append({
                    "word": clean_word,
                    "start": round(cue.start, 3),
                    "end": round(cue.end, 3),
                })
            if not clean_word.startswith("#"):
                spoken_tokens.append(clean_word)

        text = " ".join(spoken_tokens).strip()

        rec = {
            "id": cid,
            "wav_path": wav_path,
            "srt_path": srt_path,
            "text": text,
            "duration": duration,
            "category": c.get("category", "speech"),
            "youtube_id": c.get("youtube_id", ""),
            "notes": c.get("notes", ""),
        }
        if include_timestamps:
            rec["words"] = words

        records.append(rec)

    return records


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_EXPORT_DIR,
    help="Target directory for the packaged Hugging Face dataset.",
)
@click.option(
    "--include-timestamps",
    is_flag=True,
    default=False,
    help="Include word-level timestamp array in export (disabled by default).",
)
@click.option(
    "--push",
    is_flag=True,
    default=False,
    help="Push the dataset directly to the Hugging Face Hub.",
)
@click.option(
    "--repo-id",
    default=DEFAULT_REPO_ID,
    help="Hugging Face Hub repository ID (e.g. Frararo/zalia-ruta-bench-lt).",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    help="Create a private/gated repository on Hugging Face.",
)
@click.option(
    "--token",
    default=None,
    help="Hugging Face API token (defaults to HF_TOKEN environment variable).",
)
def export_dataset(
    output_dir: Path,
    include_timestamps: bool,
    push: bool,
    repo_id: str,
    private: bool,
    token: Optional[str],
):
    """Packages the 133-clip benchmark dataset for Hugging Face and optionally pushes to the Hub."""
    click.echo(f"Validating and preparing dataset from {BENCH_DIR}...")
    records = load_and_validate_records(include_timestamps=include_timestamps)
    click.echo(f"Loaded {len(records)} verified clips.")

    # 1. Prepare export directories
    data_dir = output_dir / "data"
    audio_export_dir = data_dir / "audio"
    audio_export_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines = []
    hf_dataset_rows = []

    click.echo(f"Copying audio and generating metadata in {output_dir}...")
    for r in records:
        cid = r["id"]
        dest_wav = audio_export_dir / f"{cid}.wav"
        shutil.copy2(r["wav_path"], dest_wav)

        meta_entry = {
            "file_name": f"audio/{cid}.wav",
            "id": cid,
            "text": r["text"],
            "duration": r["duration"],
            "category": r["category"],
            "youtube_id": r["youtube_id"],
            "notes": r["notes"],
        }
        if include_timestamps:
            meta_entry["words"] = r["words"]

        metadata_lines.append(json.dumps(meta_entry, ensure_ascii=False))

        hf_row = {
            "id": cid,
            "audio": {"bytes": dest_wav.read_bytes(), "path": f"{cid}.wav"},
            "text": r["text"],
            "duration": r["duration"],
            "category": r["category"],
            "youtube_id": r["youtube_id"],
            "notes": r["notes"],
        }
        if include_timestamps:
            hf_row["words"] = r["words"]

        hf_dataset_rows.append(hf_row)

    # Write metadata.jsonl
    metadata_file = data_dir / "metadata.jsonl"
    metadata_file.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    click.echo(f"Saved metadata: {metadata_file} ({len(metadata_lines)} rows)")

    # Write README.md (Dataset Card)
    readme_file = output_dir / "README.md"
    readme_text = DATASET_CARD_TEMPLATE.replace("{total_clips}", str(len(records))).strip() + "\n"
    readme_file.write_text(readme_text, encoding="utf-8")
    click.echo(f"Saved dataset card: {readme_file}")

    # Build Parquet split using datasets library
    try:
        from datasets import Audio, Dataset
        click.echo("Generating Parquet file with native Audio feature...")
        ds = Dataset.from_list(hf_dataset_rows)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        parquet_file = data_dir / "test-00000-of-00001.parquet"
        ds.to_parquet(str(parquet_file))
        click.echo(f"Parquet dataset built: {parquet_file}")
    except Exception as e:
        click.echo(f"Warning: Failed to generate Parquet with datasets ({e}). Raw files & metadata.jsonl preserved.")

    click.echo(f"\n[OK] Dataset successfully packaged at: {output_dir}")

    # Push to Hugging Face Hub if requested
    if push:
        hf_token = token or os.getenv("HF_TOKEN")
        if not hf_token and Path(".env").exists():
            for line in Path(".env").read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "HF_TOKEN":
                        hf_token = v.strip().strip("'\"")
                        break
        click.echo(f"\nPushing dataset to Hugging Face Hub ({repo_id})...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=repo_id,
                repo_type="dataset",
            )
            click.echo(f"[SUCCESS] Dataset successfully published to: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            click.echo(f"[ERROR] Failed to push to Hugging Face Hub: {e}", err=True)
            raise click.Abort()


if __name__ == "__main__":
    export_dataset()
