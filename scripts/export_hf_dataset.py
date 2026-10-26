#!/usr/bin/env python3
"""
Package and publish the 47-clip Lithuanian ASR & Word-Boundary Alignment
Benchmark (zalia-ruta-bench-lt) to Hugging Face Datasets.

Features:
- Validates all 47 audio files and ground-truth SRT references.
- Exports structured data/ metadata (JSONL + 16 kHz WAV audio).
- Builds a HuggingFace Dataset with Audio and nested word timestamps.
- Generates a full Dataset Card (README.md) with YAML tags & Phase 1 leaderboard.
- Supports pushing directly to the Hugging Face Hub (with optional private flag).

Usage:
    # 1. Export locally to inspect:
    uv run python scripts/export_hf_dataset.py

    # 2. Export and push to Hugging Face Hub:
    uv run python scripts/export_hf_dataset.py --push --repo-id Friebay/zalia-ruta-bench-lt
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
DEFAULT_REPO_ID = "Friebay/zalia-ruta-bench-lt"
DEFAULT_EXPORT_DIR = Path("hf_export/zalia-ruta-bench-lt")


DATASET_CARD_TEMPLATE = """---
language:
- lt
license: cc-by-nc-4.0
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
task_ids:
- speech-recognition
pretty_name: "Žalia Rūta Lithuanian Speech & Word-Boundary Alignment Benchmark"
tags:
- speech
- audio
- lithuanian
- asr
- whisper
- forced-alignment
- profanity-detection
- benchmark
configs:
- config_name: default
  data_files:
  - split: test
    path: "data/metadata.jsonl"
---

# Žalia Rūta: Lithuanian Speech & Word-Boundary Alignment Benchmark (`zalia-ruta-bench-lt`)

A precision-audited evaluation dataset designed to benchmark **Lithuanian speech recognition (ASR)**, **word-level boundary precision (Forced Alignment)**, and **hallucination in silence gaps (Pause Bleed)**.

---

## Dataset Overview

- **Clips:** **47 unique clips** (16 kHz mono WAV, 16-bit PCM).
- **Ground Truth:** Audited word-level subtitles (1 word per cue) refined acoustically in Praat TextGrid down to ~10–20 ms boundary precision.
- **Extreme Challenge Cases:**
  - Fast conversational speech & debates (Alexander Stubb, Virginijus Sinkevičius, Vytautas Sinica).
  - Streamer slang, gaming exclamations, vulgarities, and compound Slavic/English loanwords.
  - Multi-second dead silence pauses and music reactions designed to stress-test Whisper hallucination loops.

---

## Phase 1 Leaderboard (Evaluated on All 47 Clips)

| System / Model | Macro WER% | Macro CER% | ≤50ms Precision | Pause Bleed% | Speed | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`paprika_acoustic`** (`paprika-whisper-lt-v3` + MMS + Refiner) | **13.43%** | **5.50%** | **77.7%** | **8.9%** | *Baseline* | *Baseline* |
| **`paprika_refined`** (`paprika-whisper-lt-v3` + MMS FA) | **13.43%** | **5.50%** | **69.9%** | **7.5%** | *Baseline* | *Baseline* |
| **`azuolas_acoustic`** (`azuolas-whisper-lt` + MMS + Refiner) | **19.09%** | **10.53%** | **77.0%** | **7.8%** | 0.1× | 9,462 MB |
| **`azuolas_refined`** (`azuolas-whisper-lt` + MMS FA) | **19.09%** | **10.53%** | **69.6%** | **6.6%** | 0.1× | 9,462 MB |
| **`turbo_acoustic`** (`whisper-large-v3-turbo` + MMS + Refiner) | **39.72%** | **31.69%** | **76.7%** | **6.6%** | 0.5× | 7,444 MB |
| **`turbo_refined`** (`whisper-large-v3-turbo` + MMS FA) | **39.72%** | **31.69%** | **68.6%** | **5.5%** | 0.5× | 7,444 MB |
| **`whisper_small_lt_acoustic`** (`whisper-small-lt` + MMS + Refiner) | **169.91%** | **120.62%** | **65.9%** | **14.0%** | 0.7× | 5,205 MB |
| **`whisper_small_lt_refined`** (`whisper-small-lt` + MMS FA) | **169.91%** | **120.62%** | **58.4%** | **13.0%** | 0.7× | 5,205 MB |

*Note: Benchmarked on local NVIDIA GTX 1650 (4 GB physical VRAM). Large v3 models experience PCIe system RAM paging overhead without quantization.*

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
    "words": [
        {"word": "Stubb", "start": 0.52, "end": 0.94},
        {"word": "apie", "start": 1.01, "end": 1.25},
        ...
    ],
    "duration": 11.33,
    "num_words": 23,
    "category": "speech",
    "youtube_id": "AYzmpjtX-90",
    "notes": "Alexander Stubb news short (11.3s)"
}
```

---

## Quick Usage

```python
from datasets import load_dataset

dataset = load_dataset("Friebay/zalia-ruta-bench-lt", split="test")

sample = dataset[0]
print("Clip ID:", sample["id"])
print("Transcript:", sample["text"])
print("First word:", sample["words"][0])
```

---

## Citation & License

- **License:** CC BY-NC 4.0 (Non-Commercial Research & Benchmarking).
- Sourced and curated as part of the [`zalia-ruta`](https://github.com/Friebay/zalia-ruta) automated Lithuanian speech and profanity bleeper project.
"""


def load_and_validate_records() -> List[Dict[str, Any]]:
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

        # Parse cues
        cues = parse_srt(srt_path.read_text(encoding="utf-8"))
        words = []
        spoken_tokens = []
        for cue in cues:
            clean_word = cue.text.strip()
            if not clean_word:
                continue
            words.append({
                "word": clean_word,
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
            })
            if not clean_word.startswith("#"):
                spoken_tokens.append(clean_word)

        text = " ".join(spoken_tokens).strip()

        records.append({
            "id": cid,
            "wav_path": wav_path,
            "srt_path": srt_path,
            "text": text,
            "words": words,
            "duration": duration,
            "num_words": len(words),
            "category": c.get("category", "speech"),
            "youtube_id": c.get("youtube_id", ""),
            "notes": c.get("notes", ""),
        })

    return records


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_EXPORT_DIR,
    help="Target directory for the packaged Hugging Face dataset.",
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
    help="Hugging Face Hub repository ID (e.g. Friebay/zalia-ruta-bench-lt).",
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
    push: bool,
    repo_id: str,
    private: bool,
    token: Optional[str],
):
    """Packages the 47-clip benchmark dataset for Hugging Face and optionally pushes to the Hub."""
    click.echo(f"Validating and preparing dataset from {BENCH_DIR}...")
    records = load_and_validate_records()
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
            "words": r["words"],
            "duration": r["duration"],
            "num_words": r["num_words"],
            "category": r["category"],
            "youtube_id": r["youtube_id"],
            "notes": r["notes"],
        }
        metadata_lines.append(json.dumps(meta_entry, ensure_ascii=False))

        hf_dataset_rows.append({
            "id": cid,
            "audio": str(dest_wav),
            "text": r["text"],
            "words": r["words"],
            "duration": r["duration"],
            "num_words": r["num_words"],
            "category": r["category"],
            "youtube_id": r["youtube_id"],
            "notes": r["notes"],
        })

    # Write metadata.jsonl
    metadata_file = data_dir / "metadata.jsonl"
    metadata_file.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    click.echo(f"Saved metadata: {metadata_file} ({len(metadata_lines)} rows)")

    # Write README.md (Dataset Card)
    readme_file = output_dir / "README.md"
    readme_file.write_text(DATASET_CARD_TEMPLATE.strip() + "\n", encoding="utf-8")
    click.echo(f"Saved dataset card: {readme_file}")

    # Build Parquet split using datasets library
    try:
        from datasets import Audio, Dataset
        click.echo("Generating Parquet file with native Audio feature...")
        ds = Dataset.from_list(hf_dataset_rows)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        parquet_file = data_dir / "train-00000-of-00001.parquet"
        ds.to_parquet(str(parquet_file))
        click.echo(f"Parquet dataset built: {parquet_file}")
    except Exception as e:
        click.echo(f"Warning: Failed to generate Parquet with datasets ({e}). Raw files & metadata.jsonl preserved.")

    click.echo(f"\n[OK] Dataset successfully packaged at: {output_dir}")

    # Push to Hugging Face Hub if requested
    if push:
        hf_token = token or os.getenv("HF_TOKEN")
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
