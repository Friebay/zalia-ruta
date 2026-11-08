#!/usr/bin/env python3
"""
Package and publish the 133-clip Lithuanian ASR & Profanity Benchmark (zalia-ruta-bench-lt)
to Hugging Face Datasets.

Exports:
- 16 kHz mono WAV audio files (embedded into Parquet).
- Verbatim & loanword-normalized Lithuanian transcripts.
- Profanity metadata (contains_profanity, profane_tokens).
- Word-level alignment timestamps for the 25 human-audited Praat clips (Option A: null for others).
- Non-commercial research dataset card (CC BY-NC 4.0).

Usage:
    # 1. Export locally to inspect:
    uv run python scripts/export_hf_dataset.py

    # 2. Export and push to Hugging Face Hub:
    uv run python scripts/export_hf_dataset.py --push --repo-id Frararo/zalia-ruta-bench-lt
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import soundfile as sf

from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.profanity import ProfanityDetector


BENCH_DIR = Path("benchmark")
DEFAULT_REPO_ID = "Frararo/zalia-ruta-bench-lt"
DEFAULT_EXPORT_DIR = Path("hf_export/zalia-ruta-bench-lt")

# The 25 clips with verified human Praat micro-adjustments
PRAAT_AUDITED_CLIPS = {
    "karalius_broken_01",
    "karalius_broken_02",
    "karalius_broken_05",
    "karalius_broken_07",
    "karalius_broken_08",
    "karalius_prof_02",
    "karalius_prof_03",
    "karalius_prof_05",
    "karalius_prof_07",
    "karalius_prof_08",
    "karalius_prof_10",
    "karalius_silence_02",
    "karalius_silence_03",
    "karalius_silence_04",
    "karalius_silence_05",
    "karalius_silence_06",
    "karalius_silence_07",
    "karalius_silence_08",
    "karalius_silence_09",
    "karalius_silence_10",
    "orlauskas_prof_02",
    "orlauskas_prof_03",
    "orlauskas_prof_04",
    "orlauskas_prof_05",
    "vaziuok_nachuj",
}

# Exact token substitutions for uncodified loanwords and common variants
EXACT_LOANWORD_MAP = {
    # blet variants
    "bliad": "blet",
    "bliat": "blet",
    "bled": "blet",
    "blėt": "blet",
    "bleit": "blet",
    "bld": "blet",
    "bliacha": "blecha",
    # nachui variants
    "nahui": "nachui",
    "nachuj": "nachui",
    "nahuj": "nachui",
    "nah": "nachui",
    "poxui": "pochui",
    "pohui": "pochui",
    "paxodu": "pochodu",
    # pyzda / pyzdec
    "pizda": "pyzda",
    "pizdec": "pyzdec",
    "pyzdiec": "pyzdec",
    "pizdiec": "pyzdec",
    "pzdc": "pyzdec",
    "pzd": "pyzda",
    # kurva
    "kurwa": "kurva",
    "krv": "kurva",
    # chuj / chujnia
    "xuj": "chuj",
    "chui": "chuj",
    "xujnia": "chujnia",
    "chuine": "chujnia",
    # brand names
    "facebook": "feisbuk",
    "youtube": "jutub",
    "youtubes": "jutubės",
    "youtubės": "jutubės",
    "discord": "diskord",
    "dicord": "diskord",
}

# Stem regex rules for inflections
STEM_LOANWORD_RULES = [
    # Normalizes 'pizd-' -> 'pyzd-' in any inflection (e.g. užpizdino -> užpyzdino, supizdėjęs -> supyzdėjęs)
    (re.compile(r"(\w*)pizd(\w*)", re.IGNORECASE), r"\1pyzd\2"),
    # Normalizes 'nahui' inside compound/attached forms (e.g. sukanahui -> sukanachui)
    (re.compile(r"(\w*)nahui(\w*)", re.IGNORECASE), r"\1nachui\2"),
    # Normalizes 'xuj' -> 'chuj' (e.g. naxuj -> nachui, paxui -> pochui)
    (re.compile(r"(\w*)xuj(\w*)", re.IGNORECASE), r"\1chuj\2"),
]

PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def normalize_token(token: str) -> str:
    lower_t = token.lower()
    if lower_t in EXACT_LOANWORD_MAP:
        return EXACT_LOANWORD_MAP[lower_t]
    res = lower_t
    for rx, repl in STEM_LOANWORD_RULES:
        res = rx.sub(repl, res)
    return res


def normalize_transcript(text: str) -> str:
    tokens = text.split()
    norm_tokens = []
    for t in tokens:
        clean_t = PUNCT_RE.sub("", t)
        if not clean_t:
            norm_tokens.append(t)
            continue
        norm_t = normalize_token(clean_t)
        norm_tokens.append(norm_t)
    return " ".join(norm_tokens)


DATASET_CARD_TEMPLATE = """---
language:
- lt
license: cc-by-nc-4.0
size_categories:
- n<1K
task_categories:
- automatic-speech-recognition
- audio-classification
pretty_name: "Žalia Rūta Lithuanian Speech & Profanity Benchmark"
tags:
- speech
- audio
- lithuanian
- asr
- benchmark
- profanity-detection
configs:
- config_name: default
  data_files:
  - split: test
    path: "data/test-*"
---

# Žalia Rūta: Lietuvių šnekos ir keiksmažodžių etalonas (`zalia-ruta-bench-lt`)

> [!WARNING]
> **Content Warning (Turinio įspėjimas):** Duomenų rinkinyje yra necenzūrinės leksikos, slaviškų skolinių ir vulgarių šnekamosios kalbos posakių. Rinkinys skirtas akademiniams tyrimams, automatiniam šnekos atpažinimui (ASR) ir turinio moderavimo / filtravimo algoritmų vertinimui.

Kuruotas vertinimo duomenų rinkinys, kurį sudaro **{total_clips} lietuviški garso įrašai** su etaloniniais tekstais šnekos atpažinimo (ASR) ir necenzūrinės leksikos aptikimo sistemų testavimui.

---

## Duomenų rinkinio apžvalga

- **Įrašai:** **{total_clips} unikalūs įrašai** (16 kHz mono WAV, 16-bit PCM).
- **Transkripcijos:** Dvigubas transkripcijos standartas:
  - `verbatim_transcript`: Tikslus sakytinis tekstas su autentiška fonetika ir šnekamosios kalbos ypatumais.
  - `normalized_transcript`: Standartizuota skolinių rašyba (*blet*, *nachui*, *pyzda*, *chuj*) be dirbtinių WER baudų.
- **Lygiavimas:**
  - **25 įrašai** turi žmogaus patikrintas ir „Praat“ akustiniu lygiu suderintas žodines laiko žymas (`words` masyvas, `alignment_tier = "human_praat_audited"`).
  - Likę **108 įrašai** turi `words = null` (`alignment_tier = "none"`), garantuojant, kad sintetiniai mašininiai lygiavimai nebus supainioti su žmogaus etalonu.

---

## Duomenų schema

Kiekvienas `test` įrašas turi šiuos laukus:

```json
{
  "id": "karalius_prof_02",
  "audio": {
    "bytes": "...",
    "path": "karalius_prof_02.wav",
    "sampling_rate": 16000
  },
  "text": "eik tu nachui",
  "verbatim_transcript": "eik tu nahui",
  "normalized_transcript": "eik tu nachui",
  "duration": 2.14,
  "contains_profanity": true,
  "profane_tokens": ["nachui"],
  "alignment_tier": "human_praat_audited",
  "words": [
    {"word": "eik", "start": 0.12, "end": 0.35, "is_profane": false},
    {"word": "tu", "start": 0.38, "end": 0.50, "is_profane": false},
    {"word": "nahui", "start": 0.55, "end": 1.02, "is_profane": true}
  ],
  "category": "speech",
  "youtube_id": "...",
  "notes": "..."
}
```

---

## Greitas naudojimas

```python
from datasets import load_dataset

dataset = load_dataset("Frararo/zalia-ruta-bench-lt", split="test")

# 1. Pavyzdys su žmogaus patikrintu žodžių lygiavimu
audited_clips = [s for s in dataset if s["alignment_tier"] == "human_praat_audited"]
print(f"Patikrinti Praat įrašai: {len(audited_clips)} / {len(dataset)}")

# 2. Keiksmažodžių aptikimo įvertinimas
profane_clips = [s for s in dataset if s["contains_profanity"]]
print(f"Įrašai su keiksmažodžiais: {len(profane_clips)} / {len(dataset)}")
```

---

## Oficiali etalono lyderių lentelė

Įvertinta visuose **{total_clips} įrašuose** (5 146 etalono žodžiai):

| Vieta | Modelis | Modelio tipas | Mikro WER% (Viso korpuso) | Makro WER% | Mikro CER% | WER (Be diakritikų) | RTF | Greitis | Aparatinė įranga |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | Whisper Large v3 (CT2 INT8) | **11.08%** | **12.86%** | **4.28%** | **10.30%** | **0.28** | **3.6x** | NVIDIA GTX 1650 (4 GB) |
| 🥈 | **[paprika-whisper-lt-v3](https://huggingface.co/kristijonas/paprika-whisper-lt-v3)** | Whisper Large v3 (PyTorch FP16)* | **11.36%** | **12.33%** | **4.13%** | **10.51%** | **0.26** | **3.9x** | NVIDIA Tesla T4 (16 GB) |
| 🥉 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | Whisper Large v3 (CT2 INT8) | **11.60%** | **13.81%** | **4.96%** | **10.88%** | **1.30** | **0.8x** | NVIDIA GTX 1650 (4 GB) |
| 4 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | Whisper Large v3 Turbo (OpenAI) | **29.93%** | **33.82%** | **14.34%** | **27.40%** | **1.89** | **0.5x** | NVIDIA GTX 1650 (4 GB) |
| 5 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | Whisper Small (DeividasM) | **74.66%** | **104.44%** | **36.51%** | **71.73%** | **0.72** | **1.4x** | NVIDIA GTX 1650 (4 GB) |
| 6 | **[svogunas-turbo-lt](https://huggingface.co/svogunas/whisper-large-v3-turbo-lt)** | Whisper Large v3 Turbo (CT2 INT8)** | **87.78%** | **83.89%** | **75.00%** | **87.54%** | **1.55** | **0.6x** | NVIDIA GTX 1650 (4 GB) |

> \* **Pastaba dėl `paprika-whisper-lt-v3 (PyTorch FP16)`:** Vykdyta per Hugging Face `transformers` su `no_repeat_ngram_size=3` ir `repetition_penalty=1.15` generavimo nustatymais (kurie neleidžia Whisper modeliui patekti į haliucinacijų kilpas emocingose ar juoko atkarpose). Pilnas modelis reikalauja ~7.4 GB VRAM, todėl greitis (3.9x) išmatuotas su NVIDIA Tesla T4 (ant 4 GB kortų sistema veikia žymiai lėčiau dėl PCIe atminties puslapiavimo).
>
> \*\* **Pastaba dėl `svogunas-turbo-lt`:** Testuota su CTranslate2 INT8 kvantavimu (`int8_float16`). Dėl 4 sluoksnių Turbo architektūros tiesioginis INT8 kvantavimas sukelia svorių kolapsą ir nuolatines haliucinacijas šnekamojoje kalboje. Jei turite kokybiškesnę kvantavimo konfigūraciją ar pilną FP16/FP32 testą, pateikite PR / Diskusiją.

---

### Keiksmažodžių aptikimo lyderių lentelė

| Vieta | Modelis | Atpažinta / Viso | Praleista | Klaidingi signalai | Atpažinimas% (Recall) | Tikslumas% (Precision) | F1 balas |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3 (CT2 INT8)](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | **233 / 255** | 22 | 0 | **91.37%** | **100.00%** | **95.49%** |
| 🥈 | **[paprika-whisper-lt-v3 (PyTorch FP16)](https://huggingface.co/kristijonas/paprika-whisper-lt-v3)** | **230 / 255** | 25 | 0 | **90.20%** | **100.00%** | **94.85%** |
| 🥉 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | **224 / 255** | 31 | 1 | **87.84%** | **99.56%** | **93.33%** |
| 4 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | **206 / 255** | 49 | 0 | **80.78%** | **100.00%** | **89.37%** |
| 5 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | **162 / 255** | 93 | 1 | **63.53%** | **99.39%** | **77.51%** |
| 6 | **[svogunas-turbo-lt](https://huggingface.co/svogunas/whisper-large-v3-turbo-lt)** | **18 / 255** | 237 | 4 | **7.06%** | **81.82%** | **13.00%** |

---

## Naudojimo sąlygos ir licencija

- **Licencija:** [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
- **Paskirtis:** Ne pelno siekiantys moksliniai tyrimai, akademinis vertinimas ir individualus testavimas.
- **Pirminis projektas:** Sukurta kaip [`zalia-ruta`](https://github.com/Friebay/zalia-ruta) lietuvių šnekos apdorojimo projekto dalis.
"""


def load_and_validate_records() -> List[Dict[str, Any]]:
    """Loads manifest and cross-verifies audio files, SRT references, and profanity tags."""
    manifest_file = BENCH_DIR / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    clips = manifest_data.get("clips", [])
    records = []
    detector = ProfanityDetector()

    for c in clips:
        cid = c["id"]
        wav_path = BENCH_DIR / "audio" / f"{cid}.wav"
        srt_path = BENCH_DIR / "references" / f"{cid}.srt"

        if not wav_path.exists():
            raise FileNotFoundError(f"[{cid}] Audio missing: {wav_path}")
        if not srt_path.exists():
            raise FileNotFoundError(f"[{cid}] Reference SRT missing: {srt_path}")

        info = sf.info(str(wav_path))
        duration = round(info.duration, 3)

        cues = parse_srt(srt_path.read_text(encoding="utf-8"))
        is_praat_audited = cid in PRAAT_AUDITED_CLIPS

        spoken_tokens = []
        words_list = []
        for cue in cues:
            clean_word = cue.text.strip()
            if not clean_word:
                continue
            if not clean_word.startswith("#"):
                spoken_tokens.append(clean_word)
                if is_praat_audited:
                    words_list.append({
                        "word": clean_word,
                        "start": round(cue.start, 3),
                        "end": round(cue.end, 3),
                        "is_profane": detector.is_profanity(clean_word),
                    })

        verbatim_transcript = " ".join(spoken_tokens).strip()
        norm_transcript = normalize_transcript(verbatim_transcript)

        # Detect profanities present in the normalized transcript
        detected_profane = sorted(list({w for w in norm_transcript.split() if detector.is_profanity(w)}))
        has_prof = len(detected_profane) > 0

        rec = {
            "id": cid,
            "wav_path": wav_path,
            "srt_path": srt_path,
            "text": verbatim_transcript,
            "verbatim_transcript": verbatim_transcript,
            "normalized_transcript": norm_transcript,
            "duration": duration,
            "contains_profanity": has_prof,
            "profane_tokens": detected_profane,
            "alignment_tier": "human_praat_audited" if is_praat_audited else "none",
            "words": words_list if is_praat_audited else None,
            "category": c.get("category", "speech"),
            "youtube_id": c.get("youtube_id", ""),
            "notes": c.get("notes", ""),
        }
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
    push: bool,
    repo_id: str,
    private: bool,
    token: Optional[str],
):
    """Packages the 133-clip benchmark dataset for Hugging Face and optionally pushes to the Hub."""
    click.echo(f"Validating and preparing dataset from {BENCH_DIR}...")
    records = load_and_validate_records()
    praat_count = sum(1 for r in records if r["alignment_tier"] == "human_praat_audited")
    profane_count = sum(1 for r in records if r["contains_profanity"])
    click.echo(f"Loaded {len(records)} clips ({praat_count} human Praat-audited, {profane_count} with profanity).")

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
            "verbatim_transcript": r["verbatim_transcript"],
            "normalized_transcript": r["normalized_transcript"],
            "duration": r["duration"],
            "contains_profanity": r["contains_profanity"],
            "profane_tokens": r["profane_tokens"],
            "alignment_tier": r["alignment_tier"],
            "words": r["words"],
            "category": r["category"],
            "youtube_id": r["youtube_id"],
            "notes": r["notes"],
        }
        metadata_lines.append(json.dumps(meta_entry, ensure_ascii=False))

        hf_row = {
            "id": cid,
            "audio": {"bytes": dest_wav.read_bytes(), "path": f"{cid}.wav"},
            "text": r["text"],
            "verbatim_transcript": r["verbatim_transcript"],
            "normalized_transcript": r["normalized_transcript"],
            "duration": r["duration"],
            "contains_profanity": r["contains_profanity"],
            "profane_tokens": r["profane_tokens"],
            "alignment_tier": r["alignment_tier"],
            "words": r["words"],
            "category": r["category"],
            "youtube_id": r["youtube_id"],
            "notes": r["notes"],
        }
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
        click.echo("Generating Parquet file with native Audio feature and nullable words...")
        ds = Dataset.from_list(hf_dataset_rows)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        parquet_file = data_dir / "test-00000-of-00001.parquet"
        ds.to_parquet(str(parquet_file))
        click.echo(f"Parquet dataset built successfully: {parquet_file}")
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
