# 🌿 Žalia-Rūta (zalia-ruta)

Automatizuotas vaizdo ir garso įrašų necenzūrinės leksikos cenzūravimo (iškirpimo / „bleepinimo“) įrankis, sukurtas specialiai lietuvių kalbai.

Įrankis be nuostolių išsaugo vaizdo srautą (`-c:v copy`), atpažįsta šneką naudodamas vietinį lietuvišką „Whisper“ modelį (`kristijonas/paprika-whisper-lt-v3`), patikslina žodžių ribas pasitelkdamas priverstinį lygiavimą **MMS Forced Alignment** (`torchaudio.pipelines.MMS_FA`), aptinka linksniuojamus lietuviškus bei slaviškus keiksmažodžius, uždeda tolygiai prislopintą 1000 Hz garsinį signalą be spragsėjimų ir eksportuoja suderintus žodinio lygio `.srt` subtitrus.

---

## Pagrindinės savybės

1. **0 % vaizdo kokybės praradimas**: Naudojamas tiesioginis FFmpeg srauto kopijavimas (`-c:v copy`), todėl vaizdo įrašas neperkoduojamas, neprarandama kokybė ir negaištamas laikas.
2. **Specializuotas lietuvių kalbos ASR**: Veikia su `kristijonas/paprika-whisper-lt-v3` modeliu NVIDIA CUDA aplinkoje (`float16`), pasiekiančiu ~3,5 % vidutinį WER lietuvių šnekai.
3. **MMS priverstinio lygiavimo tikslumas**: Naudojamas `torchaudio.pipelines.MMS_FA`, kuris pašalina pauzių užgriebimą (tylos pauzių perteklinį bleepinimą sumažina nuo 35 %–100 % iki **0 %–2 %**), užtikrindamas, kad garsinis signalas pasibaigtų tiksliai kartu su ištartu žodžiu.
4. **Išmanus skaidymas pagal akustines pauzes**: Ilgi garso įrašai automatiškai karpomi ties natūraliais energijos minimumais (20,0 s – 29,0 s intervalais), taip išvengiama CTC lygiavimo poslinkių, o VRAM naudojimas išlieka mažesnis nei 50 MB bet kokio ilgio vaizdo įrašams.
5. **Linksnių ir žargono atpažinimo variklis**:
   - Diakritikams nejautri paieška (`NFKD` normalizavimas), atpažįstanti tokias šaknis kaip `blėt`, `blet`, `bliat`, `buliot`, `šūdas` ir kt.
   - Dūrinių ir priesagų aptikimas (pvz., `sukanachų`).
   - Pagrindinės šaknys: `blet*`, `nachui*`, `suka*`, `kurva*`, `pis*`, `pyzd*`, `jobt*`, `byb*`, `šikna*`, `debil*`.
6. **Bleepinimas be spragsėjimų**: Keiksmažodžiai pakeičiami 1000 Hz sinusinės bangos signalu su 5 ms tolygiu perėjimu (crossfade) ties ribomis, todėl garse nelieka jokių spragtelėjimų.
7. **Sinchronizuoti subtitrai**: Generuojami suderinti žodinio lygio `.srt` subtitrai, kuriuose cenzūruoti žodžiai pažymimi žyme `[BLEEP: žodis]`.

---

## Diegimas

### Reikalavimai
- **Python 3.11+**
- **FFmpeg**, įdiegtas ir pasiekiamas per sisteminį `PATH`.
- **NVIDIA vaizdo plokštė** su CUDA palaikymu (rekomenduojama).

### Diegimas naudojant `uv` (Rekomenduojama)

```powershell
# Klonuokite saugyklą
git clone https://github.com/Friebay/zalia-ruta.git
cd zalia-ruta

# Sinchronizuokite aplinką su fiksuotomis CUDA PyTorch priklausomybėmis
uv sync

# Pasirinktinai: įtraukite testavimo (benchmark) priklausomybes (jiwer, rapidfuzz)
uv sync --extra bench
```

---

## Naudojimas

### Pagrindinė CLI komanda

```powershell
# Paleidimas per uv
uv run zalia-ruta -i "original_videos/mano_video.mp4"

# Arba naudojant Windows paleidimo failą
.\zalia-ruta.bat -i "original_videos/mano_video.mp4"
```

Pagal numatytuosius nustatymus rezultatai išsaugomi kataloge:
```text
output/
└── mano_video/
    ├── mano_video_bleeped.mp4
    └── mano_video_bleeped.srt
```

### Pasirinktinė išvesties vieta

```powershell
uv run zalia-ruta -i "original_videos/mano_video.mp4" -o "output/mano_video/mano_cenzuruotas.mp4"
```

### Parametrai

```text
Naudojimas: zalia-ruta [OPTIONS]

Parametrai:
  -i, --input FILE                Kelias iki vaizdo arba garso failo. [privaloma]
  -o, --output FILE               Kelias iki išvesties failo. Pagal nutylėjimą: output/<failas>/<failas>_bleeped<plėtinys>.
  --custom-words FILE             Papildomų keiksmažodžių ar reguliariųjų reiškinių (regex) tekstinis failas.
  --freq FLOAT                    Garsinio signalo dažnis hercais (Hz). [numatyta: 1000.0]
  --volume FLOAT                  Garsinio signalo garsumas (nuo 0.0 iki 1.0). [numatyta: 0.35]
  --pad-start FLOAT               Apsauginis laiko tarpas sekundėmis prieš keiksmažodžio pradžią. [numatyta: 0.04]
  --pad-end FLOAT                 Apsauginis laiko tarpas sekundėmis po keiksmažodžio pabaigos. [numatyta: 0.04]
  --device [cuda|cpu]             Prietaisas „Whisper“ modeliui vykdyti. [numatyta: automatiškai nustatoma]
  --mms-align / --no-mms-align    Naudoti MMS priverstinį lygiavimą ribų tikslumui. [numatyta: mms-align]
  --export-srt / --no-export-srt  Eksportuoti atskirą .srt subtitrų failą. [numatyta: export-srt]
  --keep-temp / --no-keep-temp    Išsaugoti tarpinius išgautus WAV failus. [numatyta: no-keep-temp]
  -h, --help                      Rodyti šį pranešimą ir išeiti.
```

### Papildomų žodžių failas

Sukurkite tekstinį failą (pvz., `custom_words.txt`):
```text
# Tikslios šaknys arba šablonai su žvaigždute
durnelis
durne*
# Reguliarieji reiškiniai (regex)
^kvail.*$
```
Paleiskite nurodydami parametrą:
```powershell
uv run zalia-ruta -i "video.mp4" --custom-words "custom_words.txt"
```

---

## Lietuvių šnekos žodinio lygio etalonas (`zalia-bench`)

Saugykloje yra empirinis šnekos atpažinimo etalonas, vertinantis lietuvių kalbos ASR tikslumą ir akustinių žodžių ribų tikslumą prieš **170 žmogaus patikrintų etaloninių įrašų** (6 560 etalono žodžių): nuo žinių laidų ir politinių debatų iki „YouTube“ / „TikTok“ vaizdo įrašų bei gausios šnekamosios ir necenzūrinės leksikos.

Duomenų rinkinys viešai pasiekiamas „Hugging Face“ platformoje: [**Frararo/zalia-ruta-bench-lt**](https://huggingface.co/datasets/Frararo/zalia-ruta-bench-lt).

### Oficiali lyderių lentelė (170 įrašų)

| Vieta | Modelis | Modelio tipas | Mikro WER% (visas korpusas) | Makro WER% | Mikro CER% | WER (be diakritikų) | RTF | Greitis | Aparatinė įranga |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | Whisper Large v3 (CT2 INT8) | **13.73%** | **15.62%** | **4.75%** | **12.44%** | **0.033** | **30.1x** | Google Colab GPU (~2.7 GB VRAM) |
| 🥈 | **[paprika-whisper-lt-v3 (CT2 FP16)](https://huggingface.co/kristijonas/paprika-whisper-lt-v3)** | Whisper Large v3 (CT2 FP16) | **14.21%** | **15.85%** | **5.02%** | **12.88%** | **0.035** | **28.2x** | Google Colab GPU (~2.7 GB VRAM) |
| 🥉 | **[upelis-0.6b-lt-v1](https://huggingface.co/datolabs/upelis-0.6b-lt-v1)** | NeMo FastConformer / Hybrid (0.6B)* | **15.73%** | **17.63%** | **6.38%** | **14.70%** | **0.080** | **12.5x** | Google Colab GPU / MMS_FA |
| 4 | **[lt-ausys](https://huggingface.co/RobertasTa/lt-ausys)** | ONNX ASR (FastConformer) | **16.17%** | **22.47%** | **5.68%** | **15.15%** | **0.045** | **22.3x** | Google Colab GPU (~2.7 GB VRAM) |
| 5 | **[paprika-whisper-lt-v3](https://huggingface.co/kristijonas/paprika-whisper-lt-v3)** | Whisper Large v3 (PyTorch FP16)** | **20.21%** | **20.91%** | **6.67%** | **18.61%** | **0.211** | **4.7x** | Google Colab GPU (~6.6 GB VRAM) |
| 6 | **[svogunas-turbo-lt](https://huggingface.co/svogunas/whisper-large-v3-turbo-lt)** | Whisper Large v3 Turbo (PyTorch FP16) | **23.02%** | **32.38%** | **8.11%** | **21.31%** | **0.205** | **4.9x** | Google Colab GPU (~6.6 GB VRAM) |
| 7 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | Whisper Large v3 (PyTorch FP16) | **26.52%** | **25.89%** | **16.09%** | **25.30%** | **0.481** | **2.1x** | Google Colab GPU (~12.8 GB VRAM) |
| 8 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | Whisper Large v3 Turbo (OpenAI) | **34.97%** | **37.57%** | **13.10%** | **32.01%** | **0.216** | **4.6x** | Google Colab GPU (~6.7 GB VRAM) |
| 9 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | Whisper Small (DeividasM) | **64.83%** | **64.31%** | **24.68%** | **61.71%** | **0.194** | **5.2x** | Google Colab GPU (~5.6 GB VRAM) |

> \* **Pastaba dėl `upelis-0.6b-lt-v1`:** Žalia transkripcija (`upelis_raw`: RTF **0.050**, **20.2x** greitis, **4824 MB** VRAM) vykdyta su „NeMo“ „Google Colab“ GPU aplinkoje, o žodžių ribų lygiavimas (`upelis_acoustic`: bendras RTF **0.080**, **12.5x** greitis) pritaikytas per MMS Forced Alignment + akustinį tikslinimą.
>
> \*\* **Pastaba dėl `paprika-whisper-lt-v3 (PyTorch FP16)`:** Vykdyta per Hugging Face `transformers` su `no_repeat_ngram_size=3` ir `repetition_penalty=1.15` generavimo nustatymais. Visi aukščiau pateikti lentelės įverčiai atitinka `*_acoustic` sistemą (su MMS Forced Alignment ir akustiniu ribų tikslinimu).

* **Realaus laiko koeficientas (RTF):** Bendros transkripcijos trukmės ir garso trukmės santykis (`trukmė / garso_trukmė`). Kuo mažesnis – tuo greičiau; RTF < 1.0 reiškia greičiau nei realiu laiku.
* **Greitis:** Pagreitinimo koeficientas (`garso_trukmė / trukmė` = `1 / RTF`).

### Keiksmažodžių cenzūravimo lyderių lentelė (Atpažinimo dažnis)

Vertina, kaip patikimai kiekvienas modelis atpažįsta lietuviškus ir slaviškus keiksmažodžius tarp **417 žmogaus patvirtintų etaloninių keiksmažodžių** visame duomenų rinkinyje. Tai esminis rodiklis automatizuotam cenzūravimui ir bleepinimui:

| Vieta | Modelis | Aptikta / Viso | Praleisti | Klaidingi teigiami (False Alarms) | Aptikimo dažnis% (Recall) | Tikslumas% (Precision) | F1 įvertis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3 (CT2 FP16)](https://huggingface.co/kristijonas/paprika-whisper-lt-v3)** | **360 / 417** | 57 | 1 | **86.33%** | **99.72%** | **92.54%** |
| 🥈 | **[paprika-whisper-lt-v3 (CT2 INT8)](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | **351 / 417** | 66 | 0 | **84.17%** | **100.00%** | **91.41%** |
| 🥉 | **[paprika-whisper-lt-v3 (PyTorch FP16)](https://huggingface.co/kristijonas/paprika-whisper-lt-v3)** | **332 / 417** | 85 | 1 | **79.62%** | **99.70%** | **88.53%** |
| 4 | **[svogunas-turbo-lt (PyTorch FP16)](https://huggingface.co/svogunas/whisper-large-v3-turbo-lt)** | **315 / 417** | 102 | 2 | **75.54%** | **99.37%** | **85.83%** |
| 5 | **[azuolas-whisper-lt (PyTorch FP16)](https://huggingface.co/akisviete/azuolas-whisper-lt)** | **312 / 417** | 105 | 2 | **74.82%** | **99.36%** | **85.36%** |
| 6 | **[whisper-large-v3-turbo (OpenAI)](https://huggingface.co/openai/whisper-large-v3-turbo)** | **295 / 417** | 122 | 1 | **70.74%** | **99.66%** | **82.75%** |
| 7 | **[upelis-0.6b-lt-v1 (datolabs)](https://huggingface.co/datolabs/upelis-0.6b-lt-v1)** | **294 / 417** | 123 | 0 | **70.50%** | **100.00%** | **82.70%** |
| 8 | **[lt-ausys (RobertasTa)](https://huggingface.co/RobertasTa/lt-ausys)** | **280 / 417** | 137 | 1 | **67.15%** | **99.64%** | **80.23%** |
| 9 | **[whisper-small-lt (DeividasM)](https://huggingface.co/DeividasM/whisper-small-lt)** | **250 / 417** | 167 | 6 | **59.95%** | **97.66%** | **74.29%** |

* **Aptikimo dažnis% (Recall / Catch Rate):** `(Aptikti keiksmažodžiai / Visi etaloniniai keiksmažodžiai) × 100%`. Svarbiausias saugaus cenzūravimo rodiklis.
* **Tikslumas% (Precision):** `(Aptikti keiksmažodžiai / (Aptikti keiksmažodžiai + Klaidingi teigiami)) × 100%`. Rodo, kaip retai be reikalo užbleepinami švarūs žodžiai.

*Išsami kiekvieno įrašo ataskaita pateikiama faile [benchmark/reports/results.md](benchmark/reports/results.md).*

---

### Etalono paleidimas lokaliai

```powershell
# 1. Patikrinkite etaloninių SRT failų formatą
uv run zalia-bench validate

# 2. Paleiskite gamybinį modelį (paprika) su realaus laiko eigos juosta
uv run zalia-bench run

# 3. Paleiskite kitus modelius (pvz., whisper_small_lt, turbo, azuolas, svogunas)
uv run zalia-bench run --model whisper_small_lt
uv run zalia-bench run --model turbo
uv run zalia-bench run --model azuolas
uv run zalia-bench run --model svogunas

# 4. Įvertinkite modelių rezultatus prieš žmogaus etalonus
uv run zalia-bench score

# Pasirinktinai: vertinti konkretų įrašą ar modelį
uv run zalia-bench score --clip <clip_id>
uv run zalia-bench score --model paprika_ct2
uv run zalia-bench score --all-systems
```

---

### Kitų modelių vertinimas

Kiekvienas tyrėjas gali įvertinti savo ASR modelį su šiuo duomenų rinkiniu naudodamas „Python“:

```python
import re
import jiwer
from datasets import load_dataset

# 1. Užkraukite etalono duomenų rinkinį
ds = load_dataset("Frararo/zalia-ruta-bench-lt", split="test")

# 2. Oficialus etalono normalizatorius
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

# 3. Transkribuokite su savo modeliu
refs, hyps = [], []
for sample in ds:
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    
    predicted_text = my_model.transcribe(audio_array, sampling_rate) # <-- Čia nurodykite savo modelį
    
    refs.append(normalize(sample["text"]))
    hyps.append(normalize(predicted_text))

# 4. Oficialus lyderių lentelės WER rezultatas
micro_wer = jiwer.wer(refs, hyps) * 100
print(f"Oficialus Mikro WER: {micro_wer:.2f}%")
```

---

## Testų paleidimas

```powershell
uv run pytest
```
Automatinių testų rinkinys tikrina:
* MMS priverstinio lygiavimo tokenizavimą, diakritikų normalizavimą ir atsarginius mechanizmus
* Medijos parametrų nuskaitymą, garso išgavimą ir FFmpeg perkodavimo be nuostolių srautus
* Pauzių paiešką pagal RMS energijos minimumus ir garso skaidymą
* Keiksmažodžių šablonų atpažinimą ir dūrinių aptikimą
* Sinusinės bangos generavimą, tolygų perėjimą ir laiko ribų apsaugas
* Subtitrų formatavimą ir SRT struktūros patikrą
* Etalono vertinimo metrikas (WER, CER, pauzių perteklių ir medianinius laiko poslinkius)
