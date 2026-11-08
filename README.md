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

Saugykloje yra empirinis šnekos atpažinimo etalonas, vertinantis lietuvių kalbos ASR tikslumą ir akustinių žodžių ribų tikslumą prieš **133 žmogaus patikrintus etaloninius įrašus** (5 146 etalono žodžiai): nuo žinių laidų ir politinių debatų iki laisvo stiliaus transliuotojų (streamerių) reakcijų bei gausios necenzūrinės leksikos.

Duomenų rinkinys viešai pasiekiamas „Hugging Face“ platformoje: [**Frararo/zalia-ruta-bench-lt**](https://huggingface.co/datasets/Frararo/zalia-ruta-bench-lt).

### Oficiali lyderių lentelė (133 įrašai)

| Vieta | Modelis | Modelio tipas | Mikro WER% (Viso korpuso) | Makro WER% | Mikro CER% | WER (Be diakritikų) | RTF | Greitis |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | Whisper Large v3 (CT2 INT8) | **11.08%** | **12.86%** | **4.28%** | **10.30%** | **0.28** | **3.6x** |
| 🥈 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | Whisper Large v3 (CT2 INT8) | **11.60%** | **13.81%** | **4.96%** | **10.88%** | **1.30** | **0.8x** |
| 🥉 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | Whisper Large v3 Turbo (OpenAI) | **29.93%** | **33.82%** | **14.34%** | **27.40%** | **1.89** | **0.5x** |
| 4 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | Whisper Small (DeividasM) | **74.66%** | **104.44%** | **36.51%** | **71.73%** | **0.72** | **1.4x** |
| 5 | **[svogunas-turbo-lt](https://huggingface.co/svogunas/whisper-large-v3-turbo-lt)** | Whisper Large v3 Turbo (CT2 INT8)* | **87.78%** | **83.89%** | **75.00%** | **87.54%** | **1.55** | **0.6x** |

> \* **Pastaba dėl `svogunas-turbo-lt`:** Testuota su CTranslate2 INT8 kvantavimu (`int8_float16`). Dėl 4 sluoksnių Turbo architektūros tiesioginis INT8 kvantavimas sukelia svorių kolapsą ir nuolatines haliucinacijas šnekamojoje kalboje. Jei turite kokybiškesnę kvantavimo konfigūraciją ar pilną FP16/FP32 testą, pateikite PR / Diskusiją.

* **Realaus laiko koeficientas (RTF):** Bendros transkripcijos trukmės ir garso trukmės santykis (`trukmė / garso_trukmė`). Kuo mažesnis – tuo greičiau; RTF < 1.0 reiškia greičiau nei realiu laiku.
* **Greitis:** Pagreitinimo koeficientas (`garso_trukmė / trukmė` = `1 / RTF`). Išmatuota su NVIDIA GTX 1650 (4 GB VRAM).

### Keiksmažodžių cenzūravimo lyderių lentelė (Atpažinimo dažnis)

Vertina, kaip patikimai kiekvienas modelis atpažįsta lietuviškus ir slaviškus keiksmažodžius tarp **255 žmogaus patvirtintų etaloninių keiksmažodžių** visame duomenų rinkinyje. Tai esminis rodiklis automatizuotam cenzūravimui ir bleepinimui:

| Vieta | Modelis | Atpažinta / Viso | Praleista | Klaidingi signalai | Atpažinimas% (Recall) | Tikslumas% (Precision) | F1 balas |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **[paprika-whisper-lt-v3](https://huggingface.co/RobertasTa/paprika-whisper-lt-v3-ct2-int8)** | **233 / 255** | 22 | 0 | **91.37%** | **100.00%** | **95.49%** |
| 🥈 | **[azuolas-whisper-lt](https://huggingface.co/akisviete/azuolas-whisper-lt)** | **224 / 255** | 31 | 1 | **87.84%** | **99.56%** | **93.33%** |
| 🥉 | **[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | **206 / 255** | 49 | 0 | **80.78%** | **100.00%** | **89.37%** |
| 4 | **[whisper-small-lt](https://huggingface.co/DeividasM/whisper-small-lt)** | **162 / 255** | 93 | 1 | **63.53%** | **99.39%** | **77.51%** |
| 5 | **[svogunas-turbo-lt](https://huggingface.co/svogunas/whisper-large-v3-turbo-lt)** | **18 / 255** | 237 | 4 | **7.06%** | **81.82%** | **13.00%** |

* **Atpažinimas% (Recall / Catch Rate):** `(Atpažinti keiksmažodžiai / Visi etaloniniai keiksmažodžiai) × 100%`. Svarbiausias saugaus cenzūravimo rodiklis.
* **Tikslumas% (Precision):** `(Atpažinti keiksmažodžiai / (Atpažinti keiksmažodžiai + Klaidingi signalai)) × 100%`. Rodo, kaip retai be reikalo užbleepinami švarūs žodžiai.

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
