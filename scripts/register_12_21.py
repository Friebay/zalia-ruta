import json
import sys
from pathlib import Path
import soundfile as sf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from zalia_ruta.bench.srt_io import Cue, write_srt
from zalia_ruta.bench.systems import BenchmarkRunner, KNOWN_MODELS
from zalia_ruta.bench.validate import validate_cues

CLIPS_CONFIG = [
    {
        "id": "karalius_prof_12",
        "video_id": "6hIRVqm5kdo",
        "notes": "Karalius rant on audience, '80% Lietuvos yra debilai... visą kontą debilams... žmonių yra debilai' (4 profanities)",
    },
    {
        "id": "karalius_prof_13",
        "video_id": "6hIRVqm5kdo",
        "notes": "Karalius politics discussion, 'nėra dalbajobų valdžioj... dalbajobų valdžioj... dalbajobų nėra valdžioj' (4 profanities)",
    },
    {
        "id": "karalius_prof_14",
        "video_id": "6hIRVqm5kdo",
        "notes": "Karalius commentary, 'kas juos labiau užpiso... idealaus pasaulio, kur nebus dalbajobų' (2 profanities)",
    },
    {
        "id": "karalius_prof_15",
        "video_id": "6hIRVqm5kdo",
        "notes": "Karalius content rant, 'žiūri žmonės dalbajobai ir jie liks dalbajobai' (2 profanities)",
    },
    {
        "id": "karalius_prof_16",
        "video_id": "6hIRVqm5kdo",
        "notes": "Karalius democracy commentary, 'išrenka debilus... tupi žmonės išrinktų debilus' (2 profanities)",
    },
    {
        "id": "karalius_prof_17",
        "video_id": "6hIRVqm5kdo",
        "notes": "Karalius audience rant, 'kuriame yra 80% debilų... edukuoti bent vieną du debilus' (2 profanities)",
    },
    {
        "id": "karalius_prof_18",
        "video_id": "3_wAkycMk0c",
        "notes": "Karalius politics critique, 'Visiem išpiso protus. Visiem išpiso protus dėl migrantų' (2 profanities)",
    },
    {
        "id": "karalius_prof_19",
        "video_id": "3_wAkycMk0c",
        "notes": "Karalius election commentary, 'planas yra toks: prapisam rinkimus... nes jie prapisa' (2 profanities)",
    },
    {
        "id": "karalius_prof_20",
        "video_id": "XnkFuZ5cBB8",
        "notes": "Karalius legal commentary, 'Gerai, debilu pavadinu žmogų... kažkokį debilą debilu pavadinau' (3 profanities)",
    },
    {
        "id": "karalius_prof_21",
        "video_id": "XnkFuZ5cBB8",
        "notes": "Karalius legal commentary, 'Tai yra riba peržengta... debilų žmogų debilą pavadini' (2 profanities)",
    },
]

runner = BenchmarkRunner(Path("benchmark"))
manifest_path = Path("benchmark/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
existing_ids = {c["id"] for c in manifest["clips"]}

for cfg in CLIPS_CONFIG:
    cid = cfg["id"]
    wav_path = Path(f"benchmark/audio/{cid}.wav")
    ref_path = Path(f"benchmark/references/{cid}.srt")
    
    if not wav_path.exists():
        print(f"[{cid}] Warning: audio file {wav_path} does not exist!")
        continue
        
    audio, sr = sf.read(str(wav_path))
    dur = len(audio) / sr
    print(f"[{cid}] Running paprika_ct2_acoustic ({dur:.1f}s)...")
    words = runner.run_model_acoustic(
        KNOWN_MODELS["paprika_ct2"], "paprika_ct2", cid, audio, sr=sr, force=True
    )
    
    cues = []
    prev_end = 0.0
    for idx, w in enumerate(words, start=1):
        s = round(float(w.start), 3)
        e = round(float(w.end), 3)
        if s < prev_end:
            s = prev_end
        if e <= s:
            e = round(s + 0.05, 3)
        if e > dur:
            e = round(dur, 3)
            if s >= e:
                s = max(0.0, round(e - 0.05, 3))
        cues.append(Cue(index=idx, start=s, end=e, text=w.word))
        prev_end = e
        
    errs = validate_cues(cues, max_duration=dur)
    if errs:
        print(f"[{cid}] Validation errors: {errs}")
    else:
        write_srt(cues, ref_path)
        print(f"[{cid}] Wrote {len(cues)} cues to {ref_path}")
        
    if cid not in existing_ids:
        manifest["clips"].append({
            "id": cid,
            "file": f"original_videos/{cid}.mp4",
            "youtube_id": cfg["video_id"],
            "category": "profanity",
            "timing_audited": False,
            "notes": cfg["notes"]
        })
        existing_ids.add(cid)

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Updated manifest: total clips now {len(manifest['clips'])}")

