import json
import subprocess
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Clean up scratch scripts named test_*.py in scripts/ so pytest only tests tests/
for f in Path("scripts").glob("test_*.py"):
    target = f.parent / f.name.replace("test_", "verify_")
    print(f"Renaming {f} -> {target}")
    f.rename(target)

# 2. Import the 5 audited broken clips from Praat
audited_clips = [
    "karalius_broken_01",
    "karalius_broken_02",
    "karalius_broken_05",
    "karalius_broken_07",
    "karalius_broken_08",
]

for cid in audited_clips:
    print(f"\nImporting {cid}_adjusted.TextGrid...")
    cmd = [
        "uv", "run", "python", "scripts/import_praat_textgrid.py",
        "--clip", cid,
        "--tier", "Acoustic_Refined"
    ]
    subprocess.run(cmd, check=True)

# 3. Remove skipped clips from manifest and delete their artifacts
skipped_clips = [
    "karalius_broken_03",
    "karalius_broken_04",
    "karalius_broken_06",
]

manifest_path = Path("benchmark/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

orig_count = len(manifest["clips"])
manifest["clips"] = [c for c in manifest["clips"] if c["id"] not in skipped_clips]
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nUpdated manifest.json: {orig_count} -> {len(manifest['clips'])} clips (removed {len(skipped_clips)} skipped clips).")

for cid in skipped_clips:
    for pattern in [
        f"benchmark/references/{cid}.srt",
        f"benchmark/audio/{cid}.wav",
        f"original_videos/{cid}.mp4",
        f"praat_samples/{cid}.TextGrid",
        f"praat_samples/{cid}.wav",
    ]:
        p = Path(pattern)
        if p.exists():
            p.unlink()
            print(f"Deleted un-audited file: {p}")

print("\nFinal release setup complete! Total benchmark clips:", len(manifest["clips"]))
