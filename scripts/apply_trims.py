import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from zalia_ruta.bench.srt_io import parse_srt, write_srt
from zalia_ruta.bench.trim import (
    ClipItem,
    discover_clips,
    find_trim_collisions,
    is_effective_trim,
    load_trim_plan,
    save_trim_plan,
    shift_and_clip_cues,
)
from zalia_ruta.bench.validate import validate_cues


def _run_ffmpeg(cmd: List[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({res.returncode}): {res.stderr.strip()}")


def _trim_wav(src_wav: Path, dst_wav: Path, start: float, end: float) -> None:
    tmp = dst_wav.with_name(f"{dst_wav.stem}.tmp.wav")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_wav),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(tmp),
        ]
    )
    tmp.replace(dst_wav)


def _trim_mp4(src_mp4: Path, dst_mp4: Path, start: float, end: float) -> None:
    tmp = dst_mp4.with_name(f"{dst_mp4.stem}.tmp.mp4")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_mp4),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(tmp),
        ]
    )
    tmp.replace(dst_mp4)


def _backup_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _find_benchmark_hyp_files(root: Path, clip_id: str) -> List[Path]:
    hyp_dir = root / "benchmark" / "hyp"
    if not hyp_dir.exists():
        return []
    matches: List[Path] = []
    for sys_dir in sorted(hyp_dir.iterdir()):
        if sys_dir.is_dir():
            for f in sys_dir.glob(f"{clip_id}.*"):
                if f.is_file():
                    matches.append(f)
    return matches


def _find_praat_files(root: Path, clip_id: str) -> List[Path]:
    praat_dir = root / "praat_samples"
    if not praat_dir.exists():
        return []
    return sorted(
        p for p in praat_dir.iterdir()
        if p.is_file() and (p.stem == clip_id or p.stem == f"{clip_id}_adjusted")
    )


def apply_trims(
    root: Path,
    plan_path: Path,
    dry_run: bool = False,
    allow_word_split: bool = False,
) -> int:
    plan = load_trim_plan(plan_path)
    plan_clips = plan.get("clips", {})
    clip_map = {c.key: c for c in discover_clips(root)}

    targets: List[tuple[ClipItem, dict]] = []
    for key, entry in plan_clips.items():
        if entry.get("status") != "trim":
            continue
        item = clip_map.get(key)
        if not item:
            print(f"[SKIP] {key}: clip no longer found on disk.")
            continue
        start = float(entry["start"])
        end = float(entry["end"])
        if not is_effective_trim(start, end, item.duration):
            print(f"[SKIP] {key}: start={start:.3f}, end={end:.3f} does not trim {item.duration:.3f}s.")
            continue
        targets.append((item, entry))

    if not targets:
        print("No pending 'trim' entries found in plan.")
        return 0

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / ".trim_backups" / run_stamp
    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"[{mode}] Processing {len(targets)} trimmed clip(s)...")

    applied_count = 0
    skipped_count = 0

    for item, entry in targets:
        start = float(entry["start"])
        end = float(entry["end"])
        new_dur = round(end - start, 3)

        cues = parse_srt(item.srt_path.read_text(encoding="utf-8")) if item.srt_path.exists() else []
        collisions = find_trim_collisions(cues, start, end)
        if collisions:
            desc = "; ".join(
                f"{c.side} cuts '{c.text}' [{c.cue_start:.3f}-{c.cue_end:.3f}] by {c.clipped_ms}ms"
                for c in collisions
            )
            if item.bucket == "benchmark" and not allow_word_split:
                print(f"[ERROR/SKIP] {item.key}: {desc}. Re-adjust boundaries or pass --allow-word-split.")
                skipped_count += 1
                continue
            print(f"[WARN] {item.key}: {desc}")

        new_cues = shift_and_clip_cues(cues, start, end) if cues else []
        if item.bucket == "benchmark":
            val_errors = validate_cues(new_cues, max_duration=new_dur)
            if val_errors:
                print(f"[ERROR/SKIP] {item.key}: shifted reference failed validation: {val_errors}")
                skipped_count += 1
                continue

        hyp_files = _find_benchmark_hyp_files(root, item.id) if item.bucket == "benchmark" else []
        praat_files = _find_praat_files(root, item.id) if item.bucket == "benchmark" else []

        print(
            f"[{mode}] {item.key}: {item.duration:.3f}s -> {new_dur:.3f}s "
            f"[{start:.3f}..{end:.3f}], cues {len(cues)} -> {len(new_cues)}"
            + (f", purge {len(hyp_files)} hyp file(s)" if hyp_files else "")
        )
        if praat_files:
            print(f"  [WARN] Out-of-sync praat_samples files for {item.id}: {[p.name for p in praat_files]}")

        if dry_run:
            applied_count += 1
            continue

        bdir = backup_root / item.bucket / item.id
        bak_wav = _backup_file(item.wav_path, bdir)
        _trim_wav(bak_wav, item.wav_path, start, end)

        if item.srt_path.exists():
            _backup_file(item.srt_path, bdir)
            write_srt(new_cues, item.srt_path)

        if item.bucket == "incoming":
            if item.review_wav_path and item.review_wav_path.parent.exists():
                if item.review_wav_path.exists():
                    _backup_file(item.review_wav_path, bdir / "review")
                shutil.copy2(item.wav_path, item.review_wav_path)
            if (
                item.review_srt_path
                and item.review_srt_path != item.srt_path
                and item.review_srt_path.parent.exists()
            ):
                if item.review_srt_path.exists():
                    _backup_file(item.review_srt_path, bdir / "review")
                if item.srt_path.exists():
                    write_srt(new_cues, item.review_srt_path)
            if item.video_path and item.video_path.exists():
                bak_mp4 = _backup_file(item.video_path, bdir / "review")
                _trim_mp4(bak_mp4, item.video_path, start, end)
        else:
            for hf in hyp_files:
                hf.unlink()

        entry["status"] = "applied"
        entry["applied_duration"] = new_dur
        entry["applied_at"] = datetime.now(timezone.utc).isoformat()
        save_trim_plan(plan_path, plan)
        applied_count += 1

    print(f"\nDone ({mode}): {applied_count} processed, {skipped_count} skipped.")
    if not dry_run and applied_count > 0:
        print(f"Backups saved under: {backup_root}")
    return 1 if skipped_count else 0


def main():
    parser = argparse.ArgumentParser(description="Apply clip trims from trim_plan.json to audio, video, and SRTs.")
    parser.add_argument("--plan", type=Path, default=Path("trim_plan.json"))
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files.")
    parser.add_argument(
        "--allow-word-split",
        action="store_true",
        help="Allow trimming benchmark clips even if a cut slices through a reference word cue.",
    )
    args = parser.parse_args()
    raise SystemExit(
        apply_trims(
            root=Path("."),
            plan_path=args.plan,
            dry_run=args.dry_run,
            allow_word_split=args.allow_word_split,
        )
    )


if __name__ == "__main__":
    main()

