import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf

from zalia_ruta.bench.srt_io import Cue


@dataclass
class ClipItem:
    key: str
    bucket: str  # "incoming" or "benchmark"
    id: str
    wav_path: Path
    srt_path: Path
    duration: float
    video_path: Optional[Path] = None
    review_srt_path: Optional[Path] = None
    review_wav_path: Optional[Path] = None


@dataclass
class TrimCollision:
    side: str  # "start" or "end"
    cue_index: int
    text: str
    cue_start: float
    cue_end: float
    cut_time: float
    clipped_ms: int


def is_effective_trim(start: float, end: float, duration: float, eps: float = 0.005) -> bool:
    """Returns True if [start, end] meaningfully trims either edge of [0, duration]."""
    return start > eps or end < (duration - eps)


def find_trim_collisions(
    cues: List[Cue],
    trim_start: float,
    trim_end: float,
    tolerance: float = 0.02,
) -> List[TrimCollision]:
    """
    Detects cues whose interior is sliced by trim_start or trim_end beyond `tolerance` seconds.
    """
    collisions: List[TrimCollision] = []
    for idx, cue in enumerate(cues, start=1):
        if cue.start + tolerance < trim_start < cue.end - tolerance:
            collisions.append(
                TrimCollision(
                    side="start",
                    cue_index=idx,
                    text=cue.text,
                    cue_start=cue.start,
                    cue_end=cue.end,
                    cut_time=trim_start,
                    clipped_ms=int(round((trim_start - cue.start) * 1000)),
                )
            )
        if cue.start + tolerance < trim_end < cue.end - tolerance:
            collisions.append(
                TrimCollision(
                    side="end",
                    cue_index=idx,
                    text=cue.text,
                    cue_start=cue.start,
                    cue_end=cue.end,
                    cut_time=trim_end,
                    clipped_ms=int(round((cue.end - trim_end) * 1000)),
                )
            )
    return collisions


def shift_and_clip_cues(
    cues: List[Cue],
    trim_start: float,
    trim_end: float,
    min_duration: float = 0.005,
) -> List[Cue]:
    """
    Shifts cues onto the new [0, trim_end - trim_start] timeline, dropping cues
    outside the trimmed window or shorter than `min_duration`, and re-indexes 1..N.
    """
    if trim_end <= trim_start:
        raise ValueError(f"trim_end ({trim_end}) must be greater than trim_start ({trim_start})")

    new_dur = round(trim_end - trim_start, 3)
    shifted: List[Cue] = []
    for cue in cues:
        new_start = round(max(0.0, cue.start - trim_start), 3)
        new_end = round(min(new_dur, cue.end - trim_start), 3)
        if new_end - new_start >= min_duration:
            shifted.append(
                Cue(
                    index=len(shifted) + 1,
                    start=new_start,
                    end=new_end,
                    text=cue.text,
                )
            )
    return shifted


def _wav_duration(wav_path: Path) -> float:
    return round(float(sf.info(str(wav_path)).duration), 3)


def discover_clips(root: Path = Path(".")) -> List[ClipItem]:
    """
    Discovers the 39 incoming_clips + 133 benchmark/manifest.json clips.
    """
    items: List[ClipItem] = []

    incoming_dir = root / "incoming_clips"
    audio_dir = incoming_dir / "audio"
    drafts_dir = incoming_dir / "drafts"
    review_dir = incoming_dir / "review"

    if audio_dir.exists():
        for wav_path in sorted(audio_dir.glob("*.wav")):
            cid = wav_path.stem
            draft_srt = drafts_dir / f"{cid}.srt"
            review_srt = review_dir / f"{cid}.srt"
            srt_path = review_srt if review_srt.exists() else draft_srt
            mp4_path = review_dir / f"{cid}.mp4"
            items.append(
                ClipItem(
                    key=f"incoming:{cid}",
                    bucket="incoming",
                    id=cid,
                    wav_path=wav_path,
                    srt_path=srt_path,
                    duration=_wav_duration(wav_path),
                    video_path=mp4_path if mp4_path.exists() else None,
                    review_srt_path=review_srt,
                    review_wav_path=review_dir / f"{cid}.wav",
                )
            )

    bench_dir = root / "benchmark"
    manifest_path = bench_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("clips", []):
            cid = entry["id"]
            wav_path = bench_dir / "audio" / f"{cid}.wav"
            srt_path = bench_dir / "references" / f"{cid}.srt"
            if not wav_path.exists():
                continue
            items.append(
                ClipItem(
                    key=f"benchmark:{cid}",
                    bucket="benchmark",
                    id=cid,
                    wav_path=wav_path,
                    srt_path=srt_path,
                    duration=_wav_duration(wav_path),
                    video_path=None,
                )
            )

    return items


def load_trim_plan(plan_path: Path) -> Dict[str, Any]:
    if not plan_path.exists():
        return {"clips": {}}
    return json.loads(plan_path.read_text(encoding="utf-8"))


def save_trim_plan(plan_path: Path, data: Dict[str, Any]) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = plan_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(plan_path)

