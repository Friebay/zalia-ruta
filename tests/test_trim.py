import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from zalia_ruta.bench.srt_io import Cue, parse_srt, write_srt
from zalia_ruta.bench.trim import (
    discover_clips,
    find_trim_collisions,
    is_effective_trim,
    load_trim_plan,
    save_trim_plan,
    shift_and_clip_cues,
)
from zalia_ruta.bench.validate import validate_cues


def _write_silence_wav(path: Path, seconds: float, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(round(seconds * sr)), dtype=np.int16)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def test_shift_and_clip_cues():
    cues = [
        Cue(index=1, start=0.2, end=0.6, text="intro"),
        Cue(index=2, start=1.0, end=1.5, text="pirmas"),
        Cue(index=3, start=1.6, end=2.2, text="antras"),
        Cue(index=4, start=3.0, end=3.5, text="outro"),
    ]
    shifted = shift_and_clip_cues(cues, trim_start=0.8, trim_end=2.5)
    assert [(c.index, c.start, c.end, c.text) for c in shifted] == [
        (1, 0.2, 0.7, "pirmas"),
        (2, 0.8, 1.4, "antras"),
    ]
    assert validate_cues(shifted, max_duration=1.7) == []


def test_find_trim_collisions():
    cues = [
        Cue(index=1, start=1.0, end=1.5, text="zodis"),
        Cue(index=2, start=2.0, end=2.6, text="pabaiga"),
    ]
    # Within 20ms tolerance -> no collision
    assert find_trim_collisions(cues, trim_start=1.01, trim_end=2.59) == []

    # Cutting inside cue -> collision
    cols = find_trim_collisions(cues, trim_start=1.2, trim_end=2.3)
    assert len(cols) == 2
    assert cols[0].side == "start" and cols[0].clipped_ms == 200
    assert cols[1].side == "end" and cols[1].clipped_ms == 300


def test_discover_clips_ignores_unreferenced_benchmark_wavs(tmp_path):
    _write_silence_wav(tmp_path / "incoming_clips" / "audio" / "inc_1.wav", 3.0)
    write_srt([Cue(1, 0.5, 1.0, "old_draft")], tmp_path / "incoming_clips" / "drafts" / "inc_1.srt")
    write_srt([Cue(1, 0.5, 1.0, "edited_review")], tmp_path / "incoming_clips" / "review" / "inc_1.srt")

    _write_silence_wav(tmp_path / "incoming_clips" / "audio" / "inc_2.wav", 3.0)
    write_srt([Cue(1, 0.5, 1.0, "draft_only")], tmp_path / "incoming_clips" / "drafts" / "inc_2.srt")

    _write_silence_wav(tmp_path / "benchmark" / "audio" / "bm_1.wav", 4.0)
    _write_silence_wav(tmp_path / "benchmark" / "audio" / "extra_unlisted.wav", 4.0)
    write_srt([Cue(1, 0.5, 1.2, "rytas")], tmp_path / "benchmark" / "references" / "bm_1.srt")
    (tmp_path / "benchmark" / "manifest.json").write_text(
        json.dumps({"clips": [{"id": "bm_1"}]}), encoding="utf-8"
    )

    items = discover_clips(tmp_path)
    assert [i.key for i in items] == ["incoming:inc_1", "incoming:inc_2", "benchmark:bm_1"]
    assert items[0].srt_path.parent.name == "review"
    assert items[1].srt_path.parent.name == "drafts"


def test_apply_trims_workflow(tmp_path):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from apply_trims import apply_trims

    # Setup synthetic incoming + benchmark clips
    _write_silence_wav(tmp_path / "incoming_clips" / "audio" / "inc_1.wav", 4.0)
    (tmp_path / "incoming_clips" / "review").mkdir(parents=True, exist_ok=True)
    _write_silence_wav(tmp_path / "incoming_clips" / "review" / "inc_1.wav", 4.0)
    write_srt(
        [Cue(1, 0.1, 0.4, "old_cut"), Cue(2, 1.2, 2.0, "old_draft")],
        tmp_path / "incoming_clips" / "drafts" / "inc_1.srt",
    )
    write_srt(
        [Cue(1, 0.1, 0.4, "cut"), Cue(2, 1.2, 2.0, "edited_keep")],
        tmp_path / "incoming_clips" / "review" / "inc_1.srt",
    )

    _write_silence_wav(tmp_path / "benchmark" / "audio" / "bm_1.wav", 5.0)
    write_srt(
        [Cue(1, 1.0, 1.8, "zodis"), Cue(2, 4.2, 4.8, "galas")],
        tmp_path / "benchmark" / "references" / "bm_1.srt",
    )
    (tmp_path / "benchmark" / "manifest.json").write_text(
        json.dumps({"clips": [{"id": "bm_1"}]}), encoding="utf-8"
    )
    hyp_file = tmp_path / "benchmark" / "hyp" / "paprika_refined" / "bm_1.srt"
    hyp_file.parent.mkdir(parents=True, exist_ok=True)
    hyp_file.write_text("stale", encoding="utf-8")

    plan_path = tmp_path / "trim_plan.json"
    save_trim_plan(
        plan_path,
        {
            "clips": {
                "incoming:inc_1": {
                    "bucket": "incoming",
                    "id": "inc_1",
                    "duration": 4.0,
                    "start": 0.5,
                    "end": 3.5,
                    "status": "trim",
                },
                # Collision on bm_1 start=1.4 slices 'zodis' [1.0, 1.8]
                "benchmark:bm_1": {
                    "bucket": "benchmark",
                    "id": "bm_1",
                    "duration": 5.0,
                    "start": 1.4,
                    "end": 4.0,
                    "status": "trim",
                },
            }
        },
    )

    # 1. Dry-run should fail/skip benchmark collision and leave files intact
    rc = apply_trims(tmp_path, plan_path, dry_run=True)
    assert rc == 1
    assert abs(sf.info(str(tmp_path / "incoming_clips" / "audio" / "inc_1.wav")).duration - 4.0) < 0.01
    assert hyp_file.exists()

    # 2. Fix bm_1 trim to clean gap [0.8, 3.8]
    plan = load_trim_plan(plan_path)
    plan["clips"]["benchmark:bm_1"]["start"] = 0.8
    plan["clips"]["benchmark:bm_1"]["end"] = 3.8
    save_trim_plan(plan_path, plan)

    rc = apply_trims(tmp_path, plan_path, dry_run=False)
    assert rc == 0

    # Check trimmed durations
    assert abs(sf.info(str(tmp_path / "incoming_clips" / "audio" / "inc_1.wav")).duration - 3.0) < 0.02
    assert abs(sf.info(str(tmp_path / "benchmark" / "audio" / "bm_1.wav")).duration - 3.0) < 0.02

    # Check shifted SRTs (review SRT preserves edited text; raw draft untouched)
    inc_cues = parse_srt((tmp_path / "incoming_clips" / "review" / "inc_1.srt").read_text(encoding="utf-8"))
    assert len(inc_cues) == 1
    assert inc_cues[0].text == "edited_keep" and inc_cues[0].start == 0.7

    bm_cues = parse_srt((tmp_path / "benchmark" / "references" / "bm_1.srt").read_text(encoding="utf-8"))
    assert len(bm_cues) == 1
    assert bm_cues[0].text == "zodis" and bm_cues[0].start == 0.2 and bm_cues[0].end == 1.0

    # Stale hyp purged, backups exist, plan marked applied
    assert not hyp_file.exists()
    assert (tmp_path / ".trim_backups").exists()
    updated_plan = load_trim_plan(plan_path)
    assert updated_plan["clips"]["incoming:inc_1"]["status"] == "applied"
    assert updated_plan["clips"]["benchmark:bm_1"]["status"] == "applied"
    assert is_effective_trim(0.0, 3.0, 3.0) is False

