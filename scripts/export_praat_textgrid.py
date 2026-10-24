import shutil
from pathlib import Path
import soundfile as sf

from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.bench.systems import BenchmarkRunner


def export_clip_to_textgrid(clip_id: str):
    bench_dir = Path("benchmark")
    runner = BenchmarkRunner(bench_dir)
    out_dir = Path("praat_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_wav = bench_dir / "audio" / f"{clip_id}.wav"
    dst_wav = out_dir / f"{clip_id}.wav"
    shutil.copy2(src_wav, dst_wav)

    info = sf.info(str(dst_wav))
    total_dur = info.duration

    ref_file = bench_dir / "references" / f"{clip_id}.srt"
    ref_cues = parse_srt(ref_file.read_text(encoding="utf-8")) if ref_file.exists() else []
    mms_words = runner.load_cached("paprika_refined", clip_id) or []
    ac_words = runner.load_cached("paprika_acoustic", clip_id) or []

    def build_tier_intervals(items, duration):
        intervals = []
        curr_t = 0.0
        for it in items:
            s = getattr(it, "start")
            e = getattr(it, "end")
            txt = getattr(it, "text", None) or getattr(it, "word", "")
            if s > curr_t:
                intervals.append((curr_t, s, ""))
            intervals.append((s, e, txt))
            curr_t = e
        if curr_t < duration:
            intervals.append((curr_t, duration, ""))
        return intervals

    tiers = []
    if ref_cues:
        tiers.append(("Etalonas", build_tier_intervals(ref_cues, total_dur)))
    if mms_words:
        tiers.append(("MMS_Refined", build_tier_intervals(mms_words, total_dur)))
    if ac_words:
        tiers.append(("Acoustic_Refined", build_tier_intervals(ac_words, total_dur)))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "0.0",
        f"{total_dur:.6f}",
        "<exists>",
        f"{len(tiers)}",
    ]

    for name, intervals in tiers:
        lines.extend([
            '"IntervalTier"',
            f'"{name}"',
            "0.0",
            f"{total_dur:.6f}",
            f"{len(intervals)}",
        ])
        for s, e, txt in intervals:
            lines.extend([
                f"{s:.6f}",
                f"{e:.6f}",
                f'"{txt}"',
            ])

    tg_path = out_dir / f"{clip_id}.TextGrid"
    tg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created Praat TextGrid: {tg_path} (paired with {dst_wav})")


if __name__ == "__main__":
    export_clip_to_textgrid("karalius_short_04")
