import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
import click
import soundfile as sf

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from zalia_ruta.bench.draft import create_disagreement_draft
from zalia_ruta.bench.score import score_system
from zalia_ruta.bench.srt_io import parse_srt, write_srt
from zalia_ruta.bench.systems import BenchmarkRunner, KNOWN_MODELS, deserialize_words
from zalia_ruta.bench.validate import validate_reference_file
from zalia_ruta.bench.youtube import fetch_youtube_captions


BENCH_DIR = Path("benchmark")
MANIFEST_PATH = BENCH_DIR / "manifest.json"


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise click.ClickException(f"Manifest file not found at {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@click.group()
def main():
    """Lithuanian Word-Level ASR Benchmark Tool."""
    pass


@main.command()
@click.option("--clip", default=None, help="Process only a specific clip ID.")
def prepare(clip: Optional[str]):
    """Extract 16 kHz WAVs and download available YouTube captions."""
    manifest = load_manifest()
    clips = manifest.get("clips", [])

    audio_dir = BENCH_DIR / "audio"
    youtube_dir = BENCH_DIR / "youtube"
    audio_dir.mkdir(parents=True, exist_ok=True)
    youtube_dir.mkdir(parents=True, exist_ok=True)

    for c in clips:
        cid = c["id"]
        if clip and cid != clip:
            continue

        src_file = Path(c["file"])
        dst_wav = audio_dir / f"{cid}.wav"

        # 1. Extract 16 kHz mono WAV
        if not dst_wav.exists():
            if not src_file.exists():
                click.echo(f"[{cid}] Warning: Source video {src_file} not found, skipping WAV extract.")
            else:
                click.echo(f"[{cid}] Extracting 16 kHz audio -> {dst_wav}...")
                cmd = [
                    "ffmpeg", "-y", "-i", str(src_file),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(dst_wav)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            click.echo(f"[{cid}] Audio WAV already exists: {dst_wav}")

        # 2. Download YouTube captions if available
        yt_id = c.get("youtube_id")
        if yt_id:
            dst_json3 = youtube_dir / f"{cid}.lt-orig.json3"
            if not dst_json3.exists():
                click.echo(f"[{cid}] Fetching YouTube captions ({yt_id})...")
                fetch_youtube_captions(yt_id, dst_json3)
            else:
                click.echo(f"[{cid}] YouTube captions already exist: {dst_json3}")


@main.command()
@click.option("--system", default=None, help="Run only a specific system (e.g. paprika_acoustic, whisper_small_lt_acoustic).")
@click.option("--model", default=None, help="Run a specific model (e.g. whisper_small_lt, paprika, azuolas, turbo).")
@click.option("--clip", default=None, help="Process only a specific clip ID.")
@click.option("--force", is_flag=True, default=False, help="Overwrite cached hypotheses.")
def run(system: Optional[str], model: Optional[str], clip: Optional[str], force: bool):
    """Run ASR systems and cache word timestamps to benchmark/hyp/."""
    manifest = load_manifest()
    runner = BenchmarkRunner(BENCH_DIR)

    # Determine which model/system to run
    selected_model = model or ("whisper_small_lt" if system and "whisper_small" in system else None)

    for c in manifest.get("clips", []):
        cid = c["id"]
        if clip and cid != clip:
            continue

        wav_path = BENCH_DIR / "audio" / f"{cid}.wav"
        if not wav_path.exists():
            click.echo(f"[{cid}] Audio missing ({wav_path}), run 'zalia-bench prepare' first.")
            continue

        audio, sr = sf.read(str(wav_path))

        if selected_model:
            model_key = selected_model
            model_id = KNOWN_MODELS.get(model_key, model_key)
            prefix = model_key if model_key in KNOWN_MODELS else model_key.split("/")[-1].replace("-", "_")

            if system is None or system == f"{prefix}_refined":
                click.echo(f"[{cid}] Running {prefix}_refined...")
                runner.run_model_refined(model_id, prefix, cid, audio, sr=sr, force=force)

            if system is None or system == f"{prefix}_acoustic":
                click.echo(f"[{cid}] Running {prefix}_acoustic...")
                runner.run_model_acoustic(model_id, prefix, cid, audio, sr=sr, force=force)

            if system == f"{prefix}_raw":
                click.echo(f"[{cid}] Running {prefix}_raw...")
                runner.run_model_raw(model_id, prefix, cid, audio, sr=sr, force=force)
        else:
            # Default behavior for paprika
            if system is None or system == "paprika_refined":
                click.echo(f"[{cid}] Running paprika_refined...")
                runner.run_paprika_refined(cid, audio, sr=sr, force=force)

            if system is None or system == "paprika_acoustic":
                click.echo(f"[{cid}] Running paprika_acoustic...")
                runner.run_paprika_acoustic(cid, audio, sr=sr, force=force)

            if system == "paprika_raw":
                click.echo(f"[{cid}] Running paprika_raw...")
                runner.run_paprika_raw(cid, audio, sr=sr, force=force)

            if system == "youtube_auto":
                if c.get("youtube_id"):
                    click.echo(f"[{cid}] Running youtube_auto...")
                    runner.run_youtube_auto(cid)


@main.command()
@click.option("--clip", default=None, help="Generate draft for a specific clip ID.")
def draft(clip: Optional[str]):
    """Generate word-per-cue draft SRT marking disagreement against YouTube captions."""
    manifest = load_manifest()
    runner = BenchmarkRunner(BENCH_DIR)
    drafts_dir = BENCH_DIR / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    for c in manifest.get("clips", []):
        cid = c["id"]
        if clip and cid != clip:
            continue

        # Load Paprika + MMS_FA base words
        base_words = runner.load_cached("paprika_mmsfa", cid)
        if base_words is None:
            click.echo(f"[{cid}] paprika_mmsfa not cached. Run 'zalia-bench run --system paprika_mmsfa --clip {cid}' first.")
            continue

        # Check for alternative hypotheses (e.g. YouTube)
        alt_words_list = []
        alt_labels = []
        yt_words = runner.load_cached("youtube_auto", cid)
        if yt_words:
            alt_words_list.append(yt_words)
            alt_labels.append("youtube")

        cues = create_disagreement_draft(base_words, alt_words_list, alt_labels)
        out_path = drafts_dir / f"{cid}.draft.srt"
        write_srt(cues, out_path)
        click.echo(f"[{cid}] Draft generated -> {out_path} ({len(cues)} cues)")


@main.command()
@click.option("--clip", default=None, help="Validate reference for a specific clip ID.")
def validate(clip: Optional[str]):
    """Validate corrected reference SRT files in benchmark/references/."""
    manifest = load_manifest()
    refs_dir = BENCH_DIR / "references"
    audio_dir = BENCH_DIR / "audio"

    has_error = False
    for c in manifest.get("clips", []):
        cid = c["id"]
        if clip and cid != clip:
            continue

        ref_file = refs_dir / f"{cid}.srt"
        if not ref_file.exists():
            click.echo(f"[{cid}] No reference file found at {ref_file}")
            continue

        max_dur = None
        wav_file = audio_dir / f"{cid}.wav"
        if wav_file.exists():
            info = sf.info(str(wav_file))
            max_dur = info.duration

        errors = validate_reference_file(ref_file, max_duration=max_dur)
        if errors:
            has_error = True
            click.echo(f"[{cid}] INVALID ({len(errors)} errors):")
            for e in errors:
                click.echo(f"   - {e}")
        else:
            click.echo(f"[{cid}] VALID: {ref_file}")

    if has_error:
        raise click.ClickException("Validation failed for one or more reference files.")


@main.command()
@click.option("--system", default=None, help="Score only a specific system.")
@click.option("--model", default=None, help="Score only a specific model prefix (e.g. whisper_small_lt, paprika).")
@click.option("--clip", default=None, help="Score only a specific clip ID.")
@click.option("--all-systems", is_flag=True, default=False, help="Include legacy/experimental baselines (paprika_raw, youtube_auto).")
def score(system: Optional[str], model: Optional[str], clip: Optional[str], all_systems: bool):
    """Score systems against validated references and generate reports/results.md."""
    manifest = load_manifest()
    runner = BenchmarkRunner(BENCH_DIR)
    refs_dir = BENCH_DIR / "references"
    reports_dir = BENCH_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if system:
        systems = [system]
    elif model:
        systems = [f"{model}_acoustic", f"{model}_refined"]
    elif all_systems:
        systems = ["paprika_acoustic", "paprika_refined", "whisper_small_lt_acoustic", "whisper_small_lt_refined", "paprika_raw", "youtube_auto"]
    else:
        # Default active production and acoustic refinement systems
        systems = ["paprika_acoustic"]
        if (runner.hyp_dir / "azuolas_acoustic").exists():
            systems.append("azuolas_acoustic")
        if (runner.hyp_dir / "whisper_small_lt_acoustic").exists():
            systems.append("whisper_small_lt_acoustic")
        systems.append("paprika_refined")
        if (runner.hyp_dir / "azuolas_refined").exists():
            systems.append("azuolas_refined")
        if (runner.hyp_dir / "whisper_small_lt_refined").exists():
            systems.append("whisper_small_lt_refined")
    results = []

    for c in manifest.get("clips", []):
        cid = c["id"]
        if clip and cid != clip:
            continue
        ref_file = refs_dir / f"{cid}.srt"
        if not ref_file.exists():
            continue

        ref_cues = parse_srt(ref_file.read_text(encoding="utf-8"))

        for sys_id in systems:
            hyp_words = runner.load_cached(sys_id, cid)
            if hyp_words is None:
                continue

            has_end = (sys_id != "youtube_auto")
            res = score_system(sys_id, cid, ref_cues, hyp_words, has_end_times=has_end)
            results.append(res)

    if not results:
        click.echo("No reference files or cached hypotheses found to score.")
        return

    # Compute aggregate macro averages per system
    system_aggregates = {}
    for sys_id in systems:
        sys_res = [r for r in results if r.system_id == sys_id]
        if not sys_res:
            continue
        mean_wer = sum(r.wer for r in sys_res) / len(sys_res)
        mean_cer = sum(r.cer for r in sys_res) / len(sys_res)
        timing_list = [r.timing for r in sys_res if r.timing]
        within_50 = sum(t.within_50ms_pct for t in timing_list) / len(timing_list) if timing_list else 0.0
        bleed = sum(t.pause_bleed_pct for t in timing_list) / len(timing_list) if timing_list else 0.0
        system_aggregates[sys_id] = {
            "clips": len(sys_res),
            "wer": mean_wer,
            "cer": mean_cer,
            "within_50": within_50,
            "bleed": bleed,
        }

    # Print Executive Summary table
    click.echo("\n============================== Benchmark Executive Summary ==============================")
    sum_headers = ["System", "Clips", "Macro WER%", "Macro CER%", "≤50ms Precision", "Pause Bleed%"]
    click.echo("{:<26} {:>6} {:>12} {:>12} {:>17} {:>14}".format(*sum_headers))
    click.echo("-" * 92)
    for sys_id, agg in system_aggregates.items():
        click.echo("{:<26} {:>6} {:>11.2f}% {:>11.2f}% {:>16.1f}% {:>13.1f}%".format(
            sys_id, agg["clips"], agg["wer"], agg["cer"], agg["within_50"], agg["bleed"]
        ))
    click.echo("-" * 92)

    # Print Detailed Per-Clip Table
    click.echo("\nDetailed Per-Clip Results:")
    headers = ["System", "Clip", "WER%", "CER%", "Start Med (ms)", "End Med (ms)", "≤50ms%", "End Bias (ms)", "Pause Bleed%"]
    click.echo("{:<26} {:<20} {:>6} {:>6} {:>15} {:>13} {:>8} {:>14} {:>13}".format(*headers))
    click.echo("-" * 128)

    report_lines = [
        "# Lithuanian ASR & Word-Boundary Benchmark Results\n",
        "## Executive Summary\n",
        f"| {' | '.join(sum_headers)} |",
        f"| {' | '.join(['---'] * len(sum_headers))} |",
    ]
    for sys_id, agg in system_aggregates.items():
        report_lines.append(
            f"| {sys_id} | {agg['clips']} | {agg['wer']:.2f}% | {agg['cer']:.2f}% | {agg['within_50']:.1f}% | {agg['bleed']:.1f}% |"
        )
    report_lines.append("\n## Detailed Per-Clip Results\n")
    report_lines.append(f"| {' | '.join(headers)} |")
    report_lines.append(f"| {' | '.join(['---'] * len(headers))} |")

    for r in results:
        t = r.timing
        if t:
            row = [
                r.system_id,
                r.clip_id,
                f"{r.wer:.1f}",
                f"{r.cer:.1f}",
                f"{t.start_med_ms:.1f}",
                f"{t.end_med_ms:.1f}" if r.system_id != "youtube_auto" else "N/A",
                f"{t.within_50ms_pct:.1f}",
                f"{t.end_bias_ms:+.1f}" if r.system_id != "youtube_auto" else "N/A",
                f"{t.pause_bleed_pct:.1f}%" if r.system_id != "youtube_auto" else "N/A",
            ]
        else:
            row = [r.system_id, r.clip_id, f"{r.wer:.1f}", f"{r.cer:.1f}", "N/A", "N/A", "N/A", "N/A", "N/A"]

        click.echo("{:<26} {:<20} {:>6} {:>6} {:>15} {:>13} {:>8} {:>14} {:>13}".format(*row))
        report_lines.append(f"| {' | '.join(row)} |")

    report_file = reports_dir / "results.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    click.echo(f"\nReport written to {report_file}")


@main.command(name="burn-compare")
@click.option("--clip", required=True, help="Clip ID to burn comparison video for.")
@click.option("--output", "-o", default=None, help="Output MP4 file path.")
@click.option("--no-ref", is_flag=True, default=False, help="Do not include Ground Truth reference track.")
def burn_compare_cmd(clip: str, output: Optional[str], no_ref: bool):
    """Burn dual subtitles (paprika_refined vs paprika_acoustic) onto video."""
    from zalia_ruta.bench.burn import burn_comparison_video

    manifest = load_manifest()
    runner = BenchmarkRunner(BENCH_DIR)

    clip_info = next((c for c in manifest.get("clips", []) if c["id"] == clip), None)
    if not clip_info:
        raise click.ClickException(f"Clip '{clip}' not found in manifest.")

    src_video = Path(clip_info["file"])
    if not src_video.exists():
        raise click.ClickException(f"Source video {src_video} does not exist.")

    refined_words = runner.load_cached("paprika_refined", clip)
    acoustic_words = runner.load_cached("paprika_acoustic", clip)

    if not refined_words or not acoustic_words:
        raise click.ClickException(f"Missing cached hypotheses for '{clip}'. Run 'zalia-bench run --clip {clip}' first.")

    ref_cues = None
    if not no_ref:
        ref_path = BENCH_DIR / "references" / f"{clip}.srt"
        if ref_path.exists():
            ref_cues = parse_srt(ref_path.read_text(encoding="utf-8"))

    if output:
        out_path = Path(output)
    else:
        out_path = BENCH_DIR / "comparison_videos" / f"{clip}_compare.mp4"

    click.echo(f"Burning comparison video for '{clip}' -> {out_path}...")
    burn_comparison_video(src_video, out_path, refined_words, acoustic_words, ref_cues=ref_cues)
    click.echo(f"Done! Saved to {out_path}")


if __name__ == "__main__":
    main()
