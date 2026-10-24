import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from zalia_ruta.asr import WordTimestamp
from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.highlights import export_highlight_clip, find_profanity_highlight_windows
from zalia_ruta.profanity import ProfanityDetector

def main():
    base_dir = Path("output/Tapinas apie nesusitvarkymą su dronais, blogiausią finansų ministrą ir riaušes ｜ Karalius Reaguoja_dSObkwfw2MA")
    video_path = Path("original_videos/Tapinas apie nesusitvarkymą su dronais, blogiausią finansų ministrą ir riaušes ｜ Karalius Reaguoja_dSObkwfw2MA.mp4")
    
    # Locate full SRT
    srt_files = list(base_dir.glob("*.srt"))
    if not srt_files:
        raise FileNotFoundError(f"No SRT found in {base_dir}")
    srt_path = srt_files[0]
    print(f"Reading full transcript from: {srt_path}")
    
    cues = parse_srt(srt_path.read_text(encoding="utf-8"))
    words = [WordTimestamp(word=c.text.strip(), start=c.start, end=c.end) for c in cues if c.text.strip()]
    print(f"Loaded {len(words)} words.")
    
    detector = ProfanityDetector()
    top_k = 10
    print(f"Finding Top {top_k} ~20-30s pause-aligned highlight windows...")
    windows = find_profanity_highlight_windows(
        words=words,
        detector=detector,
        top_k=top_k,
        target_duration_sec=25.0,
        min_duration_sec=18.0,
        max_duration_sec=32.0,
        min_gap_between_clips_sec=5.0,
    )
    
    out_dir = base_dir / "highlights_20s"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(windows)} clips to {out_dir}...")
    
    report_lines = [
        f"# Profanity Highlights Report (20-30s Clips): {video_path.name}",
        f"- Target duration: ~25s (18-32s pause boundaries)",
        f"- Clips extracted: {len(windows)}\n",
        "| Rank | Start | End | Duration | Profanities | Profane Words |",
        "|------|-------|-----|----------|-------------|---------------|",
    ]
    
    for win in windows:
        print(f"  -> Clip #{win.rank:02d}: [{win.start_sec:.1f}s - {win.end_sec:.1f}s] ({win.duration_sec:.1f}s) - {win.profanity_count} profanities: {', '.join(win.profane_words)}")
        mp4_p, wav_p, srt_p = export_highlight_clip(
            input_media_path=video_path,
            window=win,
            words=words,
            output_dir=out_dir,
            clip_prefix=f"karalius_short",
        )
        words_summary = ", ".join(win.profane_words[:8])
        if len(win.profane_words) > 8:
            words_summary += f" (+{len(win.profane_words)-8} more)"
        report_lines.append(
            f"| {win.rank} | {win.start_sec:.2f}s | {win.end_sec:.2f}s | {win.duration_sec:.2f}s | {win.profanity_count} | {words_summary} |"
        )
        
    report_path = out_dir / "report_20s.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nDone! Report written to {report_path}")

if __name__ == "__main__":
    main()
