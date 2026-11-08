import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import click
import numpy as np
import soundfile as sf
import torch

from zalia_ruta.acoustic_refiner import refine_word_boundaries
from zalia_ruta.asr import LithuanianASR, WordTimestamp, acoustic_refine_words
from zalia_ruta.bench.mms_align import MMSAligner
from zalia_ruta.bench.youtube import parse_youtube_json3


def serialize_words(words: List[WordTimestamp]) -> List[Dict[str, Any]]:
    return [
        {
            "word": w.word,
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "score": w.score,
        }
        for w in words
    ]


def deserialize_words(data: List[Dict[str, Any]]) -> List[WordTimestamp]:
    return [
        WordTimestamp(
            word=d["word"],
            start=float(d["start"]),
            end=float(d["end"]),
            score=d.get("score"),
        )
        for d in data
    ]


CT2_LOCAL_PATH = Path("models/paprika-whisper-lt-v3-ct2-int8_float16")
CT2_MODEL_ID = str(CT2_LOCAL_PATH) if CT2_LOCAL_PATH.exists() else "RobertasTa/paprika-whisper-lt-v3-ct2-int8"

CT2_FP16_LOCAL_PATH = Path("models/paprika-whisper-lt-v3-ct2-float16")
CT2_FP16_MODEL_ID = str(CT2_FP16_LOCAL_PATH)

AZUOLAS_CT2_PATH = Path("models/azuolas-whisper-lt-ct2-int8_float16")
AZUOLAS_MODEL_ID = str(AZUOLAS_CT2_PATH) if AZUOLAS_CT2_PATH.exists() else "akisviete/azuolas-whisper-lt"

SVOGUNAS_CT2_PATH = Path("models/svogunas-turbo-lt-ct2-int8_float16")
SVOGUNAS_MODEL_ID = str(SVOGUNAS_CT2_PATH) if SVOGUNAS_CT2_PATH.exists() else "svogunas/whisper-large-v3-turbo-lt"

KNOWN_MODELS: Dict[str, str] = {
    "paprika": "kristijonas/paprika-whisper-lt-v3",
    "paprika_prompt": "kristijonas/paprika-whisper-lt-v3",
    "paprika_ct2": CT2_MODEL_ID,
    "paprika_ct2_prompt": CT2_MODEL_ID,
    "paprika_ct2_fp16": CT2_FP16_MODEL_ID,
    "whisper_small_lt": "DeividasM/whisper-small-lt",
    "azuolas": AZUOLAS_MODEL_ID,
    "azuolas_ct2": AZUOLAS_MODEL_ID,
    "azuolas_hf": "akisviete/azuolas-whisper-lt",
    "turbo": "openai/whisper-large-v3-turbo",
    "svogunas": SVOGUNAS_MODEL_ID,
    "svogunas_ct2": SVOGUNAS_MODEL_ID,
    "lt_ausys": "RobertasTa/lt-ausys",
}


class LtAusysASR:
    """Adapter for ONNX-ASR Parakeet-TDT (RobertasTa/lt-ausys)."""

    def __init__(self, model_id: str = "RobertasTa/lt-ausys"):
        try:
            import onnx_asr
        except ImportError as e:
            raise ImportError(
                "onnx-asr is required for lt_ausys. "
                "Install it via `uv add --optional bench onnx-asr`."
            ) from e
        self.model = onnx_asr.load_model(model_id, quantization="int8")
        self.window_step = getattr(self.model.asr, "window_step", 0.01)
        self.subsampling_factor = getattr(self.model.asr, "_subsampling_factor", 8)
        self.token_dur = self.window_step * self.subsampling_factor

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        time_offset: float = 0.0,
        initial_prompt: Optional[str] = None,
    ) -> List[WordTimestamp]:
        import re

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        res = self.model.with_timestamps().recognize(audio, sample_rate=sr)
        if not getattr(res, "tokens", None) or not getattr(res, "timestamps", None):
            return []

        words: List[WordTimestamp] = []
        curr_tokens: List[str] = []
        curr_start: Optional[float] = None
        curr_end: Optional[float] = None

        for tok, ts in zip(res.tokens, res.timestamps):
            clean_tok = tok.strip()
            is_punc = bool(re.match(r"^[^\w\s]+$", clean_tok))
            is_new_word = tok.startswith(" ") or (curr_tokens and is_punc)

            if is_new_word and curr_tokens:
                w_clean = re.sub(r"[^\w]", "", "".join(curr_tokens).strip())
                if w_clean and curr_start is not None and curr_end is not None:
                    words.append(
                        WordTimestamp(
                            word=w_clean,
                            start=round(time_offset + curr_start, 3),
                            end=round(time_offset + curr_end, 3),
                        )
                    )
                curr_tokens = []
                curr_start = None
                curr_end = None

            if not is_punc:
                if curr_start is None:
                    curr_start = ts
                curr_end = ts + self.token_dur
                curr_tokens.append(clean_tok)

        if curr_tokens and curr_start is not None and curr_end is not None:
            w_clean = re.sub(r"[^\w]", "", "".join(curr_tokens).strip())
            if w_clean:
                words.append(
                    WordTimestamp(
                        word=w_clean,
                        start=round(time_offset + curr_start, 3),
                        end=round(time_offset + curr_end, 3),
                    )
                )

        return words


class BenchmarkRunner:
    def __init__(self, bench_dir: Path):
        self.bench_dir = Path(bench_dir)
        self.audio_dir = self.bench_dir / "audio"
        self.hyp_dir = self.bench_dir / "hyp"
        self.youtube_dir = self.bench_dir / "youtube"

        self._asr_models: Dict[str, Any] = {}
        self._mms_aligner: Optional[MMSAligner] = None

    def get_asr(self, model_id: str, initial_prompt: Optional[str] = None) -> Any:
        cache_key = f"{model_id}::prompt={initial_prompt}"
        if cache_key not in self._asr_models:
            if "lt-ausys" in model_id.lower() or "lt_ausys" in model_id.lower():
                asr: Any = LtAusysASR(model_id)
            else:
                asr = LithuanianASR(model_id, use_mms_align=False, initial_prompt=initial_prompt)
                # Warm up CUDA kernels so clip 1 does not absorb one-time JIT and allocation latency
                dummy_audio = np.zeros(8000, dtype=np.float32)
                try:
                    asr.transcribe_chunk(dummy_audio, sr=16000, time_offset=0.0)
                except Exception:
                    pass
            self._asr_models[cache_key] = asr
        return self._asr_models[cache_key]

    def get_hyp_path(self, system_id: str, clip_id: str) -> Path:
        return self.hyp_dir / system_id / f"{clip_id}.json"

    def get_metrics_path(self, system_id: str) -> Path:
        return self.hyp_dir / system_id / "_metrics.json"

    def load_metrics(self, system_id: str) -> Dict[str, dict]:
        p = self.get_metrics_path(system_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_metric(self, system_id: str, clip_id: str, metric: dict) -> None:
        metrics = self.load_metrics(system_id)
        metrics[clip_id] = metric
        p = self.get_metrics_path(system_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def is_cached(self, system_id: str, clip_id: str) -> bool:
        return self.get_hyp_path(system_id, clip_id).exists()

    def load_cached(self, system_id: str, clip_id: str) -> Optional[List[WordTimestamp]]:
        p = self.get_hyp_path(system_id, clip_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return deserialize_words(data)

    def save_cached(self, system_id: str, clip_id: str, words: List[WordTimestamp]) -> None:
        p = self.get_hyp_path(system_id, clip_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(serialize_words(words), indent=2, ensure_ascii=False), encoding="utf-8")

    def run_model_raw(self, model_id: str, prefix: str, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        system_id = f"{prefix}_raw"
        if not force:
            cached = self.load_cached(system_id, clip_id)
            if cached is not None:
                m = self.load_metrics(system_id).get(clip_id, {})
                if m:
                    click.echo(f"  [{clip_id}] {system_id} (cached): {m.get('latency_s', 0):.2f}s (audio: {m.get('duration_s', 0):.2f}s | RTF: {m.get('rtf', 0):.3f} | {m.get('speedup', 0)}x speed)")
                return cached

        from zalia_ruta.asr import DEFAULT_INITIAL_PROMPT
        eff_prompt = DEFAULT_INITIAL_PROMPT if "prompt" in prefix else None
        asr = self.get_asr(model_id, initial_prompt=eff_prompt)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        words = asr.transcribe_chunk(audio, sr=sr, time_offset=0.0, initial_prompt=eff_prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        duration_s = float(len(audio) / sr)
        latency_s = float(t1 - t0)
        rtf = float(latency_s / duration_s) if duration_s > 0 else 0.0
        speedup = float(duration_s / latency_s) if latency_s > 0 else 0.0
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        click.echo(f"  [{clip_id}] {system_id}: {latency_s:.2f}s (audio: {duration_s:.2f}s | RTF: {rtf:.3f} | {speedup:.1f}x speed)")

        self.save_cached(system_id, clip_id, words)
        self.save_metric(system_id, clip_id, {
            "duration_s": round(duration_s, 3),
            "latency_s": round(latency_s, 4),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "stage": "raw",
        })
        return words

    def run_model_refined(self, model_id: str, prefix: str, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        system_id = f"{prefix}_refined"
        if not force:
            cached = self.load_cached(system_id, clip_id)
            if cached is not None:
                m = self.load_metrics(system_id).get(clip_id, {})
                if m:
                    click.echo(f"  [{clip_id}] {system_id} (cached): total {m.get('latency_s', 0):.2f}s (audio: {m.get('duration_s', 0):.2f}s | RTF: {m.get('rtf', 0):.3f} | {m.get('speedup', 0)}x speed)")
                return cached

        raw_words = self.run_model_raw(model_id, prefix, clip_id, audio, sr=sr, force=force)
        if self._mms_aligner is None:
            self._mms_aligner = MMSAligner()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        refined = self._mms_aligner.align(audio, raw_words, sr=sr, time_offset=0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        duration_s = float(len(audio) / sr)
        stage_latency_s = float(t1 - t0)

        raw_metrics = self.load_metrics(f"{prefix}_raw").get(clip_id, {})
        raw_latency = raw_metrics.get("latency_s", 0.0)
        total_latency_s = raw_latency + stage_latency_s

        rtf = float(total_latency_s / duration_s) if duration_s > 0 else 0.0
        speedup = float(duration_s / total_latency_s) if total_latency_s > 0 else 0.0
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        click.echo(f"  [{clip_id}] {system_id}: total {total_latency_s:.2f}s (align {stage_latency_s:.2f}s | audio {duration_s:.2f}s | RTF: {rtf:.3f} | {speedup:.1f}x speed)")

        self.save_cached(system_id, clip_id, refined)
        self.save_metric(system_id, clip_id, {
            "duration_s": round(duration_s, 3),
            "latency_s": round(total_latency_s, 4),
            "stage_latency_s": round(stage_latency_s, 4),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "stage": "refined",
        })
        return refined

    def run_model_acoustic(self, model_id: str, prefix: str, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        system_id = f"{prefix}_acoustic"
        if not force:
            cached = self.load_cached(system_id, clip_id)
            if cached is not None:
                m = self.load_metrics(system_id).get(clip_id, {})
                if m:
                    click.echo(f"  [{clip_id}] {system_id} (cached): total {m.get('latency_s', 0):.2f}s (audio: {m.get('duration_s', 0):.2f}s | RTF: {m.get('rtf', 0):.3f} | {m.get('speedup', 0)}x speed)")
                return cached

        base_words = self.run_model_refined(model_id, prefix, clip_id, audio, sr=sr, force=force)

        t0 = time.perf_counter()
        refined = refine_word_boundaries(audio, base_words, sr=sr)
        t1 = time.perf_counter()

        duration_s = float(len(audio) / sr)
        stage_latency_s = float(t1 - t0)

        refined_metrics = self.load_metrics(f"{prefix}_refined").get(clip_id, {})
        refined_latency = refined_metrics.get("latency_s", 0.0)
        total_latency_s = refined_latency + stage_latency_s

        rtf = float(total_latency_s / duration_s) if duration_s > 0 else 0.0
        speedup = float(duration_s / total_latency_s) if total_latency_s > 0 else 0.0
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        click.echo(f"  [{clip_id}] {system_id}: total {total_latency_s:.2f}s (boundary {stage_latency_s:.2f}s | audio {duration_s:.2f}s | RTF: {rtf:.3f} | {speedup:.1f}x speed)")

        self.save_cached(system_id, clip_id, refined)
        self.save_metric(system_id, clip_id, {
            "duration_s": round(duration_s, 3),
            "latency_s": round(total_latency_s, 4),
            "stage_latency_s": round(stage_latency_s, 4),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "stage": "acoustic",
        })
        return refined

    def run_paprika_raw(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_raw(KNOWN_MODELS["paprika"], "paprika", clip_id, audio, sr=sr, force=force)

    def run_paprika_refined(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_refined(KNOWN_MODELS["paprika"], "paprika", clip_id, audio, sr=sr, force=force)

    def run_paprika_mmsfa(self, clip_id: str, audio: np.ndarray, sr: int = 16000) -> List[WordTimestamp]:
        """Backward compatibility alias: paprika_mmsfa is identical to paprika_refined."""
        return self.run_paprika_refined(clip_id, audio, sr=sr)

    def run_paprika_acoustic(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        """Acoustically refined word boundaries over MMS forced alignment."""
        return self.run_model_acoustic(KNOWN_MODELS["paprika"], "paprika", clip_id, audio, sr=sr, force=force)

    def run_whisper_small_raw(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_raw(KNOWN_MODELS["whisper_small_lt"], "whisper_small_lt", clip_id, audio, sr=sr, force=force)

    def run_whisper_small_refined(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_refined(KNOWN_MODELS["whisper_small_lt"], "whisper_small_lt", clip_id, audio, sr=sr, force=force)

    def run_whisper_small_acoustic(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_acoustic(KNOWN_MODELS["whisper_small_lt"], "whisper_small_lt", clip_id, audio, sr=sr, force=force)

    def run_paprika_prompt_raw(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_raw(KNOWN_MODELS["paprika"], "paprika_prompt", clip_id, audio, sr=sr, force=force)

    def run_paprika_prompt_refined(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_refined(KNOWN_MODELS["paprika"], "paprika_prompt", clip_id, audio, sr=sr, force=force)

    def run_paprika_prompt_acoustic(self, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        return self.run_model_acoustic(KNOWN_MODELS["paprika"], "paprika_prompt", clip_id, audio, sr=sr, force=force)

    def run_youtube_auto(self, clip_id: str) -> Optional[List[WordTimestamp]]:
        cached = self.load_cached("youtube_auto", clip_id)
        if cached is not None:
            return cached

        json3_path = self.youtube_dir / f"{clip_id}.lt-orig.json3"
        if not json3_path.exists():
            return None

        words = parse_youtube_json3(json3_path)
        self.save_cached("youtube_auto", clip_id, words)
        return words
