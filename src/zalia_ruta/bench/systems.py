import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
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


KNOWN_MODELS: Dict[str, str] = {
    "paprika": "kristijonas/paprika-whisper-lt-v3",
    "whisper_small_lt": "DeividasM/whisper-small-lt",
    "azuolas": "akisviete/azuolas-whisper-lt",
    "turbo": "openai/whisper-large-v3-turbo",
}


class BenchmarkRunner:
    def __init__(self, bench_dir: Path):
        self.bench_dir = Path(bench_dir)
        self.audio_dir = self.bench_dir / "audio"
        self.hyp_dir = self.bench_dir / "hyp"
        self.youtube_dir = self.bench_dir / "youtube"

        self._asr_models: Dict[str, LithuanianASR] = {}
        self._mms_aligner: Optional[MMSAligner] = None

    def get_asr(self, model_id: str) -> LithuanianASR:
        if model_id not in self._asr_models:
            asr = LithuanianASR(model_id, use_mms_align=False)
            # Warm up CUDA kernels so clip 1 does not absorb one-time JIT and allocation latency
            dummy_audio = np.zeros(8000, dtype=np.float32)
            try:
                asr.transcribe_chunk(dummy_audio, sr=16000, time_offset=0.0)
            except Exception:
                pass
            self._asr_models[model_id] = asr
        return self._asr_models[model_id]

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
                return cached

        asr = self.get_asr(model_id)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        words = asr.transcribe_chunk(audio, sr=sr, time_offset=0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        duration_s = float(len(audio) / sr)
        latency_s = float(t1 - t0)
        rtf = float(latency_s / duration_s) if duration_s > 0 else 0.0
        speedup = float(duration_s / latency_s) if latency_s > 0 else 0.0
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

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
