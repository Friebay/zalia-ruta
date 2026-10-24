import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf

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
            self._asr_models[model_id] = LithuanianASR(model_id, use_mms_align=False)
        return self._asr_models[model_id]

    def get_hyp_path(self, system_id: str, clip_id: str) -> Path:
        return self.hyp_dir / system_id / f"{clip_id}.json"

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
        words = asr.transcribe_chunk(audio, sr=sr, time_offset=0.0)
        self.save_cached(system_id, clip_id, words)
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
        refined = self._mms_aligner.align(audio, raw_words, sr=sr, time_offset=0.0)
        self.save_cached(system_id, clip_id, refined)
        return refined

    def run_model_acoustic(self, model_id: str, prefix: str, clip_id: str, audio: np.ndarray, sr: int = 16000, force: bool = False) -> List[WordTimestamp]:
        system_id = f"{prefix}_acoustic"
        if not force:
            cached = self.load_cached(system_id, clip_id)
            if cached is not None:
                return cached

        base_words = self.run_model_refined(model_id, prefix, clip_id, audio, sr=sr, force=force)
        refined = refine_word_boundaries(audio, base_words, sr=sr)
        self.save_cached(system_id, clip_id, refined)
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
