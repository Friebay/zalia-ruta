from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from zalia_ruta.chunker import AudioChunk


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    score: Optional[float] = None

    def __repr__(self) -> str:
        return f"WordTimestamp('{self.word}', {self.start:.2f}s - {self.end:.2f}s)"


class LithuanianASR:
    """
    ASR transcription wrapper for Lithuanian Whisper models (default: kristijonas/paprika-whisper-lt-v3).
    Decodes audio chunks and produces global word-level timestamps.
    """

    def __init__(
        self,
        model_id: str = "kristijonas/paprika-whisper-lt-v3",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        print(f"[ASR] Loading model '{self.model_id}' on {self.device} ({self.torch_dtype})...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            language="lithuanian",
            task="transcribe",
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("[ASR] Model loaded and ready.")

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        time_offset: float = 0.0,
    ) -> List[WordTimestamp]:
        """
        Transcribes a single audio chunk and returns word timestamps with global time offset added.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Call pipeline with word-level timestamp generation
        result = self.pipe(
            {"raw": audio, "sampling_rate": sr},
            return_timestamps="word",
            generate_kwargs={
                "language": "lithuanian",
                "task": "transcribe",
                "condition_on_prev_tokens": False,
            },
        )

        words: List[WordTimestamp] = []
        raw_chunks = result.get("chunks", [])

        last_valid_end = time_offset
        for item in raw_chunks:
            word_str = item.get("text", "").strip()
            if not word_str:
                continue

            ts = item.get("timestamp", (None, None))
            raw_start, raw_end = ts[0], ts[1]

            # Handle edge case where Whisper fails to assign start or end
            if raw_start is None:
                start_sec = last_valid_end
            else:
                start_sec = time_offset + float(raw_start)

            if raw_end is None:
                end_sec = start_sec + 0.30  # default 300ms duration
            else:
                end_sec = time_offset + float(raw_end)

            # Ensure start <= end
            if end_sec < start_sec:
                end_sec = start_sec + 0.10

            last_valid_end = end_sec
            words.append(
                WordTimestamp(
                    word=word_str,
                    start=round(start_sec, 3),
                    end=round(end_sec, 3),
                )
            )

        # Refine timestamps using acoustic energy to carve out real silence and breath pauses
        refined_words = acoustic_refine_words(
            words=words,
            audio=audio,
            sr=sr,
            time_offset=time_offset,
        )

        return refined_words

    def transcribe_chunks(
        self,
        chunks: List[AudioChunk],
        sr: int = 16000,
    ) -> List[WordTimestamp]:
        """
        Transcribes a sequence of pause-aligned chunks and reconciles them into
        a single contiguous global word timeline.
        """
        all_words: List[WordTimestamp] = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            print(f"[ASR] Transcribing chunk {i + 1}/{total} ({chunk.start_sec:.2f}s -> {chunk.end_sec:.2f}s)...")
            chunk_words = self.transcribe_chunk(
                chunk.audio,
                sr=sr,
                time_offset=chunk.start_sec,
            )
            all_words.extend(chunk_words)

        return all_words


def acoustic_refine_words(
    words: List[WordTimestamp],
    audio: np.ndarray,
    sr: int = 16000,
    time_offset: float = 0.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    min_word_duration_s: float = 0.10,
    margin_s: float = 0.03,
) -> List[WordTimestamp]:
    """
    Trims unvoiced silence/breath periods from raw Whisper timestamps using RMS energy analysis.
    This prevents Whisper from stretching words across silent pauses and breathing breaks.
    """
    if not words or len(audio) == 0:
        return words

    frame_len = int((frame_ms / 1000.0) * sr)
    hop_len = int((hop_ms / 1000.0) * sr)
    refined: List[WordTimestamp] = []

    for wt in words:
        rel_start_s = max(0.0, wt.start - time_offset)
        rel_end_s = max(rel_start_s, wt.end - time_offset)

        start_samp = int(rel_start_s * sr)
        end_samp = min(len(audio), int(rel_end_s * sr))

        word_audio = audio[start_samp:end_samp]
        if len(word_audio) < frame_len:
            refined.append(wt)
            continue

        w_frames = [
            np.sqrt(np.mean(word_audio[i : i + frame_len] ** 2))
            for i in range(0, len(word_audio) - frame_len + 1, hop_len)
        ]
        if not w_frames:
            refined.append(wt)
            continue

        w_frames = np.array(w_frames)
        max_rms = float(np.max(w_frames))

        # Dynamic threshold: 12% of peak RMS or absolute floor of 0.030
        thresh = max(0.030, max_rms * 0.12)
        speech_idx = np.where(w_frames >= thresh)[0]

        if len(speech_idx) == 0:
            refined.append(wt)
            continue

        first_idx = speech_idx[0]
        last_idx = speech_idx[-1]

        rel_speech_start = (first_idx * hop_len) / sr
        rel_speech_end = (last_idx * hop_len + frame_len) / sr

        # Apply safety margin so soft consonant onsets/decays are preserved
        refined_start = max(wt.start, wt.start + rel_speech_start - margin_s)
        refined_end = min(wt.end, wt.start + rel_speech_end + margin_s)

        # Enforce minimum word duration
        if refined_end - refined_start < min_word_duration_s:
            refined_end = min(wt.end, refined_start + min_word_duration_s)

        refined.append(
            WordTimestamp(
                word=wt.word,
                start=round(refined_start, 3),
                end=round(refined_end, 3),
                score=wt.score,
            )
        )

    return refined
