import logging
import re
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Suppress repetitive HuggingFace generation/tokenization warnings that corrupt progress bars
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from zalia_ruta.chunker import AudioChunk


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    score: Optional[float] = None

    def __repr__(self) -> str:
        return f"WordTimestamp('{self.word}', {self.start:.2f}s - {self.end:.2f}s)"


# Lithuanian diacritics mapping for MMS CTC dictionary
_LT_DIACRITICS = str.maketrans({
    "ą": "a", "Ą": "a",
    "č": "c", "Č": "c",
    "ę": "e", "Ę": "e",
    "ė": "e", "Ė": "e",
    "į": "i", "Į": "i",
    "š": "s", "Š": "s",
    "ų": "u", "Ų": "u",
    "ū": "u", "Ū": "u",
    "ž": "z", "Ž": "z",
})

_VALID_CHARS_RE = re.compile(r"[^a-z'\-]")


def clean_for_mms(word: str) -> str:
    """
    Normalizes a Lithuanian word into characters supported by the MMS CTC dictionary.
    Maps Lithuanian diacritics to basic Latin letters and strips punctuation.
    """
    text = word.translate(_LT_DIACRITICS).lower()
    return _VALID_CHARS_RE.sub("", text).strip("-'")


class MMSWordAligner:
    """
    Forced alignment using torchaudio MMS_FA pipeline.
    Aligns Lithuanian speech audio with word-level transcripts.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torchaudio.pipelines.MMS_FA
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.model = self.bundle.get_model(with_star=False).to(self.device)
        self.model.eval()

    def align(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        words: Union[List[WordTimestamp], List[str]],
        sr: int = 16000,
        time_offset: float = 0.0,
    ) -> List[WordTimestamp]:
        """
        Aligns words onto audio and returns WordTimestamp list with precision start and end times.
        If alignment fails or tokens cannot be extracted, falls back to original timestamps.
        """
        if not words or len(audio) == 0:
            return [
                w if isinstance(w, WordTimestamp) else WordTimestamp(word=str(w), start=round(time_offset, 3), end=round(time_offset + 0.3, 3))
                for w in words
            ]

        # Prepare initial copy with fallback WordTimestamp objects
        results: List[WordTimestamp] = []
        for w in words:
            if isinstance(w, WordTimestamp):
                results.append(WordTimestamp(word=w.word, start=w.start, end=w.end, score=w.score))
            else:
                results.append(WordTimestamp(word=str(w), start=round(time_offset, 3), end=round(time_offset + 0.3, 3)))

        valid_indices: List[int] = []
        cleaned_words: List[str] = []
        for idx, wt in enumerate(results):
            cw = clean_for_mms(wt.word)
            if cw:
                valid_indices.append(idx)
                cleaned_words.append(cw)

        if not cleaned_words:
            return results

        try:
            tokens = self.tokenizer(cleaned_words)

            if isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                if audio.ndim == 1:
                    wav_tensor = torch.from_numpy(audio).unsqueeze(0)
                elif audio.ndim == 2:
                    wav_tensor = torch.from_numpy(audio)
                    if wav_tensor.shape[0] > 1 and wav_tensor.shape[1] == 1:
                        wav_tensor = wav_tensor.transpose(0, 1)
                else:
                    return results
            else:
                wav_tensor = audio
                if wav_tensor.ndim == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)

            wav_tensor = wav_tensor.to(self.device)

            with torch.inference_mode():
                emission, _ = self.model(wav_tensor)

            emission = emission[0].cpu()  # (T, num_tokens)
            spans_list = self.aligner(emission, tokens)

            num_samples = wav_tensor.size(1)
            num_frames = emission.size(0)
            ratio = num_samples / num_frames

            for orig_idx, span in zip(valid_indices, spans_list):
                if not span:
                    continue

                start_frame = span[0].start
                end_frame = span[-1].end

                start_sec = time_offset + (start_frame * ratio) / sr
                end_sec = time_offset + (end_frame * ratio) / sr

                if end_sec <= start_sec:
                    end_sec = start_sec + 0.05

                results[orig_idx].start = round(float(start_sec), 3)
                results[orig_idx].end = round(float(end_sec), 3)

        except Exception as exc:
            print(f"[ASR] MMS alignment fallback triggered: {exc}")

        return results


class LithuanianASR:
    """
    ASR transcription wrapper for Lithuanian Whisper models (default: kristijonas/paprika-whisper-lt-v3)
    with MMS forced alignment for word-boundary precision.
    """

    def __init__(
        self,
        model_id: str = "kristijonas/paprika-whisper-lt-v3",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_mms_align: bool = True,
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

        self.use_mms_align = use_mms_align
        self._aligner: Optional[MMSWordAligner] = None

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

        if getattr(self.model.generation_config, "no_timestamps_token_id", None) is None:
            try:
                from transformers import GenerationConfig
                base_name = "openai/whisper-small"
                if "base" in self.model_id.lower():
                    base_name = "openai/whisper-base"
                elif "medium" in self.model_id.lower():
                    base_name = "openai/whisper-medium"
                elif "large" in self.model_id.lower():
                    base_name = "openai/whisper-large-v3"
                self.model.generation_config = GenerationConfig.from_pretrained(base_name)
            except Exception:
                pass

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("[ASR] Whisper model loaded.")

        if self.use_mms_align:
            print(f"[ASR] Initializing MMS Forced Aligner on {self.device}...")
            self._aligner = MMSWordAligner(device=self.device)
            print("[ASR] MMS Forced Aligner ready.")

    def align_words(
        self,
        audio: np.ndarray,
        words: List[WordTimestamp],
        sr: int = 16000,
        time_offset: float = 0.0,
    ) -> List[WordTimestamp]:
        """
        Aligns words using MMS Forced Alignment.
        """
        if not words:
            return words
        if self._aligner is None:
            self._aligner = MMSWordAligner(device=self.device)
        return self._aligner.align(audio, words, sr=sr, time_offset=time_offset)

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        time_offset: float = 0.0,
    ) -> List[WordTimestamp]:
        """
        Transcribes a single audio chunk and returns word timestamps with global time offset added.
        If use_mms_align is True, precision forced alignment is applied to the chunk.
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

            if raw_start is None:
                start_sec = last_valid_end
            else:
                start_sec = time_offset + float(raw_start)

            if raw_end is None:
                end_sec = start_sec + 0.30
            else:
                end_sec = time_offset + float(raw_end)

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

        # Align timestamps using MMS Forced Alignment if enabled
        if self.use_mms_align and words:
            words = self.align_words(
                audio=audio,
                words=words,
                sr=sr,
                time_offset=time_offset,
            )

        return words

    def transcribe_chunks(
        self,
        chunks: List[AudioChunk],
        sr: int = 16000,
    ) -> List[WordTimestamp]:
        """
        Transcribes a sequence of pause-aligned chunks and reconciles them into
        a single contiguous global word timeline with a live tqdm progress bar.
        """
        all_words: List[WordTimestamp] = []
        total = len(chunks)

        with tqdm(
            total=total,
            desc="[ASR] Transcribing",
            unit="chunk",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            for chunk in chunks:
                pbar.set_postfix({
                    "time": f"{chunk.start_sec:.1f}s-{chunk.end_sec:.1f}s",
                    "words": len(all_words),
                })
                chunk_words = self.transcribe_chunk(
                    chunk.audio,
                    sr=sr,
                    time_offset=chunk.start_sec,
                )
                all_words.extend(chunk_words)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                pbar.update(1)

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
    [DEPRECATED] Trims unvoiced silence/breath periods from raw Whisper timestamps using RMS energy analysis.
    Superceded by MMSWordAligner.
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
