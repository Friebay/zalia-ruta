import difflib
from pathlib import Path
from typing import List, Optional

from zalia_ruta.asr import WordTimestamp
from zalia_ruta.bench.score import normalize_word
from zalia_ruta.bench.srt_io import Cue, write_srt


def create_disagreement_draft(
    base_words: List[WordTimestamp],
    alt_words_list: List[List[WordTimestamp]],
    alt_labels: Optional[List[str]] = None,
) -> List[Cue]:
    """
    Creates word-per-cue draft from base_words (e.g. Paprika + MMS_FA).
    Compares against alternative hypotheses (e.g. Turbo, YouTube).
    Where normalized words differ, marks cue text as 'base|alt1[|alt2]'.
    """
    if not base_words:
        return []

    if alt_labels is None:
        alt_labels = [f"alt{i+1}" for i in range(len(alt_words_list))]

    cues: List[Cue] = [
        Cue(index=idx, start=w.start, end=w.end, text=w.word)
        for idx, w in enumerate(base_words, start=1)
    ]

    base_tokens = [normalize_word(w.word) for w in base_words]

    for alt_words, label in zip(alt_words_list, alt_labels):
        if not alt_words:
            continue

        alt_tokens = [normalize_word(w.word) for w in alt_words]
        matcher = difflib.SequenceMatcher(None, base_tokens, alt_tokens)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            elif tag == "replace":
                p_slice = base_words[i1:i2]
                a_slice = alt_words[j1:j2]
                if len(p_slice) == len(a_slice):
                    for idx_offset in range(len(p_slice)):
                        c_idx = i1 + idx_offset
                        orig_text = cues[c_idx].text
                        alt_text = a_slice[idx_offset].word
                        if "|" not in orig_text:
                            cues[c_idx].text = f"{orig_text}|{alt_text}"
                        elif alt_text not in orig_text.split("|"):
                            cues[c_idx].text = f"{orig_text}|{alt_text}"
                else:
                    alt_combined = " ".join([w.word for w in a_slice])
                    for idx_offset in range(len(p_slice)):
                        c_idx = i1 + idx_offset
                        orig_text = cues[c_idx].text
                        if idx_offset == 0:
                            if "|" not in orig_text:
                                cues[c_idx].text = f"{orig_text}|{alt_combined}"
                            elif alt_combined not in orig_text.split("|"):
                                cues[c_idx].text = f"{orig_text}|{alt_combined}"
                        else:
                            if "|" not in orig_text:
                                cues[c_idx].text = f"{orig_text}|-"

            elif tag == "delete":
                for c_idx in range(i1, i2):
                    orig_text = cues[c_idx].text
                    if "|" not in orig_text:
                        cues[c_idx].text = f"{orig_text}|-"

            elif tag == "insert":
                target_idx = max(0, min(len(cues) - 1, i1 - 1 if i1 > 0 else 0))
                inserted_str = " ".join([w.word for w in alt_words[j1:j2]])
                orig_text = cues[target_idx].text
                if f"+{inserted_str}" not in orig_text:
                    cues[target_idx].text = f"{orig_text}|+{inserted_str}"

    return cues


def write_draft_file(
    output_path: Path,
    base_words: List[WordTimestamp],
    alt_words_list: List[List[WordTimestamp]],
    alt_labels: Optional[List[str]] = None,
) -> Path:
    cues = create_disagreement_draft(base_words, alt_words_list, alt_labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(cues, output_path)
    return output_path
