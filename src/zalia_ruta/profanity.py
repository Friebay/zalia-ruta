import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from zalia_ruta.asr import WordTimestamp


@dataclass
class BleepInterval:
    start_sec: float
    end_sec: float
    matched_word: str
    original_word: str

    def __repr__(self) -> str:
        return f"BleepInterval('{self.original_word}', {self.start_sec:.2f}s - {self.end_sec:.2f}s)"


# Default regex patterns for core Lithuanian / Slavic profanities and their inflections
DEFAULT_PROFANITY_PATTERNS = [
    # blet, blėt, bleit, bliat, bliad, buliot, ble, bliamba, bletis, blecha (does not match blefas, blynai, blokai, blitz)
    r"^(ble+[iy]?($|[tdch]|mb)|blia[tdchmb]|buliot).*$",
    # nachui, nahui, nachuj, pochui, nach, etc.
    r"^(na|po|za)[chx]+u?[ijy]?.*$",
    # suka, sukos, sukų, sukom (standalone inflections, does not match sukasi, sukau, sukti)
    r"^suk(a|os?|u|om|omis|yte|ele)$",
    # compound: sukanachu, sukanachui, sukanach
    r"^suk(a|os?|u|om)?(na|po)[chx]+.*$",
    # kurva, kurwa, kurvos, kurvų, etc.
    r"^kur[wv](a|os?|u|om|omis|yn.*|yt.*|el.*)$",
    # pist, pyst, nupiso, supistas, atsipisk, pyzda, chuj
    r"^.*(pis[tk]|pisa|piso|pys[tk]|pyzd|pizd|[chx]uj).*$",
    # jobt, joban, jeban, dalbajob, dolbajob
    r"^.*(job[atn]|jeb[atn]|dalbajob|dolbajob).*$",
    # bybys, bibis
    r"^b[iy]b([iy][rs]|u|a|k|el|o|ui).*$",
    # sikna, sikt, apsišikti, prišiko (does not match prisikviesti, prisikelti)
    r"^(sikn.*|sikt.*|apsisik.*|prisik(o|om|ot|us|ti|au|ai|es|usi|t))$",
    # debil
    r"^debil.*$",
]


def clean_token(token: str) -> str:
    """Removes punctuation, strips diacritics via NFKD normalization, and lowercases token."""
    token = unicodedata.normalize("NFKD", token.lower().strip())
    token = "".join(c for c in token if not unicodedata.combining(c))
    # Remove leading/trailing non-alphanumeric chars (punctuation, quotes, ellipses)
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
    return token


class ProfanityDetector:
    """
    Detects profanities in transcribed words using root and regex pattern matching,
    applies configurable safety padding, and merges adjacent overlapping intervals.
    """

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        custom_words_file: Optional[str | Path] = None,
        pad_start_s: float = 0.06,  # 60 ms padding before word onset
        pad_end_s: float = 0.06,    # 60 ms padding after word offset
        merge_gap_s: float = 0.15,   # merge intervals separated by < 150 ms
    ):
        self.pad_start_s = pad_start_s
        self.pad_end_s = pad_end_s
        self.merge_gap_s = merge_gap_s

        combined_patterns = list(DEFAULT_PROFANITY_PATTERNS if patterns is None else patterns)

        if custom_words_file is not None:
            custom_path = Path(custom_words_file)
            if custom_path.exists():
                with open(custom_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # If line contains regex meta-chars, treat as regex, else exact root match
                            if any(c in line for c in r"^$*+?[](){}|"):
                                combined_patterns.append(line)
                            else:
                                combined_patterns.append(rf"^{re.escape(line.lower())}.*$")

        self.compiled_regexes = [re.compile(p, re.IGNORECASE) for p in combined_patterns]

    def is_profanity(self, word: str) -> bool:
        """Checks if a cleaned word matches any profanity pattern."""
        cleaned = clean_token(word)
        if not cleaned:
            return False
        return any(rx.search(cleaned) for rx in self.compiled_regexes)

    def find_intervals(
        self,
        words: List[WordTimestamp],
        audio_duration_sec: Optional[float] = None,
    ) -> List[BleepInterval]:
        """
        Finds all profanity timestamps, adds padding, and merges adjacent intervals.
        """
        raw_intervals: List[BleepInterval] = []

        for wt in words:
            if self.is_profanity(wt.word):
                start = max(0.0, wt.start - self.pad_start_s)
                end = wt.end + self.pad_end_s
                if audio_duration_sec is not None:
                    end = min(audio_duration_sec, end)

                raw_intervals.append(
                    BleepInterval(
                        start_sec=round(start, 3),
                        end_sec=round(end, 3),
                        matched_word=clean_token(wt.word),
                        original_word=wt.word,
                    )
                )

        if not raw_intervals:
            return []

        # Merge adjacent or overlapping intervals
        merged: List[BleepInterval] = [raw_intervals[0]]
        for curr in raw_intervals[1:]:
            prev = merged[-1]
            if curr.start_sec <= prev.end_sec + self.merge_gap_s:
                # Overlap or close enough -> merge
                merged[-1] = BleepInterval(
                    start_sec=prev.start_sec,
                    end_sec=max(prev.end_sec, curr.end_sec),
                    matched_word=f"{prev.matched_word} {curr.matched_word}",
                    original_word=f"{prev.original_word} {curr.original_word}",
                )
            else:
                merged.append(curr)

        return merged
