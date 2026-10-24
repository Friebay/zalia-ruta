from zalia_ruta.asr import MMSWordAligner, WordTimestamp, _LT_DIACRITICS, clean_for_mms

# Backward compatibility alias
MMSAligner = MMSWordAligner

__all__ = ["clean_for_mms", "MMSWordAligner", "MMSAligner", "WordTimestamp", "_LT_DIACRITICS"]
