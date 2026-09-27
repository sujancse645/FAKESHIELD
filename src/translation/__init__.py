"""translation - FakeShield Phase 6 Multilingual Translation Layer.

Provides IndicTrans2-based translation into English pivot representation.
"""

from src.translation.translation_config import (
    TranslationConfig,
    LANGUAGE_MAPPING,
    TARGET_LANGUAGE,
    TARGET_TAG,
)
from src.translation.translator import (
    Translator,
    TranslationResult,
    translate_to_english,
)
from src.translation.translation_quality import TranslationQualityAnalyzer

__all__ = [
    "TranslationConfig",
    "LANGUAGE_MAPPING",
    "TARGET_LANGUAGE",
    "TARGET_TAG",
    "Translator",
    "TranslationResult",
    "translate_to_english",
    "TranslationQualityAnalyzer",
]
