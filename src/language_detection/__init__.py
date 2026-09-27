"""FakeShield Language Detection Package.

Provides language identification for news text and OCR outputs
prior to downstream machine translation.
"""

from .language_detector import (
    LanguageDetector,
    detect_language,
    SUPPORTED_LANGUAGES,
    DetectionResult,
)
from .language_quality import LanguageQualityAnalyzer

__all__ = [
    "LanguageDetector",
    "detect_language",
    "SUPPORTED_LANGUAGES",
    "DetectionResult",
    "LanguageQualityAnalyzer",
]
