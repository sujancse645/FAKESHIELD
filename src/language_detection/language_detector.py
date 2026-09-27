"""language_detector.py - FakeShield Language Identification Module.

Uses langdetect with pinned seed for deterministic identification of
English and Indic languages from raw or OCR-extracted text.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Any

from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Pin seed to guarantee 100% deterministic detection
DetectorFactory.seed = 0

# Comprehensive ISO 639-1 language mapping for FakeShield target languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}

# Minimum number of linguistic (alphabetical/script) characters required
MIN_LINGUISTIC_CHARS = 3


@dataclass
class DetectionResult:
    """Structured result of language detection."""
    sample_id: Optional[str]
    raw_text: str
    normalized_text: str
    language_code: Optional[str]
    language_name: Optional[str]
    confidence: Optional[float]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "language_code": self.language_code,
            "language_name": self.language_name,
            "confidence": self.confidence,
            "status": self.status,
        }


class LanguageDetector:
    """FakeShield Language Detector.

    Normalizes input text non-destructively, filters out non-linguistic inputs
    (empty, digits, punctuation, extreme short text), and predicts language
    using pinned deterministic langdetect profiles.
    """

    def __init__(self, min_chars: int = MIN_LINGUISTIC_CHARS):
        self.min_chars = min_chars

    def normalize_text(self, text: str) -> str:
        """Apply non-destructive Unicode NFKC normalization and whitespace collapsing."""
        if not text:
            return ""
        # Unicode normalization (handles decomposing accents/ligatures)
        norm = unicodedata.normalize("NFKC", text)
        # Collapse multiple whitespace characters to a single space
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def is_linguistically_sufficient(self, text: str) -> bool:
        """Check if text contains sufficient linguistic characters for reliable detection."""
        if not text:
            return False
        # Count characters that are letters in any script (category 'L*')
        letters = [c for c in text if unicodedata.category(c).startswith("L")]
        if len(letters) < self.min_chars:
            return False
        # Check if text is entirely numeric or punctuation
        non_punct_num = [c for c in text if not c.isdigit() and not unicodedata.category(c).startswith("P") and not c.isspace()]
        if not non_punct_num:
            return False
        return True

    def detect(self, text: str, sample_id: Optional[str] = None) -> DetectionResult:
        """Detect the primary language of the input text.

        Args:
            text: Input string (OCR text or direct text).
            sample_id: Optional identifier for tracking.

        Returns:
            DetectionResult matching the FakeShield Phase 5 schema.
        """
        raw_text = text if text is not None else ""
        norm_text = self.normalize_text(raw_text)

        # Handle empty, short, or non-linguistic text
        if not self.is_linguistically_sufficient(norm_text):
            return DetectionResult(
                sample_id=sample_id,
                raw_text=raw_text,
                normalized_text=norm_text,
                language_code=None,
                language_name=None,
                confidence=None,
                status="insufficient_text",
            )

        try:
            # Re-seed to guarantee determinism even if other threads/calls altered it
            DetectorFactory.seed = 0
            code = detect(norm_text)
            name = SUPPORTED_LANGUAGES.get(code, f"Unknown ({code})")

            return DetectionResult(
                sample_id=sample_id,
                raw_text=raw_text,
                normalized_text=norm_text,
                language_code=code,
                language_name=name,
                confidence=None,  # Null per spec; not calibrated truth confidence
                status="success",
            )
        except LangDetectException:
            return DetectionResult(
                sample_id=sample_id,
                raw_text=raw_text,
                normalized_text=norm_text,
                language_code=None,
                language_name=None,
                confidence=None,
                status="insufficient_text",
            )
        except Exception as e:
            return DetectionResult(
                sample_id=sample_id,
                raw_text=raw_text,
                normalized_text=norm_text,
                language_code=None,
                language_name=None,
                confidence=None,
                status=f"error: {str(e)}",
            )


def detect_language(text: str, sample_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience functional wrapper for LanguageDetector.detect."""
    detector = LanguageDetector()
    return detector.detect(text, sample_id=sample_id).to_dict()
