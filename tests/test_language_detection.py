"""test_language_detection.py - Pytest Test Suite for FakeShield Phase 5.

Verifies:
1.  English detection
2.  Tamil detection
3.  Hindi detection
4.  Telugu detection
5.  Empty text handling (returns insufficient_text)
6.  Whitespace-only handling
7.  Very short text handling (< 3 characters)
8.  Numeric-only handling
9.  Punctuation-only handling
10. Mixed-language handling
11. Normalization preserves original text
12. Output schema integrity and confidence=null
13. Deterministic repeated detection
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language_detection.language_detector import (
    LanguageDetector,
    detect_language,
    SUPPORTED_LANGUAGES,
    DetectionResult,
)


@pytest.fixture
def detector():
    return LanguageDetector()


def test_english_detection(detector):
    """Test 1: Standard English sentence is detected accurately."""
    res = detector.detect("The Reserve Bank of India kept the benchmark interest rate unchanged.")
    assert res.status == "success"
    assert res.language_code == "en"
    assert res.language_name == "English"


def test_tamil_detection(detector):
    """Test 2: Standard Tamil sentence is detected accurately."""
    res = detector.detect("சென்னை வானிலை ஆய்வு மையம் இன்று பல மாவட்டங்களில் கனமழை எச்சரிக்கை விடுத்துள்ளது.")
    assert res.status == "success"
    assert res.language_code == "ta"
    assert res.language_name == "Tamil"


def test_hindi_detection(detector):
    """Test 3: Standard Hindi sentence is detected accurately."""
    res = detector.detect("भारतीय अंतरिक्ष अनुसंधान संगठन ने नया उपग्रह सफलतापूर्वक प्रक्षेपित किया है।")
    assert res.status == "success"
    assert res.language_code == "hi"
    assert res.language_name == "Hindi"


def test_telugu_detection(detector):
    """Test 4: Standard Telugu sentence is detected accurately."""
    res = detector.detect("బంగాళాఖాతంలో ఏర్పడిన అల్పపీడనం కారణంగా కోస్తాంధ్ర జిల్లాల్లో భారీ వర్షాలు కురిసే అవకాశం ఉంది.")
    assert res.status == "success"
    assert res.language_code == "te"
    assert res.language_name == "Telugu"


def test_empty_text_handling(detector):
    """Test 5: Empty string returns status='insufficient_text' and null language."""
    res = detector.detect("")
    assert res.status == "insufficient_text"
    assert res.language_code is None
    assert res.language_name is None
    assert res.confidence is None


def test_whitespace_only_handling(detector):
    """Test 6: Whitespace-only string returns status='insufficient_text'."""
    res = detector.detect("   \n\t  \r  ")
    assert res.status == "insufficient_text"
    assert res.language_code is None
    assert res.language_name is None


def test_very_short_text_handling(detector):
    """Test 7: Extremely short strings (< 3 linguistic chars) return status='insufficient_text'."""
    assert detector.detect("a").status == "insufficient_text"
    assert detector.detect("த").status == "insufficient_text"
    assert detector.detect("hi").status == "insufficient_text"


def test_numeric_only_handling(detector):
    """Test 8: Numeric-only text returns status='insufficient_text'."""
    res = detector.detect("1234567890 9876543210 2026")
    assert res.status == "insufficient_text"
    assert res.language_code is None


def test_punctuation_only_handling(detector):
    """Test 9: Punctuation and symbols return status='insufficient_text'."""
    res = detector.detect("... ??? !!! --- @@@ ### $$$ %%%")
    assert res.status == "insufficient_text"
    assert res.language_code is None


def test_mixed_language_handling(detector):
    """Test 10: Code-mixed input produces a valid detected language without error."""
    res = detector.detect("Chennai-யில் heavy rain எச்சரிக்கை விடுக்கப்பட்டுள்ளது என வானிலை மையம் தெரிவித்தது.")
    assert res.status == "success"
    assert res.language_name in ("Tamil", "English")


def test_normalization_preserves_original(detector):
    """Test 11: Normalization preserves raw text while collapsing excessive whitespace in normalized text."""
    raw = "   The   quick   brown    fox   jumps.   "
    res = detector.detect(raw)
    assert res.raw_text == raw
    assert res.normalized_text == "The quick brown fox jumps."
    assert res.status == "success"


def test_output_schema_and_confidence(detector):
    """Test 12: Output dictionary has all required schema fields and confidence is null."""
    res_dict = detect_language("This is a clean verification test sentence.")
    expected_keys = {
        "sample_id", "raw_text", "normalized_text",
        "language_code", "language_name", "confidence", "status"
    }
    assert set(res_dict.keys()) == expected_keys
    assert res_dict["confidence"] is None  # Spec requires null confidence
    assert res_dict["status"] == "success"


def test_deterministic_detection(detector):
    """Test 13: Repeated detection calls on the same text produce identical results."""
    text = "காவிரி நீர் பிடிப்பு பகுதிகளில் பெய்து வரும் தொடர் மழையால் அணைகளுக்கு நீர்வரத்து அதிகரித்துள்ளது."
    results = [detector.detect(text).language_code for _ in range(5)]
    assert all(r == results[0] for r in results)
    assert results[0] == "ta"
