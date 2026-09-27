"""test_translation.py - Pytest Test Suite for FakeShield Phase 6 Translation Layer.

Verifies:
1.  Translator initialization & config loading
2.  English skip logic (status='skipped', exact text preservation)
3.  Tamil translation to English pivot
4.  Hindi translation to English pivot
5.  Empty input handling (status='insufficient_text')
6.  Whitespace-only handling
7.  Very short text (<3 chars) handling
8.  Numeric-only input handling
9.  Punctuation-only input handling
10. Unsupported language handling
11. Text normalization preserves raw text
12. Output schema integrity
13. Deterministic repeated translation (100% stability)
14. BLEU and chrF metric computation
15. Number and negation preservation checks
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.translation_config import TranslationConfig
from src.translation.translator import Translator, TranslationResult, translate_to_english
from src.translation.translation_quality import TranslationQualityAnalyzer


@pytest.fixture(scope="module")
def translator():
    """Shared Translator fixture with lazy model loading."""
    t = Translator()
    return t


def test_translator_initialization(translator):
    """Test 1: Translator initializes with default configuration and target 'en'."""
    assert translator is not None
    assert translator.config.target_language == "en"
    assert translator.config.target_tag == "eng_Latn"
    assert translator.config.is_supported("ta")
    assert translator.config.is_supported("hi")
    assert translator.config.is_supported("en")


def test_english_skip_logic(translator):
    """Test 2: English text skips translation completely with status='skipped'."""
    text = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = translator.translate(text, source_language="en")
    assert res["status"] == "skipped"
    assert res["source_language"] == "en"
    assert res["target_language"] == "en"
    assert res["english_pivot"] == text
    assert res["is_pivot_english"] is True


def test_tamil_translation(translator):
    """Test 3: Tamil text translates successfully to English pivot."""
    text = "சென்னை வானிலை ஆய்வு மையம் இன்று மழை எச்சரிக்கை விடுத்துள்ளது."
    res = translator.translate(text, source_language="ta")
    assert res["status"] == "success"
    assert res["source_language"] == "ta"
    assert len(res["english_pivot"]) > 5
    assert "Chennai" in res["english_pivot"] or "warning" in res["english_pivot"].lower()


def test_hindi_translation(translator):
    """Test 4: Hindi text translates successfully to English pivot."""
    text = "भारतीय रिजर्व बैंक ने आज मौद्रिक नीति में कोई बदलाव नहीं किया।"
    res = translator.translate(text, source_language="hi")
    assert res["status"] == "success"
    assert res["source_language"] == "hi"
    assert len(res["english_pivot"]) > 5
    pivot_lower = res["english_pivot"].lower()
    assert any(kw in pivot_lower for kw in ["bank", "policy", "change", "reserve", "monetary", "rbi"]), \
        f"Hindi translation '{res['english_pivot']}' missing expected keywords"


def test_empty_input_handling(translator):
    """Test 5: Empty input returns status='insufficient_text' and empty pivot."""
    res = translator.translate("")
    assert res["status"] == "insufficient_text"
    assert res["english_pivot"] == ""
    assert res["is_pivot_english"] is False


def test_whitespace_only_handling(translator):
    """Test 6: Whitespace-only string returns status='insufficient_text'."""
    res = translator.translate("   \t  \n  ")
    assert res["status"] == "insufficient_text"
    assert res["english_pivot"] == ""


def test_very_short_text_handling(translator):
    """Test 7: Extremely short input (<3 characters) returns status='insufficient_text'."""
    res = translator.translate("hi")
    assert res["status"] == "insufficient_text"
    assert res["english_pivot"] == ""


def test_numeric_only_handling(translator):
    """Test 8: Pure numeric strings return status='insufficient_text'."""
    res = translator.translate("1234567890 98765")
    assert res["status"] == "insufficient_text"
    assert res["english_pivot"] == ""


def test_punctuation_only_handling(translator):
    """Test 9: Pure punctuation strings return status='insufficient_text'."""
    res = translator.translate("...?!@#$%^&*()")
    assert res["status"] == "insufficient_text"
    assert res["english_pivot"] == ""


def test_unsupported_language_handling(translator):
    """Test 10: Unsupported language codes return status='unsupported_language'."""
    res = translator.translate("Bonjour tout le monde", source_language="fr")
    assert res["status"] == "unsupported_language"
    assert res["english_pivot"] == ""


def test_normalization_preserves_raw_text(translator):
    """Test 11: Normalization preserves raw_text intact while cleaning whitespace."""
    raw = "  The   Government   will   announce  new   guidelines.  "
    res = translator.translate(raw, source_language="en")
    assert res["raw_text"] == raw
    assert res["normalized_text"] == "The Government will announce new guidelines."
    assert res["english_pivot"] == "The Government will announce new guidelines."


def test_output_schema_integrity(translator):
    """Test 12: Output dictionary has all required schema fields."""
    res = translator.translate("Breaking news update today.", source_language="en")
    required_keys = {
        "raw_text", "normalized_text", "source_language", "source_language_name",
        "target_language", "english_pivot", "status", "character_count_in",
        "character_count_out", "model_name", "is_pivot_english", "execution_time_sec"
    }
    assert required_keys.issubset(set(res.keys()))


def test_deterministic_translation(translator):
    """Test 13: Three consecutive translation calls produce character-identical output."""
    text = "விவசாயிகளுக்கு 50000 ரூபாய் வட்டியில்லா கடன் வழங்கப்படும்."
    run1 = translator.translate(text, source_language="ta")["english_pivot"]
    run2 = translator.translate(text, source_language="ta")["english_pivot"]
    run3 = translator.translate(text, source_language="ta")["english_pivot"]
    assert run1 == run2 == run3
    assert len(run1) > 0


def test_bleu_and_chrf_metric_computation():
    """Test 14: TranslationQualityAnalyzer computes sentence BLEU and chrF scores."""
    hyp = "The Reserve Bank kept the repo rate unchanged."
    ref = "The Reserve Bank kept the repo rate unchanged."
    bleu = TranslationQualityAnalyzer.compute_sentence_bleu(hyp, ref)
    chrf = TranslationQualityAnalyzer.compute_sentence_chrf(hyp, ref)
    assert bleu == 100.0
    assert chrf == 100.0

    # Partial match
    partial = "The Reserve Bank changed the interest rate today."
    bleu_part = TranslationQualityAnalyzer.compute_sentence_bleu(partial, ref)
    assert 0.0 < bleu_part < 100.0


def test_number_and_negation_preservation_checks():
    """Test 15: Semantic checks identify preserved numbers and negation polarities."""
    ref_num = "The project costs 50000 rupees."
    hyp_num = "The project cost is 50000 rupees."
    check_num = TranslationQualityAnalyzer.check_number_preservation(ref_num, hyp_num)
    assert check_num["preserved"] is True
    assert check_num["status"] == "preserved"

    ref_neg = "There is no water shortage in the district."
    hyp_neg = "There is no shortage of water in the district."
    check_neg = TranslationQualityAnalyzer.check_negation_preservation(ref_neg, hyp_neg)
    assert check_neg["preserved"] is True
    assert check_neg["status"] == "preserved"
