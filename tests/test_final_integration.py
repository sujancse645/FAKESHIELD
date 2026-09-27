"""test_final_integration.py - Comprehensive Integration Test Suite for FakeShield.

Deterministic integration tests covering:
1. Pipeline import & initialization
2. English text pipeline
3. Tamil text pipeline
4. Hindi text pipeline
5. Image pipeline
6. Blank image handling & safe abstention
7. URL validation & safe failure
8. Retrieval failure behavior
9. Translation failure behavior
10. Language detection failure behavior
11. NLI output schema
12. Final verdict schema
13. Confidence schema
14. Evidence provenance schema
15. Safe UNVERIFIED behavior
"""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    FakeShieldPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineClaimResult,
)
from src.evidence_fusion.fusion import (
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
    CANONICAL_VERDICTS,
)
from src.nli.nli_model import NLIModel
from src.nli.nli_config import CANONICAL_NLI_LABELS
from src.evidence_retrieval.retriever import EvidenceRetriever
from src.language_detection.language_detector import LanguageDetector
from src.translation.translator import Translator


@pytest.fixture(scope="module")
def shared_pipeline():
    """Module-scoped shared pipeline instance."""
    cfg = PipelineConfig()
    return FakeShieldPipeline(cfg)


def test_1_pipeline_import():
    """Test 1: Verify all pipeline components are cleanly importable and instantiable."""
    assert FakeShieldPipeline is not None
    assert PipelineConfig is not None
    assert PipelineResult is not None
    assert PipelineClaimResult is not None
    cfg = PipelineConfig()
    assert cfg.max_claims == 5
    assert "text" in cfg.supported_input_types


def test_2_english_text_pipeline(shared_pipeline):
    """Test 2: Verify English claim executes cleanly through the pipeline."""
    text = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = shared_pipeline.verify(text, input_type="text", case_id="int_en_01")
    assert res.pipeline_status == "SUCCESS"
    assert res.input_type == "text"
    assert res.ocr.status == "skipped"
    assert res.language.code == "en"
    assert res.translation.status == "skipped_english"
    assert len(res.claims) >= 1
    assert res.overall_status in CANONICAL_VERDICTS


def test_3_tamil_text_pipeline(shared_pipeline):
    """Test 3: Verify Tamil text executes through language detection, translation, and pipeline."""
    text = "சென்னை வானிலை ஆய்வு மையம் இன்று பல மாவட்டங்களுக்கு கனமழை எச்சரிக்கை விடுத்துள்ளது."
    res = shared_pipeline.verify(text, input_type="text", case_id="int_ta_01")
    assert res.pipeline_status == "SUCCESS"
    assert res.language.code == "ta"
    assert res.translation.status in ("success", "skipped_english")
    assert len(res.translation.english_text) > 5
    assert res.overall_status in CANONICAL_VERDICTS


def test_4_hindi_text_pipeline(shared_pipeline):
    """Test 4: Verify Hindi text executes through language detection, translation, and pipeline."""
    text = "भारतीय रिजर्व बैंक ने मौद्रिक नीति में रेपो रेट को 6.5 प्रतिशत पर अपरिवर्तित रखा है।"
    res = shared_pipeline.verify(text, input_type="text", case_id="int_hi_01")
    assert res.pipeline_status == "SUCCESS"
    assert res.language.code == "hi"
    assert res.translation.status in ("success", "skipped_english")
    assert len(res.translation.english_text) > 5
    assert res.overall_status in CANONICAL_VERDICTS


def test_5_image_pipeline(shared_pipeline):
    """Test 5: Verify image input triggers OCR and subsequent pipeline stages."""
    img_path = os.path.join(str(PROJECT_ROOT), "data", "ocr_samples", "tamil_news_sample.png")
    if not os.path.exists(img_path):
        pytest.skip("Tamil OCR sample image not found")
    res = shared_pipeline.verify(img_path, input_type="image", case_id="int_img_01")
    assert res.pipeline_status == "SUCCESS"
    assert res.input_type == "image"
    assert res.ocr.status in ("success", "insufficient_text")
    assert res.overall_status in CANONICAL_VERDICTS


def test_6_blank_image_handling(shared_pipeline):
    """Test 6: Verify blank image degrades safely to UNVERIFIED without hallucinating text."""
    blank_path = os.path.join(str(PROJECT_ROOT), "data", "ocr_samples", "blank.png")
    if not os.path.exists(blank_path):
        pytest.skip("Blank sample image not found")
    res = shared_pipeline.verify(blank_path, input_type="image", case_id="int_blank_01")
    assert res.pipeline_status == "SUCCESS"
    assert res.overall_status == VERDICT_UNVERIFIED
    assert res.ocr.text == "" or len(res.ocr.text.strip()) < 3
    assert res.claims == []


def test_7_url_validation(shared_pipeline):
    """Test 7: Verify invalid and unreachable URLs fail safely to UNVERIFIED without crashing."""
    # Malformed URL
    res_malformed = shared_pipeline.verify("not_a_valid_url", input_type="url", case_id="int_url_01")
    assert res_malformed.overall_status == VERDICT_UNVERIFIED
    assert "failed" in (res_malformed.error_message or "").lower() or "invalid" in (res_malformed.error_message or "").lower()

    # Unreachable URL
    res_unreachable = shared_pipeline.verify("https://invalid-nonexistent-domain-fakeshield-12345.org", input_type="url", case_id="int_url_02")
    assert res_unreachable.overall_status == VERDICT_UNVERIFIED


def test_8_retrieval_failure_behavior(shared_pipeline):
    """Test 8: Verify claims with no available evidence safely abstain as UNVERIFIED."""
    rumor = "An obscure extraterrestrial artifact made of antimatter was recovered in a cave near an uncharted peak."
    res = shared_pipeline.verify(rumor, input_type="text", case_id="int_rumor_01")
    assert res.pipeline_status == "SUCCESS"
    assert res.overall_status == VERDICT_UNVERIFIED
    if res.claims:
        assert res.claims[0].verdict == VERDICT_UNVERIFIED
        assert res.claims[0].verification_confidence <= 0.40


def test_9_translation_failure_behavior():
    """Test 9: Verify translation failure handles gracefully without crash or fake pivot."""
    translator = Translator()
    # Extremely short or non-linguistic input
    res = translator.translate("12345678", source_language="ta")
    assert res["status"] == "insufficient_text"
    assert res["english_pivot"] == ""


def test_10_language_failure_behavior():
    """Test 10: Verify language detector gracefully rejects non-linguistic text."""
    detector = LanguageDetector()
    for non_text in ["12345678", "!@#$%^&*()", "     ", ""]:
        res = detector.detect(non_text)
        assert res.status == "insufficient_text"
        assert res.language_code is None


def test_11_nli_output_schema():
    """Test 11: Verify NLI engine returns complete canonical schema with all probability fields."""
    nli = NLIModel()
    out = nli.predict(
        claim="The interest rate was maintained at 6.5%.",
        evidence="The Reserve Bank kept the repo rate unchanged at 6.5%.",
    )
    assert "label" in out
    assert out["label"] in CANONICAL_NLI_LABELS
    assert "score" in out
    assert "probabilities" in out
    assert "entailment_score" in out
    assert "contradiction_score" in out
    assert "neutral_score" in out
    assert isinstance(out["score"], float)


def test_12_final_verdict_schema(shared_pipeline):
    """Test 12: Verify all verdicts strictly belong to canonical 5-verdict set."""
    for text in [
        "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent.",
        "Today the Prime Minister announced a nationwide lockdown to contain COVID-19.",
        "Random unverifiable rumor about mystical energy waves.",
    ]:
        res = shared_pipeline.verify(text, input_type="text")
        assert res.overall_status in CANONICAL_VERDICTS
        for c in res.claims:
            assert c.verdict in CANONICAL_VERDICTS


def test_13_confidence_schema(shared_pipeline):
    """Test 13: Verify verification confidence conforms to 0.0-1.0 range and proper field name."""
    res = shared_pipeline.verify("The repo rate is 6.5 percent.", input_type="text")
    assert res.pipeline_status == "SUCCESS"
    for c in res.claims:
        assert isinstance(c.verification_confidence, float)
        assert 0.0 <= c.verification_confidence <= 1.0


def test_14_evidence_provenance_schema():
    """Test 14: Verify retrieved evidence preserves full provenance fields with authentic URLs."""
    retriever = EvidenceRetriever()
    out = retriever.retrieve(query="RBI repo rate 6.5", claim_id="prov_01", claim_text="RBI repo rate unchanged at 6.5 percent")
    assert out["status"] in ("success", "no_results", "api_unavailable")
    for item in out.get("items", []):
        assert hasattr(item, "url") or "url" in item
        assert hasattr(item, "publisher") or "publisher" in item
        assert hasattr(item, "tier") or "tier" in item
        t = getattr(item, "tier", None) or item.get("tier")
        assert t in (1, 2, 3, 4)


def test_15_safe_unverified_behavior(shared_pipeline):
    """Test 15: Verify that lack of evidence leads to UNVERIFIED, not CONTRADICTED or guessed verdict."""
    unknown_claim = "The ancient city of Atlantis had four functioning geothermal generators."
    res = shared_pipeline.verify(unknown_claim, input_type="text", case_id="int_safe_01")
    assert res.overall_status == VERDICT_UNVERIFIED
    assert res.overall_status != VERDICT_CONTRADICTED
