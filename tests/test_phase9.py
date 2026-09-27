"""test_phase9.py - Pytest Test Suite for FakeShield Phase 9 End-to-End Pipeline.

Verifies:
1.  Pipeline initialization and default configuration
2.  Text input processing end-to-end
3.  Image input processing with OCR integration
4.  Empty and whitespace input handling (safe abstention)
5.  Language detection integration
6.  English translation bypass (skipped_english status)
7.  Multilingual translation to English pivot (Tamil input)
8.  Multilingual translation to English pivot (Hindi input)
9.  Claim extraction integration adhering to max_claims
10. Evidence retrieval integration preserving full provenance
11. NLI model inference with correct premise/hypothesis ordering
12. Source reliability weighting adherence (Tiers 1-4)
13. VERIFIED verdict state generation
14. UNVERIFIED abstention state generation
15. OUT-OF-CONTEXT temporal discrepancy detection
16. URL pathway safe failure recovery (no crashes)
17. Standard output schema conformance (Phase 9 Section 15)
18. End-to-end execution determinism across repeated runs
"""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    FakeShieldPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineClaimResult,
    PipelineDeterminismAuditor,
)
from src.evidence_fusion.fusion import (
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
)


@pytest.fixture(scope="module")
def pipeline():
    """Module-scoped pipeline fixture to share cached models."""
    cfg = PipelineConfig()
    return FakeShieldPipeline(cfg)


def test_pipeline_initialization(pipeline):
    """Test 1: Verify pipeline initializes cleanly with all sub-components."""
    assert pipeline is not None
    assert pipeline.language_detector is not None
    assert pipeline.translator is not None
    assert pipeline.claim_extractor is not None
    assert pipeline.retriever is not None
    assert pipeline.nli_model is not None
    assert pipeline.fusion_engine is not None


def test_text_input_processing(pipeline):
    """Test 2: Verify plain English text verification flow."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    assert isinstance(res, PipelineResult)
    assert res.pipeline_status == "SUCCESS"
    assert res.input_type == "text"
    assert len(res.claims) >= 1
    assert res.overall_status == VERDICT_VERIFIED


def test_image_input_processing(pipeline):
    """Test 3: Verify image input flow with OCR text extraction."""
    img_path = str(PROJECT_ROOT / "data" / "ocr_samples" / "tamil_news_sample.png")
    res = pipeline.verify(input_data=img_path, input_type="image")
    assert res.input_type == "image"
    assert res.ocr.status == "success"
    assert len(res.ocr.text) > 0
    assert res.pipeline_status == "SUCCESS"


def test_empty_and_short_input_abstention(pipeline):
    """Test 4: Verify short and empty inputs trigger safe abstention."""
    res_empty = pipeline.verify(input_data="   ", input_type="text")
    assert res_empty.overall_status == VERDICT_UNVERIFIED
    assert res_empty.pipeline_status == "SUCCESS"

    res_short = pipeline.verify(input_data="hi", input_type="text")
    assert res_short.overall_status == VERDICT_UNVERIFIED


def test_language_detection_integration(pipeline):
    """Test 5: Verify language detection accurately classifies input."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    assert res.language.code == "en"
    assert res.language.name == "English"
    assert res.language.status == "success"


def test_english_translation_skip_logic(pipeline):
    """Test 6: Verify English input skips translation and preserves original text."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    assert res.translation.status == "skipped_english"
    assert res.translation.english_text == claim


def test_tamil_translation_integration(pipeline):
    """Test 7: Verify Tamil input converts to English pivot text."""
    ta_text = "சென்னை வானிலை ஆய்வு மையம் இன்று மழை எச்சரிக்கை விடுத்துள்ளது."
    res = pipeline.verify(input_data=ta_text, input_type="text")
    assert res.language.code == "ta"
    assert res.translation.status == "success"
    assert len(res.translation.english_text) > 0
    assert res.translation.english_text != ta_text


def test_hindi_translation_integration(pipeline):
    """Test 8: Verify Hindi input converts to English pivot text."""
    hi_text = "भारतीय रिजर्व बैंक ने अपनी प्रमुख ब्याज दर रेपो रेट को 6.5 प्रतिशत पर अपरिवर्तित रखा है।"
    res = pipeline.verify(input_data=hi_text, input_type="text")
    assert res.language.code == "hi"
    assert res.translation.status == "success"
    assert len(res.translation.english_text) > 0


def test_claim_extraction_integration(pipeline):
    """Test 9: Verify claims are extracted and respect max_claims limit."""
    text = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent. ISRO successfully launched the PSLV mission placing satellites into polar orbit."
    res = pipeline.verify(input_data=text, input_type="text")
    assert 1 <= len(res.claims) <= 5
    for c in res.claims:
        assert isinstance(c, PipelineClaimResult)
        assert len(c.claim) > 0


def test_evidence_retrieval_provenance(pipeline):
    """Test 10: Verify retrieved evidence preserves URL, publisher, and source tier."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    assert len(res.claims) > 0
    ev_list = res.claims[0].evidence
    assert len(ev_list) > 0
    first_ev = ev_list[0]
    assert "url" in first_ev and first_ev["url"].startswith("http")
    assert "publisher" in first_ev
    assert "tier" in first_ev and 1 <= first_ev["tier"] <= 4


def test_nli_inference_relationship(pipeline):
    """Test 11: Verify NLI outputs canonical labels and scores."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    first_claim = res.claims[0]
    assert len(first_claim.nli) > 0
    nli_rel = first_claim.nli[0]
    assert nli_rel["label"] in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    assert 0.0 <= nli_rel["score"] <= 1.0


def test_source_tier_weighting_application(pipeline):
    """Test 12: Verify tier weighting influences support/contradiction score."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    first_claim = res.claims[0]
    assert first_claim.support_score > 0.5
    assert first_claim.contradiction_score < 0.2


def test_verified_verdict_generation(pipeline):
    """Test 13: Verify strong official evidence yields VERIFIED verdict."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    assert res.overall_status == VERDICT_VERIFIED
    assert res.claims[0].verification_confidence >= 0.70


def test_unverified_abstention_on_unknown_claim(pipeline):
    """Test 14: Verify obscure claim without evidence yields UNVERIFIED."""
    obscure = "Ancient underground crystalline generators were excavated by medieval monks under Lake Geneva."
    res = pipeline.verify(input_data=obscure, input_type="text")
    assert res.overall_status == VERDICT_UNVERIFIED
    assert res.claims[0].verdict == VERDICT_UNVERIFIED


def test_out_of_context_temporal_detection(pipeline):
    """Test 15: Verify currentness framing with old evidence yields OUT-OF-CONTEXT."""
    ooc_claim = "Today the Prime Minister announced a nationwide lockdown to contain COVID-19."
    res = pipeline.verify(input_data=ooc_claim, input_type="text")
    assert res.overall_status == VERDICT_OUT_OF_CONTEXT
    assert len(res.claims[0].temporal_flags) > 0


def test_url_pathway_safe_failure(pipeline):
    """Test 16: Verify URL fetch failure handles gracefully without crash."""
    bad_url = "https://nonexistent-research-domain-error-987654.com/news"
    res = pipeline.verify(input_data=bad_url, input_type="url")
    assert res.pipeline_status == "SUCCESS"
    assert res.overall_status == VERDICT_UNVERIFIED
    assert res.error_message is not None


def test_structured_output_schema_conformance(pipeline):
    """Test 17: Verify JSON serialization adheres to Phase 9 Section 15."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    d = res.to_dict()

    expected_keys = [
        "input_type",
        "raw_input",
        "ocr",
        "language",
        "translation",
        "claims",
        "overall_status",
        "pipeline_status",
    ]
    for k in expected_keys:
        assert k in d

    assert isinstance(d["ocr"], dict)
    assert isinstance(d["language"], dict)
    assert isinstance(d["translation"], dict)
    assert isinstance(d["claims"], list)


def test_pipeline_execution_determinism(pipeline):
    """Test 18: Verify repeated identical runs produce identical outputs."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    runs = [pipeline.verify(input_data=claim, input_type="text").to_dict() for _ in range(3)]
    auditor = PipelineDeterminismAuditor()
    comparison = auditor.compare_runs(runs)
    assert comparison["is_deterministic"] is True
    assert len(comparison["variances"]) == 0
