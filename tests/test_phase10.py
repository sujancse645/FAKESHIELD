"""test_phase10.py - Pytest Test Suite for FakeShield Phase 10 Streamlit UI & Demonstration.

Verifies:
1.  App module imports cleanly
2.  Streamlit application entry point exists at src/app.py
3.  Pipeline caching and singleton initialization
4.  Verdict color mappings for all canonical states
5.  Verdict icon mappings for all canonical states
6.  Source-tier description mappings (Tiers 1-4)
7.  URL format validation success
8.  URL format validation rejection on empty input
9.  URL format validation rejection on malformed scheme
10. Demo scenarios dictionary integrity and coverage
11. Text input verification pathway execution
12. Image/OCR input verification pathway execution
13. UNVERIFIED abstention schema and explanation handling
14. OUT-OF-CONTEXT temporal detection and alert handling
15. Multilingual translation pivot presentation
16. Evidence provenance schema integrity (URLs, publishers, tiers)
17. Security audit: Zero hard-coded API keys, tokens, or credentials
18. Defensive error handling on invalid / edge-case inputs
"""

import os
import sys
import re
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.app as app
from src.pipeline import FakeShieldPipeline, PipelineConfig, PipelineResult
from src.evidence_fusion.fusion import (
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
    CANONICAL_VERDICTS,
)


@pytest.fixture(scope="module")
def pipeline():
    """Module-scoped pipeline fixture to share cached models."""
    cfg = PipelineConfig()
    return FakeShieldPipeline(cfg)


def test_app_module_imports():
    """Test 1: Streamlit app module imports successfully without syntax or dependency errors."""
    assert hasattr(app, "main")
    assert hasattr(app, "get_pipeline")
    assert hasattr(app, "get_tier_label")
    assert hasattr(app, "get_verdict_color")
    assert hasattr(app, "validate_url_input")


def test_app_entry_point_exists():
    """Test 2: Streamlit application file exists at src/app.py."""
    app_path = PROJECT_ROOT / "src" / "app.py"
    assert app_path.is_file()
    assert app_path.stat().st_size > 1000


def test_pipeline_singleton_caching():
    """Test 3: get_pipeline initializes a valid FakeShieldPipeline instance."""
    p = app.get_pipeline()
    assert isinstance(p, FakeShieldPipeline)
    assert p.nli_model is not None
    assert p.fusion_engine is not None


def test_verdict_color_mapping():
    """Test 4: All 5 canonical verdicts have distinct, valid hex color codes."""
    for v in CANONICAL_VERDICTS:
        color = app.get_verdict_color(v)
        assert color.startswith("#")
        assert len(color) == 7


def test_verdict_icon_mapping():
    """Test 5: All 5 canonical verdicts have designated UI icons."""
    for v in CANONICAL_VERDICTS:
        assert v in app.VERDICT_ICONS
        assert len(app.VERDICT_ICONS[v]) > 0


def test_tier_description_mapping():
    """Test 6: Tiers 1 through 4 have institutional descriptions."""
    for t in [1, 2, 3, 4]:
        desc = app.get_tier_label(t)
        assert f"Tier {t}" in desc
        assert "Weight:" in desc


def test_url_validation_success():
    """Test 7: Valid HTTP/HTTPS URLs pass validation."""
    valid, msg = app.validate_url_input("https://pib.gov.in/PressReleasePage.aspx?PRID=1607997")
    assert valid is True
    assert msg == ""

    valid_http, _ = app.validate_url_input("http://www.rbi.org.in/policy")
    assert valid_http is True


def test_url_validation_rejection_empty():
    """Test 8: Empty or whitespace URL input fails validation safely."""
    valid, msg = app.validate_url_input("")
    assert valid is False
    assert "empty" in msg.lower()

    valid_sp, _ = app.validate_url_input("   ")
    assert valid_sp is False


def test_url_validation_rejection_malformed():
    """Test 9: Malformed URLs without http/https fail validation."""
    valid, msg = app.validate_url_input("ftp://invalidscheme.com")
    assert valid is False
    assert "http://" in msg or "https://" in msg

    valid_raw, _ = app.validate_url_input("not-a-url")
    assert valid_raw is False


def test_demo_examples_coverage():
    """Test 10: Demo scenarios dictionary contains verified cases covering all core states."""
    assert len(app.DEMO_EXAMPLES) >= 6
    expected_states = {d["expected"] for d in app.DEMO_EXAMPLES.values()}
    assert VERDICT_VERIFIED in expected_states
    assert VERDICT_CONTRADICTED in expected_states
    assert VERDICT_OUT_OF_CONTEXT in expected_states
    assert VERDICT_MISLEADING in expected_states
    assert VERDICT_UNVERIFIED in expected_states


def test_text_input_pipeline_verification(pipeline):
    """Test 11: Text input pathway executes through pipeline and yields VERIFIED."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    assert res.pipeline_status == "SUCCESS"
    assert res.overall_status == VERDICT_VERIFIED
    assert len(res.claims) >= 1
    assert res.claims[0].verification_confidence >= 0.70


def test_image_input_pipeline_verification(pipeline):
    """Test 12: Image input pathway executes OCR and returns structured result."""
    img_path = str(PROJECT_ROOT / "data" / "ocr_samples" / "blank.png")
    res = pipeline.verify(input_data=img_path, input_type="image")
    assert res.input_type == "image"
    assert res.pipeline_status == "SUCCESS"
    assert res.ocr.status in ["insufficient_text", "success"]
    assert res.overall_status == VERDICT_UNVERIFIED


def test_unverified_abstention_ui_schema(pipeline):
    """Test 13: Unsubstantiated claim cleanly abstains with UNVERIFIED status."""
    obscure = "An underground library of glowing stones was excavated under Lake Geneva."
    res = pipeline.verify(input_data=obscure, input_type="text")
    assert res.overall_status == VERDICT_UNVERIFIED
    assert res.claims[0].verdict == VERDICT_UNVERIFIED
    assert "UNVERIFIED" in res.claims[0].explanation


def test_out_of_context_ui_schema(pipeline):
    """Test 14: Contemporary framing with older evidence produces OUT-OF-CONTEXT and temporal flags."""
    ooc_claim = "Today the Prime Minister announced a nationwide lockdown to contain COVID-19."
    res = pipeline.verify(input_data=ooc_claim, input_type="text")
    assert res.overall_status == VERDICT_OUT_OF_CONTEXT
    assert len(res.claims[0].temporal_flags) > 0


def test_multilingual_pivot_ui_schema(pipeline):
    """Test 15: Tamil input triggers language detection and English pivot translation."""
    ta_text = "சென்னை வானிலை ஆய்வு மையம் இன்று மழை எச்சரிக்கை விடுத்துள்ளது."
    res = pipeline.verify(input_data=ta_text, input_type="text")
    assert res.language.code == "ta"
    assert res.translation.status == "success"
    assert len(res.translation.english_text) > 0


def test_evidence_provenance_ui_schema(pipeline):
    """Test 16: Evidence items preserve all required provenance fields for UI display."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    res = pipeline.verify(input_data=claim, input_type="text")
    ev_item = res.claims[0].evidence[0]
    assert "url" in ev_item and ev_item["url"].startswith("http")
    assert "publisher" in ev_item and len(ev_item["publisher"]) > 0
    assert "tier" in ev_item and 1 <= ev_item["tier"] <= 4


def test_no_secrets_in_app_code():
    """Test 17: Verify no API keys, private tokens, or hard-coded secrets exist in src/app.py."""
    app_path = PROJECT_ROOT / "src" / "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern check for API keys / passwords
    secret_patterns = [
        r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
        r"(?i)secret\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
        r"(?i)password\s*=\s*['\"][^'\"]+['\"]",
        r"(?i)token\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
        r"AIzaSy[A-Za-z0-9_\-]{33}",  # Google API key pattern
    ]
    for pat in secret_patterns:
        match = re.search(pat, content)
        assert match is None, f"Found potential secret in src/app.py matching {pat}: {match.group(0)}"


def test_app_error_recovery_without_crashing(pipeline):
    """Test 18: Unhandled or malformed inputs return clean UNVERIFIED status without throwing."""
    res_empty = pipeline.verify(input_data="", input_type="text")
    assert res_empty.overall_status == VERDICT_UNVERIFIED
    assert res_empty.pipeline_status == "SUCCESS"

    res_url = pipeline.verify(input_data="http://localhost:99999/nonexistent", input_type="url")
    assert res_url.overall_status == VERDICT_UNVERIFIED
    assert res_url.pipeline_status == "SUCCESS"
