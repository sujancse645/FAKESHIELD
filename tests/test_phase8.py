"""test_phase8.py - Pytest Test Suite for FakeShield Phase 8.

Verifies:
1.  NLI entailment prediction
2.  NLI contradiction prediction
3.  NLI neutral prediction
4.  NLI structured output dictionary format
5.  NLI score handling and probability distribution
6.  Source-tier weighting (Tiers 1-4)
7.  Support score aggregation
8.  Contradiction score aggregation
9.  Neutral evidence contributes zero weight
10. VERIFIED verdict threshold logic
11. CONTRADICTED verdict threshold logic
12. MISLEADING verdict condition (conflicting evidence)
13. UNVERIFIED abstention logic (insufficient evidence)
14. OUT-OF-CONTEXT detection with currentness framing and old evidence
15. Historical claim handling (preserves old evidence validity)
16. Missing date handling (defaults to undated, does not trigger false OOC)
17. Deterministic inference across repeated runs
18. Structured explanation generation
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nli import (
    NLIModel,
    NLIConfig,
    NLIQualityAnalyzer,
    LABEL_ENTAILMENT,
    LABEL_CONTRADICTION,
    LABEL_NEUTRAL,
)
from src.evidence_fusion import (
    EvidenceFusionEngine,
    FusionConfig,
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
)


@pytest.fixture(scope="module")
def nli_model():
    return NLIModel()


@pytest.fixture(scope="module")
def fusion_engine(nli_model):
    return EvidenceFusionEngine(nli_model=nli_model)


def test_nli_entailment(nli_model):
    """Test 1: NLI correctly predicts ENTAILMENT for supportive pair."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    evidence = "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent."
    res = nli_model.predict(claim, evidence)
    assert res["label"] == LABEL_ENTAILMENT
    assert res["score"] > 0.80


def test_nli_contradiction(nli_model):
    """Test 2: NLI correctly predicts CONTRADICTION for contradictory pair."""
    claim = "The Reserve Bank of India reduced the benchmark repo rate by 50 basis points to 6.0 percent."
    evidence = "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent."
    res = nli_model.predict(claim, evidence)
    assert res["label"] == LABEL_CONTRADICTION
    assert res["score"] > 0.80


def test_nli_neutral(nli_model):
    """Test 3: NLI correctly predicts NEUTRAL for unrelated pair."""
    claim = "The Reserve Bank of India plans to introduce a retail central bank digital currency next year."
    evidence = "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent."
    res = nli_model.predict(claim, evidence)
    assert res["label"] == LABEL_NEUTRAL
    assert res["score"] > 0.80


def test_nli_structured_output(nli_model):
    """Test 4: Predict returns complete structured dictionary."""
    res = nli_model.predict("Earth has a moon.", "The Moon orbits Earth.")
    required_keys = {"label", "score", "probabilities", "model", "claim", "evidence"}
    assert required_keys.issubset(set(res.keys()))
    assert isinstance(res["probabilities"], dict)


def test_nli_score_handling(nli_model):
    """Test 5: Probability distribution sums to approximately 1.0."""
    res = nli_model.predict("Apples are fruit.", "Apples grow on fruit trees.")
    probs = res["probabilities"]
    assert LABEL_ENTAILMENT in probs
    assert LABEL_CONTRADICTION in probs
    assert LABEL_NEUTRAL in probs
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01


def test_source_tier_weighting(fusion_engine):
    """Test 6: Source tier weights correctly configured."""
    weights = fusion_engine.config.source_tier_weights
    assert weights[1] == 1.00
    assert weights[2] == 0.85
    assert weights[3] == 0.70
    assert weights[4] == 0.40


def test_support_aggregation(fusion_engine):
    """Test 7: Support score accumulates appropriately from supportive evidence."""
    claim = "The Union Cabinet approved capital outlay for 10000 km of highways."
    evidence = [
        {
            "text": "The Union Cabinet approved capital outlay for 10000 km of highways.",
            "tier": 1,
            "publisher": "PIB",
            "date": "2023-09-12",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.support_score > 0.7
    assert res.contradiction_score == 0.0


def test_contradiction_aggregation(fusion_engine):
    """Test 8: Contradiction score accumulates from contradicting evidence."""
    claim = "The highway project was completely cancelled."
    evidence = [
        {
            "text": "The Union Cabinet approved capital outlay for 10000 km of highways.",
            "tier": 1,
            "publisher": "PIB",
            "date": "2023-09-12",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.contradiction_score > 0.7
    assert res.support_score == 0.0


def test_neutral_evidence_handling(fusion_engine):
    """Test 9: Neutral evidence contributes zero support and zero contradiction."""
    claim = "The stock market rose today."
    evidence = [
        {
            "text": "The Union Cabinet approved capital outlay for 10000 km of highways.",
            "tier": 1,
            "publisher": "PIB",
            "date": "2023-09-12",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.support_score == 0.0
    assert res.contradiction_score == 0.0


def test_verified_threshold(fusion_engine):
    """Test 10: Clear supportive evidence yields VERIFIED."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    evidence = [
        {
            "text": "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent.",
            "tier": 1,
            "publisher": "Reserve Bank of India",
            "date": "2024-04-05",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.verdict == VERDICT_VERIFIED


def test_contradicted_threshold(fusion_engine):
    """Test 11: Clear conflicting evidence yields CONTRADICTED."""
    claim = "The Reserve Bank of India reduced the benchmark repo rate by 50 basis points to 6.0 percent."
    evidence = [
        {
            "text": "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent.",
            "tier": 1,
            "publisher": "Reserve Bank of India",
            "date": "2024-04-05",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.verdict == VERDICT_CONTRADICTED


def test_misleading_condition(fusion_engine):
    """Test 12: Substantial conflicting evidence yields MISLEADING."""
    claim = "The highway expansion project was completed under budget."
    evidence = [
        {
            "text": "The transport ministry issued a press release claiming the highway expansion project was completed under budget.",
            "tier": 1,
            "publisher": "PIB",
            "date": "2023-09-12",
        },
        {
            "text": "Independent audit reports concluded the highway expansion project exceeded cost allocations by 20 percent and was not completed under budget.",
            "tier": 2,
            "publisher": "FactCheck.org",
            "date": "2023-09-15",
        },
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.verdict == VERDICT_MISLEADING


def test_unverified_abstention(fusion_engine):
    """Test 13: Empty or weak evidence triggers UNVERIFIED abstention."""
    claim = "An unverified entity discovered a secret island."
    res = fusion_engine.fuse(claim, [])
    assert res.verdict == VERDICT_UNVERIFIED


def test_out_of_context_detection(fusion_engine):
    """Test 14: Currentness framing in claim against old evidence triggers OUT-OF-CONTEXT."""
    claim = "Today the Prime Minister announced a nationwide lockdown to contain COVID-19."
    evidence = [
        {
            "text": "The Prime Minister today announced a nationwide lockdown to contain COVID-19 transmission.",
            "tier": 1,
            "publisher": "PIB",
            "date": "2020-03-24",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.temporal_status == "out_of_context"
    assert res.verdict == VERDICT_OUT_OF_CONTEXT


def test_historical_claim_handling(fusion_engine):
    """Test 15: Historical claim without currentness framing does not trigger OUT-OF-CONTEXT."""
    claim = "The government announced a nationwide lockdown in March 2020."
    evidence = [
        {
            "text": "The Prime Minister today announced a nationwide lockdown to contain COVID-19 transmission.",
            "tier": 1,
            "publisher": "PIB",
            "date": "2020-03-24",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.temporal_status == "consistent"
    assert res.verdict != VERDICT_OUT_OF_CONTEXT


def test_missing_date_handling(fusion_engine):
    """Test 16: Missing dates in evidence default cleanly to undated."""
    claim = "A new education policy was announced."
    evidence = [
        {
            "text": "The government released the national education policy blueprint.",
            "tier": 3,
            "publisher": "News Agency",
            "date": None,
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert res.temporal_status == "undated"
    assert res.verdict != VERDICT_OUT_OF_CONTEXT


def test_deterministic_inference(nli_model):
    """Test 17: Repeated predictions produce strictly identical outputs."""
    claim = "The Reserve Bank of India kept the repo rate at 6.5%."
    evidence = "RBI kept the benchmark repo rate unchanged at 6.50%."
    res1 = nli_model.predict(claim, evidence)
    res2 = nli_model.predict(claim, evidence)
    res3 = nli_model.predict(claim, evidence)
    assert res1["label"] == res2["label"] == res3["label"]
    assert res1["score"] == res2["score"] == res3["score"]


def test_explanation_generation(fusion_engine):
    """Test 18: Explanations are non-empty and cite source and reasoning."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    evidence = [
        {
            "text": "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent.",
            "tier": 1,
            "publisher": "Reserve Bank of India",
            "date": "2024-04-05",
        }
    ]
    res = fusion_engine.fuse(claim, evidence)
    assert len(res.explanation) > 20
    assert "Reserve Bank of India" in res.explanation
    assert "VERIFIED" in res.explanation
