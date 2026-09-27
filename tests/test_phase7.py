"""test_phase7.py - Pytest Test Suite for FakeShield Phase 7.

Verifies:
1.  Factual claim extraction
2.  Opinion and subjectivity filtering
3.  Question filtering
4.  Maximum 5 claims limit enforcement
5.  Claim type assignment (statistic, policy, event, quote, etc.)
6.  Search query generation
7.  Entity preservation in search query
8.  Number preservation in search query
9.  Date preservation in search query
10. Source reliability Tier 1 assignment (official sources)
11. Source reliability Tier 2 assignment (fact-checkers)
12. Unknown source assigned default Tier 4
13. Evidence item schema and attribution integrity
14. Evidence deduplication
15. Deterministic claim and query generation across repeated runs
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.claim_extraction import ClaimExtractor, ExtractedClaim, ClaimSet, ClaimQualityAnalyzer
from src.evidence_retrieval import (
    QueryBuilder,
    EvidenceRetriever,
    EvidenceItem,
    SourceReliabilityAnalyzer,
    EvidenceQualityAnalyzer,
)


@pytest.fixture(scope="module")
def extractor():
    return ClaimExtractor(max_claims=5)


@pytest.fixture(scope="module")
def query_builder():
    return QueryBuilder()


@pytest.fixture(scope="module")
def reliability_analyzer():
    return SourceReliabilityAnalyzer()


@pytest.fixture(scope="module")
def retriever():
    return EvidenceRetriever()


def test_factual_claim_extraction(extractor):
    """Test 1: Extractor identifies factual checkable assertions."""
    text = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    cset = extractor.extract_claims(text, input_id="t1")
    assert cset.status == "success"
    assert len(cset.claims) >= 1
    assert "Reserve Bank" in cset.claims[0].claim


def test_opinion_filtering(extractor):
    """Test 2: Subjective opinions and emotional commentary are filtered out."""
    text = "In my opinion, this policy is terrible and everyone should be ashamed. The city will build 5 new bridges."
    cset = extractor.extract_claims(text, input_id="t2")
    # Only the factual bridge statement should be extracted
    assert len(cset.claims) == 1
    assert "bridges" in cset.claims[0].claim.lower()
    assert "terrible" not in cset.claims[0].claim.lower()


def test_question_filtering(extractor):
    """Test 3: Interrogative and rhetorical questions are filtered."""
    text = "Is it true that taxes are rising? The government announced a 2 percent tax reduction."
    cset = extractor.extract_claims(text, input_id="t3")
    assert len(cset.claims) == 1
    assert not cset.claims[0].claim.endswith("?")
    assert "tax reduction" in cset.claims[0].claim.lower()


def test_maximum_five_claims(extractor):
    """Test 4: Strict cap of 5 claims is enforced per input."""
    text = """
    First, the cabinet approved 100 crore for schools.
    Second, the minister launched 10 new buses.
    Third, the hospital opened 50 new beds.
    Fourth, the university admitted 2000 students.
    Fifth, the airport opened a second runway.
    Sixth, the port handled 50000 tons of cargo.
    Seventh, the city built 20 public parks.
    """
    cset = extractor.extract_claims(text, input_id="t4")
    assert len(cset.claims) <= 5


def test_claim_type_assignment(extractor):
    """Test 5: Claim typing categorizes statistical, policy, and event claims accurately."""
    stat_claim = extractor.extract_claims("Unemployment fell by 3.5 percent this quarter.").claims[0]
    assert stat_claim.type == "statistic"

    policy_claim = extractor.extract_claims("The cabinet approved a new pension welfare scheme.").claims[0]
    assert policy_claim.type == "policy"

    event_claim = extractor.extract_claims("ISRO launched a new communication satellite yesterday.").claims[0]
    assert event_claim.type == "event"


def test_query_generation(query_builder):
    """Test 6: Query builder produces concise non-empty query."""
    claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
    q = query_builder.build_query(claim)
    assert len(q.split()) <= 7
    assert len(q) > 0


def test_entity_preservation_in_query(query_builder):
    """Test 7: Named entities and acronyms are preserved in query."""
    claim = "ISRO successfully tested the cryogenic engine in Mahendragiri."
    q = query_builder.build_query(claim)
    assert "ISRO" in q
    assert "Mahendragiri" in q


def test_number_preservation_in_query(query_builder):
    """Test 8: Quantitative metrics and decimals are preserved in query."""
    claim = "The repo rate remains at 6.5 percent with 50000 crore allocation."
    q = query_builder.build_query(claim)
    assert "6.5" in q
    assert "50000" in q


def test_date_preservation_in_query(query_builder):
    """Test 9: Dates and calendar years are preserved in query."""
    claim = "The new air service will launch on 15 October 2026."
    q = query_builder.build_query(claim)
    assert "2026" in q or "15" in q or "October" in q


def test_tier_1_official_source_assignment(reliability_analyzer):
    """Test 10: Official government and institutional URLs are assigned Tier 1."""
    t1 = reliability_analyzer.get_tier("https://rbi.org.in/scripts/pressrelease.aspx")
    assert t1["tier"] == 1
    assert "Official" in t1["tier_name"]

    t2 = reliability_analyzer.get_tier("https://pib.gov.in/PressRelease.aspx")
    assert t2["tier"] == 1


def test_tier_2_fact_checker_assignment(reliability_analyzer):
    """Test 11: Established fact-checking organizations are assigned Tier 2."""
    res1 = reliability_analyzer.get_tier("https://www.boomlive.in/fact-check/story-123")
    assert res1["tier"] == 2
    assert "Fact-Checking" in res1["tier_name"]

    res2 = reliability_analyzer.get_tier("https://www.snopes.com/fact-check/rbi-claim/")
    assert res2["tier"] == 2


def test_unknown_source_assigned_tier_4(reliability_analyzer):
    """Test 12: Unrecognized domain defaults to Tier 4 without claiming it is false."""
    res = reliability_analyzer.get_tier("https://random-unverified-blog12345.xyz/news/post")
    assert res["tier"] == 4
    assert "Unknown" in res["tier_name"] or "Unverified" in res["tier_name"]


def test_evidence_schema_integrity():
    """Test 13: EvidenceItem dictionary contains all required provenance fields."""
    item = EvidenceItem(
        claim_id="c_01",
        evidence_text="Sample official statement",
        url="https://rbi.org.in/statement",
        publisher="Reserve Bank of India",
        title="Repo Rate Statement",
        date="2024-04-05",
        rating="True",
        source_type="official",
        tier=1,
        retrieval_method="benchmark_corpus",
        retrieval_query="RBI repo rate 6.5",
        relevance_label="relevant"
    )
    d = item.to_dict()
    required_keys = {"claim_id", "evidence_text", "url", "publisher", "title", "tier", "retrieval_method"}
    assert required_keys.issubset(set(d.keys()))


def test_evidence_deduplication(retriever):
    """Test 14: Duplicate URLs and identical titles are filtered out."""
    items = [
        EvidenceItem("c1", "text 1", "https://pib.gov.in/release1", "PIB", "Title A", None, None, "official", 1, "test", "q"),
        EvidenceItem("c1", "text 2", "https://pib.gov.in/release1", "PIB", "Title A", None, None, "official", 1, "test", "q"),
        EvidenceItem("c1", "text 3", "https://pib.gov.in/release2", "PIB", "Title B", None, None, "official", 1, "test", "q"),
    ]
    deduped = retriever.deduplicate_evidence(items)
    assert len(deduped) == 2
    urls = [it.url for it in deduped]
    assert len(urls) == len(set(urls))


def test_deterministic_claim_and_query_generation(extractor):
    """Test 15: Three repeated runs produce 100% identical claims and search queries."""
    text = "The Chief Minister announced that interest-free crop loans of 50000 rupees will be provided to farmers."
    r1 = extractor.extract_claims(text, input_id="det_1")
    r2 = extractor.extract_claims(text, input_id="det_1")
    r3 = extractor.extract_claims(text, input_id="det_1")

    assert [c.claim for c in r1.claims] == [c.claim for c in r2.claims] == [c.claim for c in r3.claims]
    assert [c.search_query for c in r1.claims] == [c.search_query for c in r2.claims] == [c.search_query for c in r3.claims]
