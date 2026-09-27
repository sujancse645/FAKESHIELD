"""evidence_retrieval - FakeShield Phase 7 Evidence Acquisition & Reliability Layer.

Provides query building, fact-check retrieval, source reliability tiering, and deduplication.
"""

from src.evidence_retrieval.query_builder import QueryBuilder
from src.evidence_retrieval.source_reliability import SourceReliabilityAnalyzer
from src.evidence_retrieval.retriever import EvidenceRetriever, EvidenceItem
from src.evidence_retrieval.evidence_quality import EvidenceQualityAnalyzer

__all__ = [
    "QueryBuilder",
    "SourceReliabilityAnalyzer",
    "EvidenceRetriever",
    "EvidenceItem",
    "EvidenceQualityAnalyzer",
]
