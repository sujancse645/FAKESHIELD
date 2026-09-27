"""claim_extraction - FakeShield Phase 7 Factual Claim Extraction Module.

Extracts checkable factual statements from English text / Phase 6 English pivot representation.
"""

from src.claim_extraction.claim_extractor import (
    ClaimExtractor,
    ExtractedClaim,
    ClaimSet,
    CLAIM_TYPES,
)
from src.claim_extraction.claim_quality import ClaimQualityAnalyzer

__all__ = [
    "ClaimExtractor",
    "ExtractedClaim",
    "ClaimSet",
    "CLAIM_TYPES",
    "ClaimQualityAnalyzer",
]
