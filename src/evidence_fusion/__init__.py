"""FakeShield Evidence Fusion package."""

from src.evidence_fusion.fusion_config import FusionConfig
from src.evidence_fusion.fusion import (
    EvidenceFusionEngine,
    ScoredEvidenceItem,
    ReasoningResult,
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
    CANONICAL_VERDICTS,
)

__all__ = [
    "FusionConfig",
    "EvidenceFusionEngine",
    "ScoredEvidenceItem",
    "ReasoningResult",
    "VERDICT_VERIFIED",
    "VERDICT_CONTRADICTED",
    "VERDICT_MISLEADING",
    "VERDICT_OUT_OF_CONTEXT",
    "VERDICT_UNVERIFIED",
    "CANONICAL_VERDICTS",
]
