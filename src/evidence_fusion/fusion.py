"""fusion.py - Evidence fusion, source tier weighting, abstention, and out-of-context reasoning."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.nli.nli_model import NLIModel
from src.nli.nli_config import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)
from src.evidence_fusion.fusion_config import FusionConfig


VERDICT_VERIFIED = "VERIFIED"
VERDICT_CONTRADICTED = "CONTRADICTED"
VERDICT_MISLEADING = "MISLEADING"
VERDICT_OUT_OF_CONTEXT = "OUT-OF-CONTEXT"
VERDICT_UNVERIFIED = "UNVERIFIED"

CANONICAL_VERDICTS = [
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
]


@dataclass
class ScoredEvidenceItem:
    """Represents a single retrieved evidence item scored against a claim."""

    evidence_id: str
    text: str
    url: str
    publisher: str
    tier: int
    tier_weight: float
    nli_label: str
    nli_score: float
    weighted_support: float
    weighted_contradiction: float
    date: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "url": self.url,
            "publisher": self.publisher,
            "tier": self.tier,
            "tier_weight": self.tier_weight,
            "nli_label": self.nli_label,
            "nli_score": round(self.nli_score, 4),
            "weighted_support": round(self.weighted_support, 4),
            "weighted_contradiction": round(self.weighted_contradiction, 4),
            "date": self.date,
            "title": self.title,
        }


@dataclass
class ReasoningResult:
    """Final structured reasoning object produced by FakeShield Evidence Fusion."""

    claim_id: str
    claim: str
    evidence: List[ScoredEvidenceItem]
    support_score: float
    contradiction_score: float
    temporal_status: str
    verdict: str
    confidence: float
    explanation: str
    temporal_flags: List[str] = field(default_factory=list)

    @property
    def verification_confidence(self) -> float:
        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "evidence": [e.to_dict() for e in self.evidence],
            "nli_results": [
                {
                    "evidence_id": e.evidence_id,
                    "label": e.nli_label,
                    "score": round(e.nli_score, 4),
                    "tier": e.tier,
                    "weighted_support": round(e.weighted_support, 4),
                    "weighted_contradiction": round(e.weighted_contradiction, 4),
                }
                for e in self.evidence
            ],
            "support_score": round(self.support_score, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "temporal_flags": self.temporal_flags,
            "temporal_status": self.temporal_status,
            "verdict": self.verdict,
            "verification_confidence": round(self.confidence, 4),
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
        }


def _get_item_val(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class EvidenceFusionEngine:
    """Combines NLI predictions, source reliability tiers, abstention, and out-of-context detection."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        nli_model: Optional[NLIModel] = None,
    ):
        self.config = config or FusionConfig()
        self.nli_model = nli_model or NLIModel()

    def check_temporal_consistency(
        self, claim: str, evidence_items: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Detects whether old evidence is being applied to a claim presented as current.

        Returns:
            (temporal_status, detail_message)
            where status is 'out_of_context', 'consistent', or 'undated'.
        """
        claim_lower = claim.lower()
        has_current_framing = any(
            indicator in claim_lower for indicator in self.config.currentness_indicators
        )

        # Extract publication dates from evidence items
        valid_dates = []
        for item in evidence_items:
            raw_date = _get_item_val(item, "date")
            if raw_date and isinstance(raw_date, str):
                match = re.search(r"(\d{4})", raw_date)
                if match:
                    valid_dates.append((int(match.group(1)), raw_date, _get_item_val(item, "publisher", "Unknown")))

        if not valid_dates:
            return "undated", "No publication dates available for temporal verification."

        current_ref_year = datetime.now().year
        oldest_date = min(valid_dates, key=lambda x: x[0])
        year_diff = current_ref_year - oldest_date[0]

        # Only trigger out-of-context if the claim explicitly uses current-time framing
        # AND the evidence is significantly older (>= max_temporal_gap_years)
        if has_current_framing and year_diff >= self.config.max_temporal_gap_years:
            msg = (
                f"Claim frames event as current or recent, but primary retrieved evidence from "
                f"{oldest_date[2]} is dated {oldest_date[1]} ({year_diff} years prior)."
            )
            return "out_of_context", msg

        return "consistent", "Temporal framing aligns with retrieved publication dates."

    def fuse(
        self,
        claim: str,
        evidence_items: List[Dict[str, Any]],
        claim_id: str = "claim_0",
    ) -> ReasoningResult:
        """Executes full evidence fusion pipeline for a claim and its retrieved evidence."""
        scored_evidence: List[ScoredEvidenceItem] = []

        total_weighted_support = 0.0
        total_weighted_contradiction = 0.0

        for idx, item in enumerate(evidence_items):
            ev_id = _get_item_val(item, "evidence_id", f"{claim_id}_ev{idx+1}")
            ev_text = _get_item_val(item, "evidence_text") or _get_item_val(item, "text", "")
            ev_url = _get_item_val(item, "url", "")
            ev_pub = _get_item_val(item, "publisher", "Unknown")
            ev_title = _get_item_val(item, "title", "")
            ev_date = _get_item_val(item, "date")

            # Determine Tier and weight
            raw_tier = _get_item_val(item, "tier", 4)
            try:
                tier_int = int(raw_tier)
            except (ValueError, TypeError):
                tier_int = 4
            tier_weight = self.config.source_tier_weights.get(tier_int, 0.40)

            # NLI scoring
            nli_res = self.nli_model.predict(claim=claim, evidence=ev_text)
            nli_label = nli_res["label"]
            nli_score = nli_res["score"]

            w_sup = 0.0
            w_con = 0.0

            if nli_label == LABEL_ENTAILMENT and nli_score >= self.config.nli_score_threshold:
                w_sup = tier_weight * nli_score
            elif nli_label == LABEL_CONTRADICTION and nli_score >= self.config.nli_score_threshold:
                w_con = tier_weight * nli_score
            # NEUTRAL contributes 0.0 to both support and contradiction

            total_weighted_support += w_sup
            total_weighted_contradiction += w_con

            scored_item = ScoredEvidenceItem(
                evidence_id=ev_id,
                text=ev_text,
                url=ev_url,
                publisher=ev_pub,
                tier=tier_int,
                tier_weight=tier_weight,
                nli_label=nli_label,
                nli_score=nli_score,
                weighted_support=w_sup,
                weighted_contradiction=w_con,
                date=ev_date,
                title=ev_title,
            )
            scored_evidence.append(scored_item)

        # Temporal / Out-of-context check
        temporal_status, temporal_msg = self.check_temporal_consistency(claim, evidence_items)

        # Verdict logic with explicit abstention
        sup = total_weighted_support
        con = total_weighted_contradiction

        if temporal_status == "out_of_context" and (
            sup >= self.config.min_evidence_strength
            or con >= self.config.min_evidence_strength
            or any(s.nli_label in (LABEL_ENTAILMENT, LABEL_CONTRADICTION) for s in scored_evidence)
        ):
            verdict = VERDICT_OUT_OF_CONTEXT
        elif sup < self.config.min_evidence_strength and con < self.config.min_evidence_strength:
            verdict = VERDICT_UNVERIFIED
        elif (
            sup >= self.config.conflict_threshold
            and con >= self.config.conflict_threshold
            and abs(sup - con) < self.config.support_margin
        ):
            verdict = VERDICT_MISLEADING
        elif con >= self.config.min_evidence_strength and (con - sup) >= self.config.contradiction_margin:
            verdict = VERDICT_CONTRADICTED
        elif sup >= self.config.min_evidence_strength and (sup - con) >= self.config.support_margin:
            verdict = VERDICT_VERIFIED
        else:
            verdict = VERDICT_UNVERIFIED

        # Verification Confidence Calculation (Evidence confidence, NOT truth probability)
        if verdict == VERDICT_UNVERIFIED:
            conf = round(max(sup, con) * 0.4, 4)
        else:
            total_weight = sup + con
            margin = abs(sup - con)
            ratio = margin / (total_weight + 1e-6)
            highest_tier_weight = max([s.tier_weight for s in scored_evidence], default=0.40)
            conf = round(min(0.99, max(0.40, 0.45 + 0.50 * ratio * highest_tier_weight)), 4)

        # Generate structured explanation
        explanation = self._generate_explanation(
            claim=claim,
            verdict=verdict,
            scored_evidence=scored_evidence,
            sup=sup,
            con=con,
            temporal_status=temporal_status,
            temporal_msg=temporal_msg,
        )

        temporal_flags = [temporal_msg] if temporal_status == "out_of_context" else []

        return ReasoningResult(
            claim_id=claim_id,
            claim=claim,
            evidence=scored_evidence,
            support_score=round(sup, 4),
            contradiction_score=round(con, 4),
            temporal_status=temporal_status,
            verdict=verdict,
            confidence=conf,
            explanation=explanation,
            temporal_flags=temporal_flags,
        )

    def _generate_explanation(
        self,
        claim: str,
        verdict: str,
        scored_evidence: List[ScoredEvidenceItem],
        sup: float,
        con: float,
        temporal_status: str,
        temporal_msg: str,
    ) -> str:
        """Generates a deterministic, factual explanation grounded strictly in evidence."""
        if not scored_evidence:
            return "UNVERIFIED: No credible external evidence could be retrieved to substantiate or refute the claim."

        if verdict == VERDICT_OUT_OF_CONTEXT:
            return f"OUT-OF-CONTEXT: {temporal_msg}"

        if verdict == VERDICT_UNVERIFIED:
            return (
                f"UNVERIFIED: Retrieved evidence strength (Support={sup:.2f}, Contradiction={con:.2f}) "
                f"does not exceed the confidence threshold ({self.config.min_evidence_strength:.2f}) required for a conclusive verdict."
            )

        if verdict == VERDICT_MISLEADING:
            top_sup = next((e for e in scored_evidence if e.weighted_support > 0), None)
            top_con = next((e for e in scored_evidence if e.weighted_contradiction > 0), None)
            sup_pub = top_sup.publisher if top_sup else "Source"
            con_pub = top_con.publisher if top_con else "Source"
            return (
                f"MISLEADING: Substantial conflicting evidence detected. Evidence from {sup_pub} provides support "
                f"(weight {sup:.2f}), while reporting from {con_pub} directly contradicts aspects of the claim (weight {con:.2f})."
            )

        if verdict == VERDICT_CONTRADICTED:
            top_con = max(scored_evidence, key=lambda e: e.weighted_contradiction)
            return (
                f"CONTRADICTED: Credible evidence from {top_con.publisher} (Tier {top_con.tier}) directly contradicts the claim "
                f"with high NLI confidence ({top_con.nli_score:.2f}). Total contradiction score: {con:.2f}."
            )

        if verdict == VERDICT_VERIFIED:
            top_sup = max(scored_evidence, key=lambda e: e.weighted_support)
            return (
                f"VERIFIED: Credible evidence from {top_sup.publisher} (Tier {top_sup.tier}) entails and confirms the claim "
                f"with high NLI confidence ({top_sup.nli_score:.2f}). Total support score: {sup:.2f}."
            )

        return f"Verdict reached: {verdict}."
