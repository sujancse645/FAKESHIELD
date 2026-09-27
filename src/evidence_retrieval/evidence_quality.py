"""evidence_quality.py - Quality and Evaluation Metrics for FakeShield Evidence Retrieval.

Computes hit rate, source tier distribution, relevance breakdown,
provenance completeness, and deduplication statistics.
"""

from typing import List, Dict, Any
from collections import Counter


class EvidenceQualityAnalyzer:
    """Evaluates the quality, relevance, and provenance of retrieved evidence packages."""

    @staticmethod
    def evaluate_retrieval_batch(retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate retrieval quality metrics across all evaluated claims."""
        total_claims = len(retrieval_results)
        if total_claims == 0:
            return {"total_claims": 0, "hit_rate": 0.0, "total_evidence_items": 0}

        hits = [r for r in retrieval_results if r.get("item_count", 0) > 0]
        hit_rate = round((len(hits) / total_claims) * 100.0, 2)

        all_items = []
        for r in retrieval_results:
            all_items.extend(r.get("items", []))

        total_items = len(all_items)

        def _get(it: Any, k: str, default: Any = None) -> Any:
            if isinstance(it, dict):
                return it.get(k, default)
            return getattr(it, k, default)

        # Tier breakdown
        tiers = [_get(item, "tier", 4) for item in all_items]
        tier_counts = dict(Counter(tiers))
        tier_percentages = {t: round((cnt / total_items) * 100.0, 2) for t, cnt in tier_counts.items()} if total_items else {}

        # Relevance breakdown
        relevances = [_get(item, "relevance_label", "relevant") for item in all_items]
        relevance_counts = dict(Counter(relevances))

        # Provenance completeness (items with valid URL and publisher)
        complete_provenance = [
            item for item in all_items
            if _get(item, "url") and _get(item, "publisher") and str(_get(item, "url")).startswith("http")
        ]
        provenance_rate = round((len(complete_provenance) / total_items) * 100.0, 2) if total_items else 100.0

        return {
            "total_claims_queried": total_claims,
            "claims_with_evidence": len(hits),
            "hit_rate_pct": hit_rate,
            "total_evidence_items": total_items,
            "mean_evidence_per_claim": round(total_items / total_claims, 2),
            "tier_counts": tier_counts,
            "tier_percentages": tier_percentages,
            "relevance_counts": relevance_counts,
            "provenance_completeness_pct": provenance_rate,
        }
