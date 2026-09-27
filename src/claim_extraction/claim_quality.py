"""claim_quality.py - Quality and evaluation metrics for FakeShield Claim Extraction.

Evaluates claim coverage, type distributions, entity/number preservation,
and review metrics without fabricating subjective accuracy scores.
"""

import re
from typing import List, Dict, Any
from collections import Counter


class ClaimQualityAnalyzer:
    """Computes quality and distribution metrics for extracted claims."""

    @staticmethod
    def compute_type_distribution(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute the breakdown and percentages of claim types."""
        if not claims:
            return {"counts": {}, "percentages": {}, "total_claims": 0}

        types = [c.get("type", "other") for c in claims]
        total = len(types)
        counts = dict(Counter(types))
        percentages = {t: round((cnt / total) * 100.0, 2) for t, cnt in counts.items()}

        return {
            "counts": counts,
            "percentages": percentages,
            "total_claims": total,
        }

    @staticmethod
    def check_number_preservation(source_sentence: str, claim_text: str) -> Dict[str, Any]:
        """Verify whether numeric quantities in source sentence are preserved in extracted claim."""
        src_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", source_sentence))
        claim_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", claim_text))

        if not src_numbers:
            return {"status": "none", "preserved": True, "missing": []}

        missing = src_numbers - claim_numbers
        return {
            "status": "preserved" if not missing else "missing_numbers",
            "preserved": len(missing) == 0,
            "source_numbers": sorted(list(src_numbers)),
            "claim_numbers": sorted(list(claim_numbers)),
            "missing": sorted(list(missing)),
        }

    @staticmethod
    def evaluate_batch(claim_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate quality metrics across a batch of extracted claim sets."""
        total_inputs = len(claim_sets)
        if total_inputs == 0:
            return {"total_inputs": 0, "coverage_rate": 0.0, "mean_claims_per_input": 0.0}

        successful = [cs for cs in claim_sets if cs.get("status") == "success" and len(cs.get("claims", [])) > 0]
        coverage_rate = round((len(successful) / total_inputs) * 100.0, 2)

        all_claims = []
        for cs in claim_sets:
            all_claims.extend(cs.get("claims", []))

        mean_claims = round(len(all_claims) / total_inputs, 2)
        type_dist = ClaimQualityAnalyzer.compute_type_distribution(all_claims)

        # Number preservation check across all extracted claims
        num_checks = [
            ClaimQualityAnalyzer.check_number_preservation(c.get("source_sentence", ""), c.get("claim", ""))
            for c in all_claims
        ]
        with_numbers = [nc for nc in num_checks if nc["status"] != "none"]
        preserved_numbers = [nc for nc in with_numbers if nc["preserved"]]
        num_pres_rate = round((len(preserved_numbers) / len(with_numbers)) * 100.0, 2) if with_numbers else 100.0

        return {
            "total_inputs": total_inputs,
            "successful_inputs": len(successful),
            "coverage_rate": coverage_rate,
            "total_claims_extracted": len(all_claims),
            "mean_claims_per_input": mean_claims,
            "type_distribution": type_dist,
            "number_preservation_rate": num_pres_rate,
        }
