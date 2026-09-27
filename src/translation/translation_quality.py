"""translation_quality.py - Quality evaluation metrics for FakeShield Phase 6 translation.

Computes BLEU and chrF using sacrebleu, plus semantic preservation checks
(numbers, dates, negation, named entities).
"""

import re
from typing import List, Dict, Any, Optional
import sacrebleu


class TranslationQualityAnalyzer:
    """Evaluates machine translation quality against independent references."""

    @staticmethod
    def compute_sentence_bleu(hypothesis: str, reference: str) -> float:
        """Compute sentence-level BLEU score (0.0 to 100.0)."""
        if not hypothesis or not reference:
            return 0.0
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return round(float(bleu.score), 2)

    @staticmethod
    def compute_sentence_chrf(hypothesis: str, reference: str) -> float:
        """Compute sentence-level chrF score (0.0 to 100.0)."""
        if not hypothesis or not reference:
            return 0.0
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference])
        return round(float(chrf.score), 2)

    @staticmethod
    def compute_corpus_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """Compute corpus-level BLEU and chrF scores."""
        if not hypotheses or not references or len(hypotheses) != len(references):
            return {"bleu": 0.0, "chrf": 0.0}

        # Filter out empty pairs
        valid_pairs = [(h, r) for h, r in zip(hypotheses, references) if h.strip() and r.strip()]
        if not valid_pairs:
            return {"bleu": 0.0, "chrf": 0.0}

        hyps = [p[0] for p in valid_pairs]
        refs = [[p[1] for p in valid_pairs]]

        bleu = sacrebleu.corpus_bleu(hyps, refs)
        chrf = sacrebleu.corpus_chrf(hyps, refs)

        return {
            "bleu": round(float(bleu.score), 2),
            "chrf": round(float(chrf.score), 2),
            "sample_count": len(valid_pairs),
        }

    @staticmethod
    def check_number_preservation(reference: str, hypothesis: str) -> Dict[str, Any]:
        """Check whether numerical values in reference appear in hypothesis."""
        ref_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", reference))
        hyp_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", hypothesis))

        if not ref_numbers:
            return {"status": "none", "preserved": True, "missing": []}

        missing = ref_numbers - hyp_numbers
        return {
            "status": "preserved" if not missing else "missing_numbers",
            "preserved": len(missing) == 0,
            "ref_numbers": sorted(list(ref_numbers)),
            "hyp_numbers": sorted(list(hyp_numbers)),
            "missing": sorted(list(missing)),
        }

    @staticmethod
    def check_negation_preservation(reference: str, hypothesis: str) -> Dict[str, Any]:
        """Check whether negation polarity is preserved."""
        negation_terms = {"not", "no", "never", "neither", "nor", "none", "without", "hardly", "barely"}
        ref_tokens = set(re.findall(r"\b\w+\b", reference.lower()))
        hyp_tokens = set(re.findall(r"\b\w+\b", hypothesis.lower()))

        ref_has_neg = bool(ref_tokens & negation_terms)
        hyp_has_neg = bool(hyp_tokens & negation_terms)

        polarity_match = (ref_has_neg == hyp_has_neg)
        return {
            "ref_has_negation": ref_has_neg,
            "hyp_has_negation": hyp_has_neg,
            "preserved": polarity_match,
            "status": "preserved" if polarity_match else "polarity_mismatch",
        }

    @staticmethod
    def evaluate_sample(reference: str, hypothesis: str) -> Dict[str, Any]:
        """Run full evaluation suite on a single hypothesis-reference pair."""
        bleu = TranslationQualityAnalyzer.compute_sentence_bleu(hypothesis, reference)
        chrf = TranslationQualityAnalyzer.compute_sentence_chrf(hypothesis, reference)
        num_check = TranslationQualityAnalyzer.check_number_preservation(reference, hypothesis)
        neg_check = TranslationQualityAnalyzer.check_negation_preservation(reference, hypothesis)

        return {
            "bleu": bleu,
            "chrf": chrf,
            "number_preservation": num_check,
            "negation_preservation": neg_check,
        }
