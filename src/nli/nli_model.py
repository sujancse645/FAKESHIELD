"""nli_model.py - NLI inference module for FakeShield.

Provides a unified, reproducible interface for scoring claim-evidence pairs
using cross-encoder / sequence classification NLI transformers on CPU.
"""

import re
import html
from typing import Dict, List, Tuple, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from src.nli.nli_config import (
    NLIConfig,
    DEFAULT_NLI_MODEL_ID,
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    CANONICAL_NLI_LABELS,
)


class NLIModel:
    """NLI inference engine for evaluating claim-evidence semantic relationships."""

    def __init__(self, config: Optional[NLIConfig] = None):
        self.config = config or NLIConfig()
        self.model_id = self.config.model_id
        self._pipe = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Initializes Hugging Face sequence classification pipeline on CPU."""
        self._pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=self.config.device,
            top_k=None,  # Return all class probabilities
            truncation=True,
            max_length=self.config.max_length,
        )

    def preprocess_text(self, text: str, is_evidence: bool = False) -> str:
        """Cleans whitespace and HTML while strictly preserving numbers, dates, and negation."""
        if not text:
            return ""

        # Unescape HTML entities
        cleaned = html.unescape(text)

        if self.config.clean_html:
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        if self.config.normalize_whitespace:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if is_evidence and len(cleaned) > self.config.max_evidence_chars:
            cleaned = cleaned[: self.config.max_evidence_chars].rsplit(" ", 1)[0] + "..."

        return cleaned

    def _normalize_label(self, raw_label: str) -> str:
        """Maps model output labels to canonical FakeShield uppercase labels."""
        clean = raw_label.strip().upper()
        if "ENTAIL" in clean:
            return LABEL_ENTAILMENT
        elif "CONTRAD" in clean:
            return LABEL_CONTRADICTION
        elif "NEUTR" in clean:
            return LABEL_NEUTRAL
        return LABEL_NEUTRAL

    def predict(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Scores a single (claim, evidence) pair.

        In standard NLI formulation:
        - Premise = Evidence (the source text being referenced)
        - Hypothesis = Claim (the statement being evaluated)
        """
        clean_claim = self.preprocess_text(claim, is_evidence=False)
        clean_evidence = self.preprocess_text(evidence, is_evidence=True)

        if not clean_claim or not clean_evidence:
            return {
                "label": LABEL_NEUTRAL,
                "score": 0.0,
                "predicted_score": 0.0,
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 1.0,
                "probabilities": {
                    LABEL_CONTRADICTION: 0.0,
                    LABEL_ENTAILMENT: 0.0,
                    LABEL_NEUTRAL: 1.0,
                },
                "model": self.model_id,
                "claim": clean_claim,
                "evidence": clean_evidence,
            }

        # text=premise (evidence), text_pair=hypothesis (claim)
        try:
            raw_outputs = self._pipe({"text": clean_evidence, "text_pair": clean_claim})
        except Exception:
            return {
                "label": LABEL_NEUTRAL,
                "score": 0.0,
                "predicted_score": 0.0,
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 1.0,
                "probabilities": {
                    LABEL_CONTRADICTION: 0.0,
                    LABEL_ENTAILMENT: 0.0,
                    LABEL_NEUTRAL: 1.0,
                },
                "model": self.model_id,
                "claim": clean_claim,
                "evidence": clean_evidence,
            }

        probabilities = {LABEL_CONTRADICTION: 0.0, LABEL_ENTAILMENT: 0.0, LABEL_NEUTRAL: 0.0}

        for item in raw_outputs:
            canonical = self._normalize_label(item["label"])
            probabilities[canonical] = float(item["score"])

        # Determine top label
        top_label = max(probabilities, key=probabilities.get)
        top_score = probabilities[top_label]

        return {
            "label": top_label,
            "score": top_score,
            "predicted_score": round(top_score, 4),
            "entailment_score": round(probabilities.get(LABEL_ENTAILMENT, 0.0), 4),
            "contradiction_score": round(probabilities.get(LABEL_CONTRADICTION, 0.0), 4),
            "neutral_score": round(probabilities.get(LABEL_NEUTRAL, 0.0), 4),
            "probabilities": probabilities,
            "model": self.model_id,
            "claim": clean_claim,
            "evidence": clean_evidence,
        }

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Scores a batch of (claim, evidence) pairs."""
        return [self.predict(c, e) for c, e in pairs]
