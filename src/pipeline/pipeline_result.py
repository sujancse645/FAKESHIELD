"""pipeline_result.py - Structured output schema for FakeShield verification pipeline."""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.evidence_fusion.fusion import (
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
    CANONICAL_VERDICTS,
)


@dataclass
class OCRMetadata:
    """OCR processing stage metadata."""
    text: str = ""
    quality: Dict[str, Any] = field(default_factory=dict)
    status: str = "skipped"  # skipped, success, failed, insufficient_text


@dataclass
class LanguageMetadata:
    """Language detection stage metadata."""
    code: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "pending"  # success, insufficient_text, unknown


@dataclass
class TranslationMetadata:
    """Translation and English pivot stage metadata."""
    original_text: str = ""
    english_text: str = ""
    status: str = "pending"  # skipped_english, success, failed, insufficient_text


@dataclass
class EvidenceMetadata:
    """Retrieved evidence item with provenance."""
    text: str
    url: str
    title: str
    publisher: str
    date: Optional[str] = None
    tier: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "publisher": self.publisher,
            "date": self.date,
            "tier": self.tier,
        }


@dataclass
class NLIMetadata:
    """NLI inference relationship output."""
    label: str
    score: float
    entailment_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    neutral_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"label": self.label, "score": round(self.score, 4)}
        if self.entailment_score is not None:
            d["entailment_score"] = round(self.entailment_score, 4)
        if self.contradiction_score is not None:
            d["contradiction_score"] = round(self.contradiction_score, 4)
        if self.neutral_score is not None:
            d["neutral_score"] = round(self.neutral_score, 4)
        return d


@dataclass
class PipelineClaimResult:
    """Structured verification output for a single extracted claim."""
    claim_id: str
    claim: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    nli: List[Dict[str, Any]] = field(default_factory=list)
    support_score: float = 0.0
    contradiction_score: float = 0.0
    temporal_flags: List[str] = field(default_factory=list)
    verdict: str = VERDICT_UNVERIFIED
    verification_confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "evidence": self.evidence,
            "nli": self.nli,
            "support_score": round(self.support_score, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "temporal_flags": self.temporal_flags,
            "verdict": self.verdict,
            "verification_confidence": round(self.verification_confidence, 4),
            "explanation": self.explanation,
        }


@dataclass
class PipelineResult:
    """Master structured verification result satisfying Phase 9 Section 15 schema."""
    input_type: str  # text, image, url
    raw_input: str

    ocr: OCRMetadata = field(default_factory=OCRMetadata)
    language: LanguageMetadata = field(default_factory=LanguageMetadata)
    translation: TranslationMetadata = field(default_factory=TranslationMetadata)

    claims: List[PipelineClaimResult] = field(default_factory=list)

    overall_status: str = VERDICT_UNVERIFIED
    pipeline_status: str = "SUCCESS"  # SUCCESS, FAILED
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Section 15 standard dictionary."""
        ocr_dict = {
            "text": self.ocr.text,
            "quality": self.ocr.quality,
            "status": self.ocr.status,
        }

        lang_dict = {
            "code": self.language.code,
            "name": self.language.name,
            "status": self.language.status,
        }
        if self.language.confidence is not None:
            lang_dict["confidence"] = round(self.language.confidence, 4)

        trans_dict = {
            "original_text": self.translation.original_text,
            "english_text": self.translation.english_text,
            "status": self.translation.status,
        }

        claims_list = [c.to_dict() for c in self.claims]

        res = {
            "input_type": self.input_type,
            "raw_input": self.raw_input,
            "ocr": ocr_dict,
            "language": lang_dict,
            "translation": trans_dict,
            "claims": claims_list,
            "overall_status": self.overall_status,
            "pipeline_status": self.pipeline_status,
        }
        if self.error_message:
            res["error_message"] = self.error_message
        res["execution_time_seconds"] = round(self.execution_time_seconds, 4)
        return res

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
