"""pipeline.py - Core Orchestrator for FakeShield Multimodal News Verification System.

Integrates:
- OCR Pipeline (Phase 3 & 4)
- Language Detection (Phase 5)
- Multilingual Translation & English Pivot (Phase 6)
- Claim Extraction & Evidence Retrieval (Phase 7)
- Source Reliability Weighting (Phase 7 & 8)
- DeBERTa Cross-Encoder NLI (Phase 8)
- Evidence Fusion, Abstention & Out-of-Context Reasoning (Phase 8)
"""

import os
import re
import html
import time
import urllib.request
from typing import Dict, Any, List, Optional, Tuple

from src.ocr.ocr_pipeline import OCRPipeline
from src.language_detection.language_detector import LanguageDetector
from src.translation.translator import Translator
from src.claim_extraction.claim_extractor import ClaimExtractor
from src.evidence_retrieval.retriever import EvidenceRetriever
from src.evidence_retrieval.query_builder import QueryBuilder
from src.nli.nli_model import NLIModel
from src.evidence_fusion.fusion import (
    EvidenceFusionEngine,
    VERDICT_VERIFIED,
    VERDICT_CONTRADICTED,
    VERDICT_MISLEADING,
    VERDICT_OUT_OF_CONTEXT,
    VERDICT_UNVERIFIED,
)
from src.pipeline.pipeline_config import PipelineConfig
from src.pipeline.pipeline_result import (
    PipelineResult,
    PipelineClaimResult,
    OCRMetadata,
    LanguageMetadata,
    TranslationMetadata,
)
from src.pipeline.pipeline_quality import PipelineTracer


class FakeShieldPipeline:
    """Production-grade, research-verified Multimodal News Verification Pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        nli_model: Optional[NLIModel] = None,
        translator: Optional[Translator] = None,
    ):
        self.config = config or PipelineConfig()

        # Shared/cached models to minimize CPU initialization overhead
        self.ocr_pipeline = OCRPipeline(
            preprocessing_config=self.config.preprocessing_config,
            ocr_config=self.config.ocr_config,
        )
        self.language_detector = LanguageDetector()
        self.translator = translator or Translator(self.config.translation_config)
        self.claim_extractor = ClaimExtractor(max_claims=self.config.max_claims)
        self.query_builder = QueryBuilder()
        self.retriever = EvidenceRetriever()

        # Phase 8 Reasoning Components
        self.nli_model = nli_model or NLIModel(self.config.nli_config)
        self.fusion_engine = EvidenceFusionEngine(
            config=self.config.fusion_config,
            nli_model=self.nli_model,
        )

    def verify(
        self,
        input_data: str,
        input_type: str = "text",
        case_id: str = "case_01",
        tracer: Optional[PipelineTracer] = None,
    ) -> PipelineResult:
        """Run full end-to-end verification pipeline.

        Args:
            input_data: Plain text, image file path, or URL.
            input_type: One of 'text', 'image', 'url'.
            case_id: Tracking identifier for logging.
            tracer: Optional telemetry tracer for execution audits.

        Returns:
            PipelineResult conforming to Phase 9 Section 15 schema.
        """
        t_start = time.time()
        input_type = (input_type or "text").lower().strip()
        if input_type not in self.config.supported_input_types:
            input_type = "text"

        res = PipelineResult(
            input_type=input_type,
            raw_input=input_data,
        )

        try:
            # ----------------------------------------------------
            # STAGE 1: INGESTION & TEXT ACQUISITION
            # ----------------------------------------------------
            raw_text = ""
            if input_type == "text":
                if tracer:
                    tracer.start_stage("ingestion")
                raw_text = (input_data or "").strip()
                res.ocr.status = "skipped"
                if tracer:
                    tracer.end_stage("ingestion", "SUCCESS", f"Text input length: {len(raw_text)}")

            elif input_type == "image":
                if tracer:
                    tracer.start_stage("ocr")
                image_path = input_data.strip()
                if not os.path.exists(image_path):
                    res.ocr.status = "failed"
                    res.ocr.text = ""
                    res.overall_status = VERDICT_UNVERIFIED
                    res.pipeline_status = "SUCCESS"
                    res.error_message = f"Image file not found: {image_path}"
                    if tracer:
                        tracer.end_stage("ocr", "FAILED", "File not found", res.error_message)
                    res.execution_time_seconds = time.time() - t_start
                    return res

                ocr_dict = self.ocr_pipeline.process_image(image_path)
                ocr_text = ocr_dict.get("ocr_text_clean", "").strip()
                res.ocr.text = ocr_text
                res.ocr.quality = {
                    "ocr_confidence_mean": ocr_dict.get("ocr_confidence_mean", 0.0),
                    "image_width": ocr_dict.get("image_width", 0),
                    "image_height": ocr_dict.get("image_height", 0),
                    "preprocessing": ocr_dict.get("preprocessing", {}),
                }

                if len(ocr_text) < 3:
                    res.ocr.status = "insufficient_text"
                    res.overall_status = VERDICT_UNVERIFIED
                    res.pipeline_status = "SUCCESS"
                    res.error_message = "OCR produced empty or unreadable text. Cannot extract factual claims."
                    if tracer:
                        tracer.end_stage("ocr", "ABSTAINED", "Insufficient OCR text")
                    res.execution_time_seconds = time.time() - t_start
                    return res

                res.ocr.status = "success"
                raw_text = ocr_text
                if tracer:
                    tracer.end_stage("ocr", "SUCCESS", f"OCR extracted {len(raw_text)} chars")

            elif input_type == "url":
                if tracer:
                    tracer.start_stage("url_fetch")
                url = input_data.strip()
                res.ocr.status = "skipped"
                fetched_text, url_status, url_err = self._fetch_url_content(url)
                if url_status != "success":
                    res.overall_status = VERDICT_UNVERIFIED
                    res.pipeline_status = "SUCCESS"
                    res.error_message = f"URL acquisition failed ({url_status}): {url_err.get('error', 'Unknown error')}"
                    if tracer:
                        tracer.end_stage("url_fetch", "FAILED", url_status, res.error_message)
                    res.execution_time_seconds = time.time() - t_start
                    return res

                raw_text = fetched_text
                if tracer:
                    tracer.end_stage("url_fetch", "SUCCESS", f"Fetched {len(raw_text)} chars from URL")

            # Validate usable text
            if len(raw_text) < 3:
                res.language.status = "insufficient_text"
                res.overall_status = VERDICT_UNVERIFIED
                res.pipeline_status = "SUCCESS"
                res.error_message = "Input contains insufficient text for factual verification."
                res.execution_time_seconds = time.time() - t_start
                return res

            # ----------------------------------------------------
            # STAGE 2: LANGUAGE IDENTIFICATION (Phase 5)
            # ----------------------------------------------------
            if tracer:
                tracer.start_stage("language_detection")
            lang_res = self.language_detector.detect(raw_text)
            res.language = LanguageMetadata(
                code=lang_res.language_code,
                name=lang_res.language_name,
                confidence=lang_res.confidence,
                status=lang_res.status,
            )
            if tracer:
                tracer.end_stage(
                    "language_detection",
                    "SUCCESS" if lang_res.status == "success" else "FAILED",
                    f"{lang_res.language_name} ({lang_res.language_code})",
                )

            if lang_res.status == "insufficient_text" or not lang_res.language_code:
                res.overall_status = VERDICT_UNVERIFIED
                res.pipeline_status = "SUCCESS"
                res.error_message = "Language detection could not identify supported language (insufficient script tokens)."
                res.execution_time_seconds = time.time() - t_start
                return res

            # ----------------------------------------------------
            # STAGE 3: TRANSLATION & ENGLISH PIVOT (Phase 6)
            # ----------------------------------------------------
            if tracer:
                tracer.start_stage("translation")
            lang_code = lang_res.language_code
            if lang_code == "en":
                res.translation = TranslationMetadata(
                    original_text=raw_text,
                    english_text=raw_text,
                    status="skipped_english",
                )
                english_pivot = raw_text
                if tracer:
                    tracer.end_stage("translation", "SKIPPED", "English input; translation skipped")
            else:
                trans_res = self.translator.translate(raw_text, source_language=lang_code)
                english_text = trans_res.get("english_pivot", "")
                trans_status = trans_res.get("status", "failed")
                res.translation = TranslationMetadata(
                    original_text=raw_text,
                    english_text=english_text,
                    status=trans_status,
                )
                english_pivot = english_text
                if tracer:
                    tracer.end_stage("translation", "SUCCESS", f"Translated {lang_code}->en: {english_pivot[:60]}...")

            if not english_pivot or len(english_pivot.strip()) < 3:
                res.overall_status = VERDICT_UNVERIFIED
                res.pipeline_status = "SUCCESS"
                res.error_message = "Translation failed to produce viable English pivot text."
                res.execution_time_seconds = time.time() - t_start
                return res

            # ----------------------------------------------------
            # STAGE 4: CLAIM EXTRACTION (Phase 7)
            # ----------------------------------------------------
            if tracer:
                tracer.start_stage("claim_extraction")
            claim_set = self.claim_extractor.extract_claims(
                english_pivot, input_id=case_id, source_language=lang_code
            )
            extracted_claims = list(claim_set.claims)

            # Fallback: if text is a single short sentence that didn't hit a specific pattern,
            # use the sentence directly as a candidate claim rather than dropping it.
            if not extracted_claims and len(english_pivot.split()) >= self.config.min_claim_words:
                cleaned_cand = english_pivot.strip()
                if not re.search(r"\?$", cleaned_cand):
                    from src.claim_extraction.claim_extractor import ExtractedClaim
                    query = self.query_builder.build_query(cleaned_cand)
                    extracted_claims = [
                        ExtractedClaim(
                            claim_id=f"{case_id}_c01",
                            claim=cleaned_cand,
                            type="entity_fact",
                            source_sentence=cleaned_cand,
                            search_query=query,
                            confidence=None,
                        )
                    ]

            if tracer:
                tracer.end_stage(
                    "claim_extraction",
                    "SUCCESS" if extracted_claims else "ABSTAINED",
                    f"Extracted {len(extracted_claims)} claim(s)",
                )

            if not extracted_claims:
                res.overall_status = VERDICT_UNVERIFIED
                res.pipeline_status = "SUCCESS"
                res.error_message = "No checkable factual claims identified in input."
                res.execution_time_seconds = time.time() - t_start
                return res

            # ----------------------------------------------------
            # STAGES 5-8: RETRIEVAL, NLI, FUSION & VERDICTS
            # ----------------------------------------------------
            claim_results: List[PipelineClaimResult] = []

            for c_obj in extracted_claims:
                cid = c_obj.claim_id
                c_text = c_obj.claim

                # 5. Query Builder & Retrieval
                if tracer:
                    tracer.start_stage(f"retrieval_{cid}")
                query = c_obj.search_query or self.query_builder.build_query(c_text)
                ret_out = self.retriever.retrieve(
                    query=query, claim_id=cid, claim_text=c_text
                )
                evidence_items = ret_out.get("items", [])[: self.config.max_evidence_per_claim]

                if tracer:
                    tracer.end_stage(
                        f"retrieval_{cid}",
                        "SUCCESS" if evidence_items else "NO_RESULTS",
                        f"Retrieved {len(evidence_items)} items for query: '{query}'",
                    )

                # Prepare raw evidence dicts for fusion
                raw_ev_dicts = []
                for ev in evidence_items:
                    ev_d = ev.to_dict() if hasattr(ev, "to_dict") else ev
                    # Map field aliases if needed
                    raw_ev_dicts.append({
                        "text": ev_d.get("evidence_text", ev_d.get("text", "")),
                        "url": ev_d.get("url", ""),
                        "title": ev_d.get("title", ""),
                        "publisher": ev_d.get("publisher", ""),
                        "date": ev_d.get("date"),
                        "tier": ev_d.get("tier", 4),
                    })

                # 6 & 7 & 8. NLI + Evidence Fusion + Reasoning (Phase 8)
                if tracer:
                    tracer.start_stage(f"reasoning_{cid}")
                reasoning_res = self.fusion_engine.fuse(
                    claim=c_text,
                    evidence_items=raw_ev_dicts,
                    claim_id=cid,
                )
                if tracer:
                    tracer.end_stage(
                        f"reasoning_{cid}",
                        "SUCCESS",
                        f"Verdict: {reasoning_res.verdict} (conf: {reasoning_res.confidence})",
                    )

                # Extract NLI relationships
                nli_list = []
                for sc in reasoning_res.evidence:
                    nli_list.append({
                        "label": sc.nli_label,
                        "score": sc.nli_score,
                    })

                p_claim = PipelineClaimResult(
                    claim_id=cid,
                    claim=c_text,
                    evidence=raw_ev_dicts,
                    nli=nli_list,
                    support_score=reasoning_res.support_score,
                    contradiction_score=reasoning_res.contradiction_score,
                    temporal_flags=reasoning_res.temporal_flags,
                    verdict=reasoning_res.verdict,
                    verification_confidence=reasoning_res.confidence,
                    explanation=reasoning_res.explanation,
                )
                claim_results.append(p_claim)

            res.claims = claim_results

            # Determine aggregate overall status
            res.overall_status = self._aggregate_overall_status(claim_results)
            res.pipeline_status = "SUCCESS"

        except Exception as e:
            res.pipeline_status = "FAILED"
            res.overall_status = VERDICT_UNVERIFIED
            res.error_message = f"Pipeline execution error: {str(e)}"
            if tracer:
                tracer.end_stage("pipeline", "FAILED", "Unhandled error", str(e))

        res.execution_time_seconds = time.time() - t_start
        return res

    def _fetch_url_content(self, url: str) -> Tuple[str, str, Dict[str, Any]]:
        """Safely fetch and extract clean text from a web URL.

        Returns (clean_text, status, error_details).
        """
        if not url or not url.startswith(("http://", "https://")):
            return "", "invalid_url", {"error": "URL must start with http:// or https://"}

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FakeShield-Verifier/1.0 (Research Pipeline)"},
            )
            with urllib.request.urlopen(req, timeout=self.config.url_request_timeout) as resp:
                raw_bytes = resp.read(self.config.url_max_chars * 2)
                raw_html = raw_bytes.decode("utf-8", errors="replace")

                # Clean markup
                text = re.sub(r"<(script|style).*?>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = html.unescape(text)
                text = re.sub(r"\s+", " ", text).strip()

                if len(text) < 15:
                    return text, "empty_content", {"error": "Extracted text content too short or empty"}
                return text, "success", {}
        except Exception as e:
            return "", "fetch_failed", {"error": str(e)}

    @staticmethod
    def _aggregate_overall_status(claims: List[PipelineClaimResult]) -> str:
        """Deterministically synthesize a single overall verification verdict across all claims."""
        if not claims:
            return VERDICT_UNVERIFIED

        verdicts = [c.verdict for c in claims]

        # If any claim is explicitly contradicted, the overall input is flagged
        if VERDICT_CONTRADICTED in verdicts:
            return VERDICT_CONTRADICTED
        if VERDICT_MISLEADING in verdicts:
            return VERDICT_MISLEADING
        if VERDICT_OUT_OF_CONTEXT in verdicts:
            return VERDICT_OUT_OF_CONTEXT

        # If all extracted claims are verified, the input is verified
        if all(v == VERDICT_VERIFIED for v in verdicts):
            return VERDICT_VERIFIED

        # If some verified and some unverified, be conservative
        if VERDICT_VERIFIED in verdicts and VERDICT_UNVERIFIED in verdicts:
            return VERDICT_UNVERIFIED  # Not all claims are substantiated

        return VERDICT_UNVERIFIED
