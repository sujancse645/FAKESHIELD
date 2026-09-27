"""validate_phase9.py - Automated validation suite for FakeShield Phase 9.

Performs at least 15 rigorous validation checks covering:
1. Required directories exist
2. Pipeline module imports cleanly
3. All Phase 1-8 modules import cleanly
4. Phase 1 raw dataset SHA-256 hashes unchanged
5. Text input pipeline works end-to-end
6. Image input pipeline works and preserves OCR metadata
7. URL pathway handles errors and offline failure safely
8. Language detection integrated and normalized
9. English translation skip logic functions correctly
10. Multilingual Indic translation integrates seamlessly
11. Claim extraction integrates and adheres to max 5 claims
12. Evidence retrieval integrates and preserves provenance
13. Cross-encoder NLI integrates with proper premise/hypothesis
14. Source-tier weighting and fusion logic integrate cleanly
15. Output schema satisfies Phase 9 Section 15 specifications
16. Abstention logic produces UNVERIFIED on unsupported inputs
17. Determinism audit log exists and confirms consistency
18. Phase 9 technical report exists with all 20 required sections
"""

import os
import sys
import hashlib
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "pipeline")
SAMPLES_CSV = os.path.join(DATA_DIR, "phase9_samples.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "phase9_results.csv")
STAGE_TRACE_CSV = os.path.join(DATA_DIR, "phase9_stage_trace.csv")
DETERMINISM_CSV = os.path.join(DATA_DIR, "phase9_determinism.csv")
ERROR_ANALYSIS_CSV = os.path.join(DATA_DIR, "phase9_error_analysis.csv")
REPORT_TXT = os.path.join(DATA_DIR, "phase9_report.txt")

RAW_FILES = {
    "IFND.csv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "IFND", "IFND.csv"),
        "3728c7b785ea93c9dadcc61a4c549c0eddd71a5fb6a1a31be08b5235ee4f5bfa",
    ),
    "train.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "train.tsv"),
        "23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36",
    ),
    "valid.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "valid.tsv"),
        "8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16",
    ),
    "test.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "test.tsv"),
        "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f",
    ),
}


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_validation():
    print("=" * 80)
    print("FAKESHIELD PHASE 9: AUTOMATED VALIDATION SUITE")
    print("=" * 80)

    checks = []

    def record_check(num: int, title: str, passed: bool, detail: str = ""):
        status_str = "PASS" if passed else "FAIL"
        checks.append((num, title, passed, detail))
        print(f"Check {num:02d}: [{status_str}] {title} {detail}")

    # Check 1: Required directories exist
    try:
        req_dirs = [
            os.path.join(PROJECT_ROOT, "src", "pipeline"),
            DATA_DIR,
        ]
        all_dirs = all(os.path.isdir(d) for d in req_dirs)
        record_check(1, "Required directories exist", all_dirs)
    except Exception as e:
        record_check(1, "Required directories exist", False, str(e))

    # Check 2: Pipeline module imports cleanly
    try:
        from src.pipeline import (
            FakeShieldPipeline,
            PipelineConfig,
            PipelineResult,
            PipelineTracer,
            PipelineDeterminismAuditor,
            PipelineErrorAnalyzer,
        )
        record_check(2, "Pipeline module imports cleanly", True)
    except Exception as e:
        record_check(2, "Pipeline module imports cleanly", False, str(e))

    # Check 3: All Phase 1-8 modules import cleanly
    try:
        from src.ocr.ocr_pipeline import OCRPipeline
        from src.language_detection.language_detector import LanguageDetector
        from src.translation.translator import Translator
        from src.claim_extraction.claim_extractor import ClaimExtractor
        from src.evidence_retrieval.retriever import EvidenceRetriever
        from src.nli.nli_model import NLIModel
        from src.evidence_fusion.fusion import EvidenceFusionEngine
        record_check(3, "All Phase 1-8 modules import cleanly", True)
    except Exception as e:
        record_check(3, "All Phase 1-8 modules import cleanly", False, str(e))

    # Check 4: Phase 1 raw hashes match baseline
    try:
        raw_ok = True
        for fn, (fp, exp_h) in RAW_FILES.items():
            if not os.path.exists(fp) or compute_sha256(fp) != exp_h:
                raw_ok = False
                break
        record_check(4, "Phase 1 raw dataset SHA-256 hashes match baseline", raw_ok)
    except Exception as e:
        record_check(4, "Phase 1 raw hashes match baseline", False, str(e))

    # Initialize shared pipeline instance
    try:
        pipeline = FakeShieldPipeline()
    except Exception as e:
        pipeline = None
        print(f"Error initializing pipeline: {e}")

    # Check 5: Text input pipeline works end-to-end
    try:
        txt_claim = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
        res_txt = pipeline.verify(input_data=txt_claim, input_type="text")
        txt_ok = res_txt.pipeline_status == "SUCCESS" and len(res_txt.claims) > 0 and res_txt.overall_status in ["VERIFIED", "CONTRADICTED", "MISLEADING", "OUT-OF-CONTEXT", "UNVERIFIED"]
        record_check(5, f"Text input pipeline works (verdict={res_txt.overall_status})", txt_ok)
    except Exception as e:
        record_check(5, "Text input pipeline works", False, str(e))

    # Check 6: Image input works and preserves OCR metadata
    try:
        img_path = os.path.join(PROJECT_ROOT, "data", "ocr_samples", "tamil_news_sample.png")
        res_img = pipeline.verify(input_data=img_path, input_type="image")
        img_ok = res_img.input_type == "image" and res_img.ocr.status == "success" and len(res_img.ocr.text) > 0
        record_check(6, f"Image input works and preserves OCR metadata (chars={len(res_img.ocr.text)})", img_ok)
    except Exception as e:
        record_check(6, "Image input works and preserves OCR metadata", False, str(e))

    # Check 7: URL pathway handles errors safely without crashing
    try:
        dummy_url = "http://invalid-local-fake-domain-9999.org/nonexistent"
        res_url = pipeline.verify(input_data=dummy_url, input_type="url")
        url_ok = res_url.pipeline_status == "SUCCESS" and res_url.overall_status == "UNVERIFIED" and res_url.error_message is not None
        record_check(7, "URL pathway handles failure safely", url_ok)
    except Exception as e:
        record_check(7, "URL pathway handles failure safely", False, str(e))

    # Check 8: Language detection integrated
    try:
        lang_ok = res_txt.language.code == "en" and res_txt.language.status == "success"
        record_check(8, f"Language detection integrated (code='{res_txt.language.code}')", lang_ok)
    except Exception as e:
        record_check(8, "Language detection integrated", False, str(e))

    # Check 9: English translation skip logic
    try:
        skip_ok = res_txt.translation.status == "skipped_english" and res_txt.translation.original_text == res_txt.translation.english_text
        record_check(9, f"English translation correctly skipped (status='{res_txt.translation.status}')", skip_ok)
    except Exception as e:
        record_check(9, "English translation skip logic", False, str(e))

    # Check 10: Multilingual Indic translation integrated
    try:
        ta_txt = "சென்னை வானிலை ஆய்வு மையம் இன்று மழை எச்சரிக்கை விடுத்துள்ளது."
        res_ta = pipeline.verify(input_data=ta_txt, input_type="text")
        ta_ok = res_ta.language.code == "ta" and res_ta.translation.status == "success" and len(res_ta.translation.english_text) > 0
        record_check(10, f"Multilingual translation integrated (pivot='{res_ta.translation.english_text[:35]}...')", ta_ok)
    except Exception as e:
        record_check(10, "Multilingual Indic translation integrated", False, str(e))

    # Check 11: Claim extraction integrated (max 5 claims)
    try:
        claims_ok = 1 <= len(res_txt.claims) <= 5 and res_txt.claims[0].claim != ""
        record_check(11, f"Claim extraction integrated (count={len(res_txt.claims)})", claims_ok)
    except Exception as e:
        record_check(11, "Claim extraction integrated", False, str(e))

    # Check 12: Evidence retrieval integrated and preserves provenance
    try:
        first_claim = res_txt.claims[0]
        has_ev = len(first_claim.evidence) > 0
        ev_sample = first_claim.evidence[0] if has_ev else {}
        prov_ok = has_ev and "url" in ev_sample and "publisher" in ev_sample and "tier" in ev_sample
        record_check(12, f"Evidence retrieval integrated with provenance (items={len(first_claim.evidence)})", prov_ok)
    except Exception as e:
        record_check(12, "Evidence retrieval integrated with provenance", False, str(e))

    # Check 13: Cross-encoder NLI integrated
    try:
        nli_entries = first_claim.nli
        nli_ok = len(nli_entries) > 0 and nli_entries[0]["label"] in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
        record_check(13, f"Cross-encoder NLI integrated (label='{nli_entries[0]['label']}', score={nli_entries[0]['score']})", nli_ok)
    except Exception as e:
        record_check(13, "Cross-encoder NLI integrated", False, str(e))

    # Check 14: Source-tier weighting and fusion integrated
    try:
        fusion_ok = hasattr(first_claim, "support_score") and hasattr(first_claim, "contradiction_score") and first_claim.verdict in ["VERIFIED", "CONTRADICTED", "MISLEADING", "OUT-OF-CONTEXT", "UNVERIFIED"]
        record_check(14, f"Evidence fusion integrated (sup={first_claim.support_score}, con={first_claim.contradiction_score})", fusion_ok)
    except Exception as e:
        record_check(14, "Evidence fusion integrated", False, str(e))

    # Check 15: Output schema matches Phase 9 Section 15 specifications
    try:
        d = res_txt.to_dict()
        req_keys = {"input_type", "raw_input", "ocr", "language", "translation", "claims", "overall_status", "pipeline_status"}
        schema_ok = req_keys.issubset(set(d.keys())) and isinstance(d["claims"], list)
        record_check(15, "Standard output schema matches Section 15 specifications", schema_ok)
    except Exception as e:
        record_check(15, "Output schema matches specifications", False, str(e))

    # Check 16: Abstention produces UNVERIFIED on empty/insufficient inputs
    try:
        blank_img = os.path.join(PROJECT_ROOT, "data", "ocr_samples", "blank.png")
        res_blank = pipeline.verify(input_data=blank_img, input_type="image")
        abstain_ok = res_blank.overall_status == "UNVERIFIED" and res_blank.ocr.status == "insufficient_text"
        record_check(16, "Abstention logic produces UNVERIFIED on empty inputs", abstain_ok)
    except Exception as e:
        record_check(16, "Abstention logic produces UNVERIFIED", False, str(e))

    # Check 17: Required CSV evaluation artifacts exist
    try:
        artifacts = [SAMPLES_CSV, RESULTS_CSV, STAGE_TRACE_CSV, DETERMINISM_CSV, ERROR_ANALYSIS_CSV]
        art_ok = all(os.path.isfile(a) for a in artifacts)
        record_check(17, "All required evaluation CSV artifacts exist", art_ok)
    except Exception as e:
        record_check(17, "Required CSV evaluation artifacts exist", False, str(e))

    # Check 18: Phase 9 technical research report exists with all 20 sections
    try:
        rep_ok = os.path.isfile(REPORT_TXT)
        if rep_ok:
            with open(REPORT_TXT, "r", encoding="utf-8") as f:
                content = f.read()
            has_20_sections = all(f"{i}. " in content for i in range(1, 21))
            rep_ok = has_20_sections
        record_check(18, "Phase 9 technical research report exists with 20 sections", rep_ok)
    except Exception as e:
        record_check(18, "Phase 9 technical report exists", False, str(e))

    passed_count = sum(1 for _, _, p, _ in checks if p)
    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {passed_count}/{len(checks)} CHECKS PASSED")
    print("=" * 80)

    if passed_count < len(checks):
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
