"""validate_phase7.py - Automated validation suite for FakeShield Phase 7.

Performs 15 rigorous verification checks covering:
1. Claim extraction module imports
2. Retrieval module imports
3. Source reliability module imports
4. Evaluation dataset exists (20 benchmark claims)
5. Claim schema validity
6. Claim limit enforced (<= 5 claims per input)
7. Query generation functions and is non-empty
8. Source tier YAML configuration valid
9. Evidence schema validity
10. URL traceability valid (no anonymous sources)
11. Duplicate handling and deduplication works
12. No-result handling works without errors
13. Determinism artifact valid (100% stability)
14. 18-section technical report exists
15. Phase 1 raw dataset SHA-256 hashes match baseline
"""

import os
import sys
import hashlib
import yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "evidence_retrieval")
CLAIMS_CSV = os.path.join(DATA_DIR, "evaluation_claims.csv")
CLAIM_RESULTS_CSV = os.path.join(DATA_DIR, "claim_extraction_results.csv")
RETRIEVAL_RESULTS_CSV = os.path.join(DATA_DIR, "retrieval_results.csv")
EVIDENCE_ITEMS_CSV = os.path.join(DATA_DIR, "evidence_items.csv")
SOURCE_RELIABILITY_CSV = os.path.join(DATA_DIR, "source_reliability_results.csv")
DETERMINISM_CSV = os.path.join(DATA_DIR, "retrieval_determinism.csv")
REPORT_PATH = os.path.join(DATA_DIR, "evidence_retrieval_report.txt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "source_reliability.yaml")

RAW_FILES = {
    "IFND.csv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "IFND", "IFND.csv"),
        "3728c7b785ea93c9dadcc61a4c549c0eddd71a5fb6a1a31be08b5235ee4f5bfa",
    ),
    "train.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "train.tsv"),
        "23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36",
    ),
    "test.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "test.tsv"),
        "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f",
    ),
    "valid.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "valid.tsv"),
        "8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16",
    ),
}


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_validation():
    print("=" * 80)
    print("FAKESHIELD PHASE 7: AUTOMATED VALIDATION SUITE")
    print("=" * 80)

    checks = []

    def record_check(num: int, title: str, passed: bool, detail: str = ""):
        status_str = "PASS" if passed else "FAIL"
        checks.append((num, title, passed, detail))
        print(f"Check {num:02d}: [{status_str}] {title} {detail}")

    # Check 1: Claim extraction module imports cleanly
    try:
        from src.claim_extraction import ClaimExtractor, ExtractedClaim, ClaimSet, ClaimQualityAnalyzer
        record_check(1, "Claim extraction module imports cleanly", True)
    except Exception as e:
        record_check(1, "Claim extraction module imports cleanly", False, str(e))

    # Check 2: Evidence retrieval module imports cleanly
    try:
        from src.evidence_retrieval import QueryBuilder, EvidenceRetriever, EvidenceItem, EvidenceQualityAnalyzer
        record_check(2, "Evidence retrieval module imports cleanly", True)
    except Exception as e:
        record_check(2, "Evidence retrieval module imports cleanly", False, str(e))

    # Check 3: Source reliability module imports cleanly
    try:
        from src.evidence_retrieval.source_reliability import SourceReliabilityAnalyzer
        record_check(3, "Source reliability module imports cleanly", True)
    except Exception as e:
        record_check(3, "Source reliability module imports cleanly", False, str(e))

    # Check 4: Evaluation dataset exists and has 20 samples
    try:
        exists_s = os.path.exists(CLAIMS_CSV)
        df_claims = pd.read_csv(CLAIMS_CSV) if exists_s else pd.DataFrame()
        count_ok = len(df_claims) == 20
        record_check(4, f"Evaluation dataset exists (20 samples)", exists_s and count_ok)
    except Exception as e:
        record_check(4, "Evaluation dataset exists", False, str(e))

    # Check 5: Claim schema valid in results
    try:
        df_cres = pd.read_csv(CLAIM_RESULTS_CSV)
        req_cols = {"sample_id", "claim_id", "extracted_claim", "claim_type", "search_query"}
        schema_ok = req_cols.issubset(set(df_cres.columns)) and len(df_cres) > 0
        record_check(5, f"Extracted claim schema valid ({len(df_cres)} claims)", schema_ok)
    except Exception as e:
        record_check(5, "Extracted claim schema valid", False, str(e))

    # Check 6: Claim limit enforced (<= 5 per input)
    try:
        counts = df_cres.groupby("sample_id").size()
        limit_ok = (counts <= 5).all()
        record_check(6, f"Claim limit enforced (max {counts.max()} <= 5)", limit_ok)
    except Exception as e:
        record_check(6, "Claim limit enforced", False, str(e))

    # Check 7: Query generation works and non-empty
    try:
        qb = QueryBuilder()
        q = qb.build_query("The Reserve Bank of India kept the repo rate at 6.5 percent.")
        query_ok = len(q) > 5 and "6.5" in q and "repo" in q.lower()
        record_check(7, f"Query generation functional (query='{q}')", query_ok)
    except Exception as e:
        record_check(7, "Query generation functional", False, str(e))

    # Check 8: Source tier YAML configuration valid
    try:
        exists_cfg = os.path.exists(CONFIG_PATH)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        has_tiers = all(t in cfg.get("tiers", {}) for t in [1, 2, 3, 4])
        record_check(8, "Source tier YAML configuration valid (Tiers 1-4)", exists_cfg and has_tiers)
    except Exception as e:
        record_check(8, "Source tier YAML configuration valid", False, str(e))

    # Check 9: Evidence schema valid
    try:
        df_ev = pd.read_csv(EVIDENCE_ITEMS_CSV)
        req_ev_cols = {"claim_id", "evidence_text", "url", "publisher", "tier", "retrieval_method"}
        ev_schema_ok = req_ev_cols.issubset(set(df_ev.columns)) and len(df_ev) > 0
        record_check(9, f"Evidence schema valid ({len(df_ev)} items recorded)", ev_schema_ok)
    except Exception as e:
        record_check(9, "Evidence schema valid", False, str(e))

    # Check 10: URL traceability valid (no anonymous sources)
    try:
        has_urls = (df_ev["url"].str.startswith("http")).all()
        has_pubs = (df_ev["publisher"].str.len() > 0).all()
        record_check(10, "URL and publisher traceability valid (100% attributed)", has_urls and has_pubs)
    except Exception as e:
        record_check(10, "URL traceability valid", False, str(e))

    # Check 11: Duplicate handling works
    try:
        retriever = EvidenceRetriever()
        dummy_items = [
            EvidenceItem("c1", "text", "https://example.com/1", "Pub", "Title", None, None, "news", 3, "test", "q"),
            EvidenceItem("c1", "text", "https://example.com/1", "Pub", "Title", None, None, "news", 3, "test", "q"),
        ]
        deduped = retriever.deduplicate_evidence(dummy_items)
        dedup_ok = len(deduped) == 1
        record_check(11, "Duplicate evidence handling and filtering works", dedup_ok)
    except Exception as e:
        record_check(11, "Duplicate handling works", False, str(e))

    # Check 12: No-result / api_unavailable handling works
    try:
        res = retriever.retrieve("completely_unmatched_gibberish_query_xyz123", claim_id="c_test")
        no_res_ok = res["status"] in ("no_results", "api_unavailable") and res["item_count"] == 0
        record_check(12, f"No-result retrieval state handled cleanly (status='{res['status']}')", no_res_ok)
    except Exception as e:
        record_check(12, "No-result handling works", False, str(e))

    # Check 13: Determinism artifact valid (100% stability)
    try:
        df_det = pd.read_csv(DETERMINISM_CSV)
        det_ok = len(df_det) >= 5 and df_det["deterministic"].all()
        record_check(13, f"Determinism confirmed (100% across {len(df_det)} samples)", det_ok)
    except Exception as e:
        record_check(13, "Determinism artifact valid", False, str(e))

    # Check 14: 18-section technical report exists
    try:
        exists_rep = os.path.exists(REPORT_PATH)
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        all_sec = all(f"{i}. " in content for i in range(1, 19))
        record_check(14, "Technical report exists with all 18 sections", exists_rep and all_sec)
    except Exception as e:
        record_check(14, "Technical report exists", False, str(e))

    # Check 15: Phase 1 raw file hashes unchanged
    try:
        hashes_match = True
        for name, (path, expected) in RAW_FILES.items():
            actual = compute_sha256(path)
            if actual != expected:
                hashes_match = False
                break
        record_check(15, "Phase 1 raw dataset SHA-256 hashes match baseline", hashes_match)
    except Exception as e:
        record_check(15, "Phase 1 raw dataset SHA-256 hashes match baseline", False, str(e))

    print("\n" + "=" * 80)
    passed_count = sum(1 for c in checks if c[2])
    total_count = len(checks)
    print(f"VALIDATION SUMMARY: {passed_count}/{total_count} CHECKS PASSED")
    print("=" * 80)

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(run_validation())
