"""validate_translation.py - Automated validation suite for FakeShield Phase 6 Translation.

Performs 15 rigorous verification checks across module imports, pipeline integrity,
English skip logic, edge-case guards, evaluation artifacts, determinism, and Phase 1 hashes.
"""

import os
import sys
import hashlib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "translation")
SAMPLES_PATH = os.path.join(DATA_DIR, "translation_samples.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "translation_results.csv")
DETERMINISM_PATH = os.path.join(DATA_DIR, "translation_determinism.csv")
METRICS_PATH = os.path.join(DATA_DIR, "translation_metrics.csv")
ERROR_ANALYSIS_PATH = os.path.join(DATA_DIR, "translation_error_analysis.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "translation_report.txt")

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
    """Execute all 15 validation checks."""
    print("=" * 80)
    print("FAKESHIELD PHASE 6: AUTOMATED VALIDATION SUITE")
    print("=" * 80)

    checks = []

    def record_check(num: int, title: str, passed: bool, detail: str = ""):
        status_str = "PASS" if passed else "FAIL"
        checks.append((num, title, passed, detail))
        print(f"Check {num:02d}: [{status_str}] {title} {detail}")

    # Check 1: Translation module imports cleanly
    try:
        from src.translation import (
            Translator,
            TranslationConfig,
            TranslationQualityAnalyzer,
            translate_to_english,
        )
        record_check(1, "Translation module imports cleanly", True)
    except Exception as e:
        record_check(1, "Translation module imports cleanly", False, str(e))

    # Check 2: Translator can be instantiated
    try:
        t = Translator()
        record_check(2, "Translator instantiates cleanly", True)
    except Exception as e:
        record_check(2, "Translator instantiates cleanly", False, str(e))

    # Check 3: Translation configuration has target 'en' and valid parameters
    try:
        cfg = TranslationConfig()
        is_valid_cfg = (cfg.target_language == "en" and cfg.target_tag == "eng_Latn" and cfg.num_beams >= 1)
        record_check(3, "TranslationConfig valid (target='en')", is_valid_cfg)
    except Exception as e:
        record_check(3, "TranslationConfig valid (target='en')", False, str(e))

    # Check 4: TranslationQualityAnalyzer functions correctly
    try:
        b = TranslationQualityAnalyzer.compute_sentence_bleu("hello world", "hello world")
        c = TranslationQualityAnalyzer.compute_sentence_chrf("hello world", "hello world")
        np = TranslationQualityAnalyzer.check_number_preservation("There are 10 cars", "I see 10 cars")
        neg = TranslationQualityAnalyzer.check_negation_preservation("There is no doubt", "There is no doubt")
        analyzer_ok = (b > 90.0 and c > 90.0 and np["preserved"] and neg["preserved"])
        record_check(4, "TranslationQualityAnalyzer computes BLEU, chrF, numbers, negation", analyzer_ok)
    except Exception as e:
        record_check(4, "TranslationQualityAnalyzer computes BLEU, chrF, numbers, negation", False, str(e))

    # Check 5: Evaluation dataset exists and has >= 70 samples
    try:
        exists_s = os.path.exists(SAMPLES_PATH)
        df_samples = pd.read_csv(SAMPLES_PATH) if exists_s else pd.DataFrame()
        count_ok = len(df_samples) >= 70
        record_check(5, f"Evaluation samples exist ({len(df_samples)} >= 70)", exists_s and count_ok)
    except Exception as e:
        record_check(5, "Evaluation samples exist", False, str(e))

    # Check 6: Translation results exist with matching sample count
    try:
        exists_r = os.path.exists(RESULTS_PATH)
        df_results = pd.read_csv(RESULTS_PATH) if exists_r else pd.DataFrame()
        count_match = len(df_results) == len(df_samples)
        record_check(6, f"Results exist and match sample count ({len(df_results)})", exists_r and count_match)
    except Exception as e:
        record_check(6, "Results exist and match sample count", False, str(e))

    # Check 7: Required result columns exist
    try:
        required_cols = {
            "sample_id", "source_language", "source_text", "english_pivot",
            "reference_english", "status", "bleu", "chrf", "number_status", "negation_status"
        }
        cols_present = required_cols.issubset(set(df_results.columns))
        record_check(7, "Required columns present in results CSV", cols_present)
    except Exception as e:
        record_check(7, "Required columns present in results CSV", False, str(e))

    # Check 8: English skip logic operates with 100% fidelity
    try:
        en_rows = df_results[df_results["source_language"] == "en"]
        en_skipped = (en_rows["status"] == "skipped").all()
        en_exact = (en_rows["english_pivot"] == en_rows["normalized_text"]).all()
        record_check(8, f"English skip logic: 100% skipped with exact identity ({len(en_rows)} samples)", en_skipped and en_exact)
    except Exception as e:
        record_check(8, "English skip logic operates with 100% fidelity", False, str(e))

    # Check 9: Edge cases correctly identified as insufficient_text / unsupported_language
    try:
        edge_rows = df_results[df_results["category"] == "edge_case"]
        insufficient_rows = edge_rows[edge_rows["sample_id"] != "trans_edge_unsupported"]
        unsupported_rows = edge_rows[edge_rows["sample_id"] == "trans_edge_unsupported"]
        insufficient_ok = (insufficient_rows["status"] == "insufficient_text").all()
        unsupported_ok = (unsupported_rows["status"] == "unsupported_language").all()
        edge_ok = insufficient_ok and unsupported_ok
        record_check(9, f"Edge cases intercepted ({len(insufficient_rows)} insufficient, {len(unsupported_rows)} unsupported)", edge_ok)
    except Exception as e:
        record_check(9, "Edge cases correctly intercepted", False, str(e))

    # Check 10: Non-edge Indic translations produced valid English output
    try:
        indic_rows = df_results[(df_results["source_language"] != "en") & (df_results["category"] != "edge_case")]
        indic_success = (indic_rows["status"] == "success").all()
        indic_nonempty = (indic_rows["english_pivot"].str.len() > 0).all()
        record_check(10, f"Indic translations successful ({len(indic_rows)} samples)", indic_success and indic_nonempty)
    except Exception as e:
        record_check(10, "Non-edge Indic translations produced valid English output", False, str(e))

    # Check 11: Translation metrics file exists and scores are valid
    try:
        exists_m = os.path.exists(METRICS_PATH)
        df_metrics = pd.read_csv(METRICS_PATH) if exists_m else pd.DataFrame()
        metrics_valid = len(df_metrics) > 0 and (df_metrics["mean_bleu"] >= 0).all()
        record_check(11, f"Translation metrics valid ({len(df_metrics)} languages recorded)", exists_m and metrics_valid)
    except Exception as e:
        record_check(11, "Translation metrics valid", False, str(e))

    # Check 12: Determinism file exists and confirms 100% reproducibility
    try:
        exists_d = os.path.exists(DETERMINISM_PATH)
        df_det = pd.read_csv(DETERMINISM_PATH) if exists_d else pd.DataFrame()
        det_ok = len(df_det) >= 5 and df_det["deterministic"].all()
        record_check(12, f"Determinism confirmed 100% ({len(df_det)}/ {len(df_det)} matches)", exists_d and det_ok)
    except Exception as e:
        record_check(12, "Determinism file exists and confirms 100% reproducibility", False, str(e))

    # Check 13: Error analysis file exists
    try:
        exists_err = os.path.exists(ERROR_ANALYSIS_PATH)
        record_check(13, "Error analysis CSV exists", exists_err)
    except Exception as e:
        record_check(13, "Error analysis CSV exists", False, str(e))

    # Check 14: Translation report exists and has all 18 sections
    try:
        exists_rep = os.path.exists(REPORT_PATH)
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            rep_text = f.read()
        all_sections = all(f"{i}. " in rep_text for i in range(1, 19))
        record_check(14, f"Translation report contains all 18 required sections", exists_rep and all_sections)
    except Exception as e:
        record_check(14, "Translation report contains all 18 required sections", False, str(e))

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

    if passed_count == total_count:
        print("\nALL PHASE 6 VALIDATION CHECKS PASSED SUCCESSFULLY.")
        return 0
    else:
        print(f"\nVALIDATION FAILED: {total_count - passed_count} checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
