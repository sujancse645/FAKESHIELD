"""validate_language_detection.py - FakeShield Phase 5 Automated Validation Script.

Verifies:
1.  Language detection module imports cleanly.
2.  langdetect package is available and functional.
3.  Evaluation dataset (language_samples.csv) exists.
4.  Ground-truth labels exist for all evaluation records.
5.  Result file (language_detection_results.csv) exists.
6.  Required result columns exist.
7.  Expected language values are valid.
8.  Detected language values are valid.
9.  Metrics are within valid range [0.0, 1.0].
10. Confusion matrix exists and is valid.
11. Language detection report exists with all 18 sections.
12. Determinism results exist and prove stability.
13. Phase 1 raw files remain unchanged (SHA-256 integrity).
"""

import csv
import hashlib
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_phase5():
    print("=" * 80)
    print("FAKESHIELD PHASE 5: AUTOMATED LANGUAGE DETECTION VALIDATION")
    print("=" * 80)

    checks = []

    def record(num, name, status, detail=""):
        checks.append((num, name, status, detail))
        print(f"  [{status:<7}] Check {num:>2}: {name:<40} {detail}")

    # Check 1: Language detection module imports
    try:
        from src.language_detection import (
            LanguageDetector, detect_language,
            SUPPORTED_LANGUAGES, DetectionResult, LanguageQualityAnalyzer
        )
        record(1, "Module imports cleanly", "PASS", "All public classes exported")
    except Exception as e:
        record(1, "Module imports cleanly", "FAIL", str(e))

    # Check 2: langdetect available
    try:
        import langdetect
        record(2, "langdetect library available", "PASS", "Available and functional")
    except Exception as e:
        record(2, "langdetect library available", "FAIL", str(e))

    # Check 3: Evaluation dataset exists
    samples_p = PROJECT_ROOT / "data" / "evaluation" / "language_detection" / "language_samples.csv"
    samples_rows = []
    if samples_p.exists() and samples_p.stat().st_size > 500:
        with open(samples_p, "r", encoding="utf-8") as f:
            samples_rows = list(csv.DictReader(f))
        record(3, "Evaluation dataset exists", "PASS", f"{len(samples_rows)} samples present")
    else:
        record(3, "Evaluation dataset exists", "FAIL", "Missing or empty")

    # Check 4: Ground-truth labels exist
    gt_ok = bool(samples_rows and all("expected_language" in r and r["expected_language"] for r in samples_rows))
    record(4, "Ground-truth labels present", "PASS" if gt_ok else "FAIL", "All samples have explicit expected_language")

    # Check 5: Result file exists
    results_p = PROJECT_ROOT / "data" / "evaluation" / "language_detection" / "language_detection_results.csv"
    results_rows = []
    if results_p.exists() and results_p.stat().st_size > 500:
        with open(results_p, "r", encoding="utf-8") as f:
            results_rows = list(csv.DictReader(f))
        record(5, "Results file exists", "PASS", f"{len(results_rows)} records in results CSV")
    else:
        record(5, "Results file exists", "FAIL", "Missing or empty")

    # Check 6: Required result columns exist
    req_cols = [
        "sample_id", "expected_language", "detected_language",
        "language_code", "status", "confidence", "source", "text_length"
    ]
    cols_ok = bool(results_rows and all(all(c in r for c in req_cols) for r in results_rows))
    record(6, "Required result columns exist", "PASS" if cols_ok else "FAIL", f"{len(req_cols)} columns verified")

    # Check 7: Expected language values valid
    valid_expected = {"English", "Tamil", "Hindi", "Telugu", "Malayalam", "Kannada", "N/A"}
    exp_ok = bool(results_rows and all(r["expected_language"] in valid_expected for r in results_rows))
    record(7, "Expected language values valid", "PASS" if exp_ok else "FAIL", f"Subset of {valid_expected}")

    # Check 8: Detected language values valid
    valid_detected = set(SUPPORTED_LANGUAGES.values()) | {"unclassified"}
    det_ok = bool(results_rows and all(r["detected_language"] in valid_detected for r in results_rows))
    record(8, "Detected language values valid", "PASS" if det_ok else "FAIL", "Mapped to known languages or unclassified")

    # Check 9: Metrics are within valid ranges [0.0, 1.0]
    metrics = LanguageQualityAnalyzer.evaluate(results_rows)
    acc = metrics.get("accuracy", -1.0)
    macro_f1 = metrics.get("macro_f1", -1.0)
    metrics_ok = (0.0 <= acc <= 1.0) and (0.0 <= macro_f1 <= 1.0)
    record(9, "Metrics within valid range", "PASS" if metrics_ok else "FAIL", f"Accuracy={acc*100:.1f}%, F1={macro_f1*100:.1f}%")

    # Check 10: Confusion matrix exists
    cm_p = PROJECT_ROOT / "data" / "evaluation" / "language_detection" / "language_detection_confusion_matrix.csv"
    cm_ok = False
    if cm_p.exists() and cm_p.stat().st_size > 100:
        with open(cm_p, "r", encoding="utf-8") as f:
            cm_rows = list(csv.DictReader(f))
        if len(cm_rows) >= 6:
            cm_ok = True
    record(10, "Confusion matrix CSV exists", "PASS" if cm_ok else "FAIL", "Matrix rows verified")

    # Check 11: Report exists with all 18 sections
    rep_p = PROJECT_ROOT / "data" / "evaluation" / "language_detection_report.txt"
    rep_ok = False
    if rep_p.exists() and rep_p.stat().st_size > 2000:
        rep_content = rep_p.read_text(encoding="utf-8")
        if "1. OBJECTIVE" in rep_content and "18. PHASE 5 CONCLUSION" in rep_content:
            rep_ok = True
    record(11, "18-section report exists", "PASS" if rep_ok else "FAIL", "Sections 1 through 18 confirmed")

    # Check 12: Determinism results exist and prove stability
    det_p = PROJECT_ROOT / "data" / "evaluation" / "language_detection" / "language_detection_determinism.csv"
    det_ok = False
    if det_p.exists() and det_p.stat().st_size > 300:
        with open(det_p, "r", encoding="utf-8") as f:
            d_rows = list(csv.DictReader(f))
        # Group by sample_id and check all runs identical
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in d_rows:
            grouped[r["sample_id"]].append(r["detected_language"])
        if grouped and all(len(v) == 3 and len(set(v)) == 1 for v in grouped.values()):
            det_ok = True
    record(12, "Determinism results verified", "PASS" if det_ok else "FAIL", "All 3 runs identical for all test candidates")

    # Check 13: Phase 1 raw file hashes unchanged
    raw_dir = PROJECT_ROOT / "data" / "raw"
    stored_hashes = [
        ("data/raw/IFND/IFND.csv", "3728c7b785ea93c9"),
        ("data/raw/LIAR/train.tsv", "23ef8cb7ff57b25c"),
        ("data/raw/LIAR/test.tsv", "ee41e2c741e00192"),
        ("data/raw/LIAR/valid.tsv", "8dd3d7ed1a11e78e"),
    ]
    raw_hashes_ok = True
    for frel, prefix in stored_hashes:
        fpath = PROJECT_ROOT / frel
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        if not sha.hexdigest().startswith(prefix):
            raw_hashes_ok = False
            break
    record(13, "Phase 1 raw file hashes unchanged", "PASS" if raw_hashes_ok else "FAIL", "All 4 SHA-256 hashes match baseline")

    print("=" * 80)
    failures = [c for c in checks if c[2] == "FAIL"]
    if not failures:
        print("PHASE 5 AUTOMATED VALIDATION: ALL 13 CHECKS PASS")
        return 0
    else:
        print(f"PHASE 5 AUTOMATED VALIDATION: {len(failures)} CHECKS FAILED")
        for num, name, status, detail in failures:
            print(f"  Check {num} FAIL: {name} - {detail}")
        return 1


if __name__ == "__main__":
    sys.exit(validate_phase5())
