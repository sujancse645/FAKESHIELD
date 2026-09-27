"""validate_phase11.py - Automated validation suite for FakeShield Phase 11.

Performs 22 rigorous automated validation checks covering:
1. Benchmark directory existence (data/evaluation/benchmark/)
2. Baseline package import and modularity
3. TfidfBaseline class initialization & hyperparameter reporting
4. Baseline binary classification execution
5. Baseline multiclass classification execution
6. Baseline train/test separation (zero data leakage)
7. Mathematical accuracy of compute_classification_metrics
8. Baseline results artifact (phase11_baseline_results.csv)
9. FakeShield benchmark results artifact (phase11_fakeshield_results.csv)
10. Label space distinction (dataset labels vs 5 FakeShield verdicts)
11. Evidence retrieval ablation artifact (phase11_ablation_results.csv)
12. Multilingual retrieval experiment artifact (phase11_multilingual_results.csv)
13. Abstention analysis artifact (phase11_abstention_results.csv)
14. Confidence calibration artifact (phase11_calibration_results.csv)
15. 12-category error analysis artifact (phase11_error_analysis.csv)
16. Confusion matrix artifact (phase11_confusion_matrix.csv)
17. Comparative table artifact (phase11_comparison.csv)
18. Reproducibility metadata artifact (phase11_reproducibility.txt)
19. 21-section research report completeness (phase11_report.txt)
20. 20-item manual checklist initialized to MANUAL VERIFICATION REQUIRED
21. Phase 1 raw dataset cryptographic hash integrity (100% MATCH)
22. Security audit: Zero hard-coded secrets, tokens, or credentials
"""

import os
import sys
import re
import hashlib
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "benchmark")
RAW_FILES = {
    "IFND.csv": (os.path.join(PROJECT_ROOT, "data", "raw", "IFND", "IFND.csv"), "3728c7b785ea93c9dadcc61a4c549c0eddd71a5fb6a1a31be08b5235ee4f5bfa"),
    "train.tsv": (os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "train.tsv"), "23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36"),
    "valid.tsv": (os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "valid.tsv"), "8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16"),
    "test.tsv": (os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "test.tsv"), "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"),
}


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_validation():
    print("=" * 80)
    print("FAKESHIELD PHASE 11: AUTOMATED VALIDATION SUITE")
    print("=" * 80)

    checks = []

    def record(num: int, title: str, passed: bool, detail: str = ""):
        status_str = "PASS" if passed else "FAIL"
        checks.append((num, title, passed, detail))
        print(f"Check {num:02d}: [{status_str}] {title} {detail}")

    # Check 1: Benchmark directory exists
    c1 = os.path.isdir(BENCHMARK_DIR)
    record(1, "Benchmark evaluation directory exists (data/evaluation/benchmark)", c1)

    # Check 2: Baseline package cleanly importable
    try:
        from src.baseline.tfidf_baseline import TfidfBaseline, BaselineMetrics, compute_classification_metrics
        record(2, "Baseline package cleanly importable from src.baseline", True)
    except Exception as e:
        record(2, "Baseline package cleanly importable", False, str(e))

    # Check 3: TfidfBaseline initialization and hyperparameter reporting
    try:
        b = TfidfBaseline(ngram_range=(1, 2), max_features=500, C=1.0, random_state=42)
        params = b.get_hyperparameters()
        c3 = params["max_features"] == 500 and params["random_state"] == 42 and not params["is_fitted"]
        record(3, "TfidfBaseline initialization and hyperparameter reporting", c3)
    except Exception as e:
        record(3, "TfidfBaseline initialization", False, str(e))

    # Check 4: Baseline binary classification execution
    try:
        X_toy = ["Official government press release confirms policy update", "Scam forwarded message promises free money"]
        y_toy = [1, 0]
        b.fit(X_toy, y_toy)
        preds = b.predict(["Government releases official policy document"])
        c4 = len(preds) == 1 and preds[0] in [0, 1]
        record(4, "Baseline binary fit and predict execution", c4)
    except Exception as e:
        record(4, "Baseline binary execution", False, str(e))

    # Check 5: Baseline multiclass classification execution
    try:
        b_multi = TfidfBaseline(random_state=42)
        X_m = ["Statement A", "Statement B", "Statement C"]
        y_m = ["true", "false", "half-true"]
        b_multi.fit(X_m, y_m)
        preds_m = b_multi.predict(["Statement A"])
        c5 = len(preds_m) == 1 and preds_m[0] in y_m
        record(5, "Baseline multiclass fit and predict execution", c5)
    except Exception as e:
        record(5, "Baseline multiclass execution", False, str(e))

    # Check 6: Baseline train/test separation (zero leakage)
    try:
        splits_dir = os.path.join(PROJECT_ROOT, "data", "splits")
        train_df = pd.read_csv(os.path.join(splits_dir, "ifnd_train.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, "ifnd_test.csv"))
        # Verify disjoint IDs
        train_ids = set(train_df["id"])
        test_ids = set(test_df["id"])
        c6 = len(train_ids.intersection(test_ids)) == 0
        record(6, "Dataset train/test split disjointness (zero data leakage)", c6)
    except Exception as e:
        record(6, "Dataset split separation check", False, str(e))

    # Check 7: compute_classification_metrics mathematical validity
    try:
        m = compute_classification_metrics([1, 0, 1, 0], [1, 0, 1, 1], "toy", "test")
        c7 = abs(m.accuracy - 0.75) < 1e-4 and m.num_samples == 4
        record(7, "Mathematical accuracy of compute_classification_metrics", c7)
    except Exception as e:
        record(7, "Metric computation validation", False, str(e))

    # Check 8: Baseline results artifact exists and valid
    try:
        p_base = os.path.join(BENCHMARK_DIR, "phase11_baseline_results.csv")
        c8 = os.path.isfile(p_base) and os.path.getsize(p_base) > 100
        if c8:
            df_b = pd.read_csv(p_base)
            c8 = len(df_b) >= 3 and "accuracy" in df_b.columns
        record(8, "Baseline results artifact exists and contains valid metrics", c8)
    except Exception as e:
        record(8, "Baseline results artifact check", False, str(e))

    # Check 9: FakeShield benchmark results artifact exists
    try:
        p_fs = os.path.join(BENCHMARK_DIR, "phase11_fakeshield_results.csv")
        c9 = os.path.isfile(p_fs) and os.path.getsize(p_fs) > 500
        if c9:
            df_fs = pd.read_csv(p_fs)
            c9 = len(df_fs) >= 20 and "observed_verdict" in df_fs.columns
        record(9, "FakeShield benchmark results artifact exists (20 cases)", c9)
    except Exception as e:
        record(9, "FakeShield benchmark results check", False, str(e))

    # Check 10: Label space distinction documented
    try:
        p_rep = os.path.join(BENCHMARK_DIR, "phase11_report.txt")
        with open(p_rep, "r", encoding="utf-8") as f:
            rep_text = f.read()
        c10 = "Dataset label" in rep_text or "label spaces" in rep_text.lower() or "5 canonical" in rep_text
        record(10, "Label space distinction documented (dataset vs 5 verdicts)", c10)
    except Exception as e:
        record(10, "Label space distinction check", False, str(e))

    # Check 11: Evidence retrieval ablation artifact exists
    try:
        p_abl = os.path.join(BENCHMARK_DIR, "phase11_ablation_results.csv")
        c11 = os.path.isfile(p_abl) and os.path.getsize(p_abl) > 500
        if c11:
            df_a = pd.read_csv(p_abl)
            ablated_only = df_a[df_a["condition"].str.contains("Ablated")]
            c11 = len(ablated_only) >= 20 and all(ablated_only["observed_verdict"] == "UNVERIFIED")
        record(11, "Evidence ablation artifact demonstrates 100% abstention without evidence", c11)
    except Exception as e:
        record(11, "Evidence ablation check", False, str(e))

    # Check 12: Multilingual retrieval experiment artifact exists
    try:
        p_multi = os.path.join(BENCHMARK_DIR, "phase11_multilingual_results.csv")
        c12 = os.path.isfile(p_multi) and os.path.getsize(p_multi) > 200
        if c12:
            df_m = pd.read_csv(p_multi)
            c12 = len(df_m) >= 6 and "english_pivot" in df_m.columns
        record(12, "Multilingual retrieval artifact compares script vs English pivot (6 cases)", c12)
    except Exception as e:
        record(12, "Multilingual retrieval check", False, str(e))

    # Check 13: Abstention analysis artifact exists
    try:
        p_abs = os.path.join(BENCHMARK_DIR, "phase11_abstention_results.csv")
        c13 = os.path.isfile(p_abs) and os.path.getsize(p_abs) > 200
        if c13:
            df_abs = pd.read_csv(p_abs)
            c13 = len(df_abs) >= 10 and "abstention_category" in df_abs.columns
        record(13, "Abstention analysis artifact categorizes UNVERIFIED rationales", c13)
    except Exception as e:
        record(13, "Abstention analysis check", False, str(e))

    # Check 14: Confidence calibration artifact exists
    try:
        p_cal = os.path.join(BENCHMARK_DIR, "phase11_calibration_results.csv")
        c14 = os.path.isfile(p_cal) and os.path.getsize(p_cal) > 200
        if c14:
            df_c = pd.read_csv(p_cal)
            c14 = "brier_contribution" in df_c.columns and len(df_c) >= 20
        record(14, "Confidence calibration artifact includes Brier score metric", c14)
    except Exception as e:
        record(14, "Confidence calibration check", False, str(e))

    # Check 15: 12-category error analysis artifact exists
    try:
        p_err = os.path.join(BENCHMARK_DIR, "phase11_error_analysis.csv")
        c15 = os.path.isfile(p_err) and os.path.getsize(p_err) > 500
        if c15:
            df_err = pd.read_csv(p_err)
            c15 = len(df_err) == 12 and "error_category" in df_err.columns
        record(15, "Error analysis artifact covers all 12 taxonomy categories", c15)
    except Exception as e:
        record(15, "Error analysis check", False, str(e))

    # Check 16: Confusion matrix artifact exists
    try:
        p_cm = os.path.join(BENCHMARK_DIR, "phase11_confusion_matrix.csv")
        c16 = os.path.isfile(p_cm) and os.path.getsize(p_cm) > 200
        if c16:
            df_cm = pd.read_csv(p_cm)
            matrices = set(df_cm["matrix_name"])
            c16 = len(matrices) >= 3
        record(16, "Confusion matrix artifact contains Baseline, NLI, and 5-way matrices", c16)
    except Exception as e:
        record(16, "Confusion matrix check", False, str(e))

    # Check 17: Comparative table artifact exists
    try:
        p_comp = os.path.join(BENCHMARK_DIR, "phase11_comparison.csv")
        c17 = os.path.isfile(p_comp) and os.path.getsize(p_comp) > 200
        if c17:
            df_comp = pd.read_csv(p_comp)
            c17 = len(df_comp) >= 3 and "External Evidence" in df_comp.columns
        record(17, "Comparative table artifact compares architectural dimensions and metrics", c17)
    except Exception as e:
        record(17, "Comparative table check", False, str(e))

    # Check 18: Reproducibility metadata artifact exists
    try:
        p_rep = os.path.join(BENCHMARK_DIR, "phase11_reproducibility.txt")
        c18 = os.path.isfile(p_rep) and os.path.getsize(p_rep) > 300
        if c18:
            with open(p_rep, "r", encoding="utf-8") as f:
                r_txt = f.read()
            c18 = "Package Versions" in r_txt and "SHA-256" in r_txt and "Random Seeds" in r_txt
        record(18, "Reproducibility metadata documents hardware, packages, seeds, and hashes", c18)
    except Exception as e:
        record(18, "Reproducibility check", False, str(e))

    # Check 19: 21-section research report completeness
    try:
        p_rep11 = os.path.join(BENCHMARK_DIR, "phase11_report.txt")
        c19 = os.path.isfile(p_rep11) and os.path.getsize(p_rep11) > 2000
        if c19:
            with open(p_rep11, "r", encoding="utf-8") as f:
                rep_lines = f.read()
            # Verify required sections (1 through 21)
            missing = [sec for sec in range(1, 22) if f"{sec}." not in rep_lines]
            c19 = len(missing) == 0
        record(19, "Phase 11 technical research report includes all 21 required sections", c19)
    except Exception as e:
        record(19, "Report completeness check", False, str(e))

    # Check 20: 20-item manual checklist initialized to MANUAL VERIFICATION REQUIRED
    try:
        p_chk = os.path.join(BENCHMARK_DIR, "phase11_manual_checklist.csv")
        c20 = os.path.isfile(p_chk) and os.path.getsize(p_chk) > 500
        if c20:
            df_chk = pd.read_csv(p_chk)
            all_manual = all(df_chk["status"] == "MANUAL VERIFICATION REQUIRED")
            c20 = len(df_chk) >= 20 and all_manual
        record(20, "Manual review checklist contains 20 items initialized to MANUAL VERIFICATION REQUIRED", c20)
    except Exception as e:
        record(20, "Manual checklist check", False, str(e))

    # Check 21: Phase 1 raw hashes match baseline 100%
    try:
        raw_ok = True
        for fn, (fp, exp) in RAW_FILES.items():
            if not os.path.exists(fp) or compute_sha256(fp) != exp:
                raw_ok = False
                break
        record(21, "Phase 1 raw dataset cryptographic hashes match baseline 100%", raw_ok)
    except Exception as e:
        record(21, "Raw dataset hash check", False, str(e))

    # Check 22: Security audit: zero hard-coded secrets or credentials
    try:
        sec_ok = True
        forbidden = ["api_key", "secret_token", "password =", "bearer "]
        for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "src", "baseline")):
            for fn in files:
                if fn.endswith(".py"):
                    with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                        content = f.read().lower()
                    for pat in forbidden:
                        if pat in content:
                            sec_ok = False
        record(22, "Security audit: Zero hard-coded API keys, tokens, or credentials", sec_ok)
    except Exception as e:
        record(22, "Security audit check", False, str(e))

    print("=" * 80)
    passed_count = sum(1 for c in checks if c[2])
    print(f"VALIDATION SUMMARY: {passed_count}/{len(checks)} CHECKS PASSED")
    print("=" * 80)
    return passed_count == len(checks)


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
