#!/usr/bin/env python
"""
validate_datasets.py - FakeShield Automated Dataset Validation Suite

Checks the 17 research-grade validation criteria:
1. All raw files exist
2. All raw hashes unchanged
3. Processed files exist
4. Processed files load
5. Required columns exist
6. IFND labels valid
7. LIAR labels unchanged
8. Original text preserved
9. Normalized text exists
10. No unexpected null labels
11. No cross-split exact normalized statement duplicates in IFND
12. No identical IDs across splits
13. LIAR official splits preserved
14. No fabricated values
15. No fabricated dates
16. No fabricated labels
17. No fabricated URLs / Sources
"""

import sys
import hashlib
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

RAW_FILES = {
    "IFND": RAW_DIR / "IFND" / "IFND.csv",
    "LIAR_train": RAW_DIR / "LIAR" / "train.tsv",
    "LIAR_test": RAW_DIR / "LIAR" / "test.tsv",
    "LIAR_valid": RAW_DIR / "LIAR" / "valid.tsv",
}

IFND_EXPECTED_COLUMNS = [
    "id",
    "original_statement",
    "normalized_statement",
    "text_quality_flag",
    "duplicate_flag",
    "label_original",
    "label_binary",
    "category_original",
    "category_normalized",
    "category_quality_flag",
    "date_original",
    "date_normalized",
    "date_parse_status",
    "source_original",
    "source_normalized",
    "image_reference_original",
    "image_reference_quality",
]

LIAR_EXPECTED_COLUMNS = [
    "id",
    "label",
    "statement_original",
    "statement_normalized",
    "text_quality_flag",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
    "split",
]


def run_validation():
    print("=" * 80)
    print("FAKESHIELD AUTOMATED DATASET VALIDATION SUITE")
    print("=" * 80)
    results = []

    def record_check(check_id, desc, passed, details=""):
        status = "PASS" if passed else "FAIL"
        results.append((check_id, desc, status, details))
        mark = "[PASS]" if passed else "[FAIL]"
        print(f"  {mark:6s} Check {check_id:2d}: {desc}")
        if details and not passed:
            print(f"          Detail: {details}")

    # 1. All raw files exist
    all_raw_exist = all(p.exists() for p in RAW_FILES.values())
    record_check(1, "All raw files exist", all_raw_exist)

    # 2. All raw hashes unchanged
    before_csv = EVAL_DIR / "raw_file_hashes_before.csv"
    hashes_ok = False
    details_h = ""
    if before_csv.exists():
        df_before = pd.read_csv(before_csv)
        live_hashes = []
        for _, row in df_before.iterrows():
            fpath = RAW_DIR / ("IFND/IFND.csv" if row["dataset"] == "IFND" else f"LIAR/{row['file']}")
            sha = hashlib.sha256()
            with open(fpath, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            live_hashes.append(sha.hexdigest())
        df_before["live_sha256"] = live_hashes
        hashes_ok = (df_before["sha256"] == df_before["live_sha256"]).all()
        if not hashes_ok:
            details_h = "Mismatch between recorded before-hashes and live hashes!"
    else:
        details_h = "raw_file_hashes_before.csv missing"
    record_check(2, "All raw hashes unchanged", hashes_ok, details_h)

    # 3. Processed files exist
    required_files = [
        PROCESSED_DIR / "ifnd_clean.csv",
        PROCESSED_DIR / "liar_train_clean.csv",
        PROCESSED_DIR / "liar_validation_clean.csv",
        PROCESSED_DIR / "liar_test_clean.csv",
        PROCESSED_DIR / "liar_combined.csv",
        SPLITS_DIR / "ifnd_train.csv",
        SPLITS_DIR / "ifnd_validation.csv",
        SPLITS_DIR / "ifnd_test.csv",
        SPLITS_DIR / "liar_train.csv",
        SPLITS_DIR / "liar_validation.csv",
        SPLITS_DIR / "liar_test.csv",
        EVAL_DIR / "duplicate_analysis.csv",
        EVAL_DIR / "data_leakage_report.txt",
        EVAL_DIR / "preprocessing_audit.csv",
        EVAL_DIR / "data_quality_report.csv",
        EVAL_DIR / "reproducibility_info.txt",
    ]
    proc_exist = all(p.exists() for p in required_files)
    missing_proc = [str(p.name) for p in required_files if not p.exists()]
    record_check(3, "All processed and evaluation files exist", proc_exist, f"Missing: {missing_proc}")

    # 4. Processed files load
    files_load_ok = False
    details_load = ""
    try:
        df_ifnd = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", dtype={"label_original": str}, low_memory=False)
        df_liar_comb = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", low_memory=False)
        df_ifnd_tr = pd.read_csv(SPLITS_DIR / "ifnd_train.csv", dtype={"label_original": str}, low_memory=False)
        df_ifnd_va = pd.read_csv(SPLITS_DIR / "ifnd_validation.csv", dtype={"label_original": str}, low_memory=False)
        df_ifnd_te = pd.read_csv(SPLITS_DIR / "ifnd_test.csv", dtype={"label_original": str}, low_memory=False)
        files_load_ok = True
    except Exception as e:
        files_load_ok = False
        details_load = str(e)
    record_check(4, "Processed files load successfully", files_load_ok, details_load)

    # 5. Required columns exist
    if files_load_ok:
        ifnd_cols_ok = set(IFND_EXPECTED_COLUMNS).issubset(set(df_ifnd.columns))
        liar_cols_ok = set(LIAR_EXPECTED_COLUMNS).issubset(set(df_liar_comb.columns))
        cols_ok = ifnd_cols_ok and liar_cols_ok
        details_cols = f"IFND missing: {set(IFND_EXPECTED_COLUMNS) - set(df_ifnd.columns)}" if not ifnd_cols_ok else ""
    else:
        cols_ok = False
        details_cols = "Could not verify columns due to load failure."
    record_check(5, "Required columns exist in processed datasets", cols_ok, details_cols)

    # 6. IFND labels valid
    if files_load_ok:
        labels = set(df_ifnd["label_original"].unique())
        expected_ifnd_labels = {"TRUE", "Fake"}
        binary_ok = set(df_ifnd["label_binary"].unique()).issubset({0, 1})
        mapping_ok = (
            ((df_ifnd["label_original"] == "TRUE") == (df_ifnd["label_binary"] == 1)).all() and
            ((df_ifnd["label_original"] == "Fake") == (df_ifnd["label_binary"] == 0)).all()
        )
        ifnd_labels_ok = (labels == expected_ifnd_labels) and binary_ok and mapping_ok
    else:
        ifnd_labels_ok = False
    record_check(6, "IFND labels valid (TRUE=1, Fake=0; no other labels)", ifnd_labels_ok)

    # 7. LIAR labels unchanged
    if files_load_ok:
        liar_labels = set(df_liar_comb["label"].unique())
        expected_liar_labels = {"half-true", "false", "mostly-true", "barely-true", "true", "pants-fire"}
        liar_labels_ok = (liar_labels == expected_liar_labels)
    else:
        liar_labels_ok = False
    record_check(7, "LIAR six original labels unchanged", liar_labels_ok)

    # 8. Original text preserved
    if files_load_ok:
        raw_ifnd = pd.read_csv(RAW_FILES["IFND"], encoding="latin1")
        orig_match = (raw_ifnd["Statement"] == df_ifnd["original_statement"]).all()
    else:
        orig_match = False
    record_check(8, "Original text preserved without mutation", orig_match)

    # 9. Normalized text exists
    if files_load_ok:
        norm_exists = (
            (df_ifnd["normalized_statement"].notnull().all()) and
            (df_liar_comb["statement_normalized"].notnull().all())
        )
    else:
        norm_exists = False
    record_check(9, "Normalized text exists for all rows", norm_exists)

    # 10. No unexpected null labels
    if files_load_ok:
        null_labels_ok = (
            df_ifnd["label_original"].isnull().sum() == 0 and
            df_ifnd["label_binary"].isnull().sum() == 0 and
            df_liar_comb["label"].isnull().sum() == 0
        )
    else:
        null_labels_ok = False
    record_check(10, "No unexpected null labels", null_labels_ok)

    # 11. No cross-split exact normalized statement duplicates in IFND
    if files_load_ok:
        s_tr = set(df_ifnd_tr["normalized_statement"])
        s_va = set(df_ifnd_va["normalized_statement"])
        s_te = set(df_ifnd_te["normalized_statement"])
        leakage = len(s_tr.intersection(s_va)) + len(s_tr.intersection(s_te)) + len(s_va.intersection(s_te))
        no_leakage = (leakage == 0)
        details_leak = f"Found {leakage} cross-split statement duplicates" if not no_leakage else ""
    else:
        no_leakage = False
        details_leak = "Not loaded"
    record_check(11, "No cross-split normalized statement duplicates in IFND", no_leakage, details_leak)

    # 12. No identical IDs across splits
    if files_load_ok:
        id_tr = set(df_ifnd_tr["id"])
        id_va = set(df_ifnd_va["id"])
        id_te = set(df_ifnd_te["id"])
        ifnd_id_leak = len(id_tr.intersection(id_va)) + len(id_tr.intersection(id_te)) + len(id_va.intersection(id_te))
        
        liar_tr = pd.read_csv(PROCESSED_DIR / "liar_train_clean.csv")
        liar_va = pd.read_csv(PROCESSED_DIR / "liar_validation_clean.csv")
        liar_te = pd.read_csv(PROCESSED_DIR / "liar_test_clean.csv")
        liar_id_leak = (
            len(set(liar_tr["id"]).intersection(set(liar_va["id"]))) +
            len(set(liar_tr["id"]).intersection(set(liar_te["id"]))) +
            len(set(liar_va["id"]).intersection(set(liar_te["id"])))
        )
        no_id_leak = (ifnd_id_leak == 0 and liar_id_leak == 0)
    else:
        no_id_leak = False
    record_check(12, "No identical IDs across splits (IFND and LIAR)", no_id_leak)

    # 13. LIAR official splits preserved
    if files_load_ok:
        liar_counts_ok = (
            len(liar_tr) == 10240 and
            len(liar_va) == 1284 and
            len(liar_te) == 1267 and
            len(df_liar_comb) == 12791
        )
    else:
        liar_counts_ok = False
    record_check(13, "LIAR official split counts preserved (10240 / 1284 / 1267)", liar_counts_ok)

    # 14. No fabricated values
    if files_load_ok:
        row_counts_ok = (len(df_ifnd) == 56714 and len(df_liar_comb) == 12791)
    else:
        row_counts_ok = False
    record_check(14, "No fabricated or deleted values (row counts match raw)", row_counts_ok)

    # 15. No fabricated dates
    if files_load_ok:
        raw_missing_dates = raw_ifnd["Date"].isnull().sum()
        proc_missing_dates = (df_ifnd["date_parse_status"] == "MISSING").sum()
        parsed_dates = df_ifnd[df_ifnd["date_parse_status"] == "PARSED_YEAR_MONTH"]["date_normalized"]
        all_yyyy_mm = parsed_dates.str.match(r"^\d{4}-\d{2}$").all()
        no_fab_dates = (raw_missing_dates == proc_missing_dates) and all_yyyy_mm
    else:
        no_fab_dates = False
    record_check(15, "No fabricated dates (missing stay missing, formats YYYY-MM)", no_fab_dates)

    # 16. No fabricated labels
    if files_load_ok:
        label_match = (
            (df_ifnd["label_original"] == "TRUE").sum() == 37800 and
            (df_ifnd["label_original"] == "Fake").sum() == 18914
        )
    else:
        label_match = False
    record_check(16, "No fabricated labels (counts 37,800 TRUE and 18,914 Fake match)", label_match)

    # 17. No fabricated URLs / Sources
    if files_load_ok:
        source_preserved = (df_ifnd["source_original"] == raw_ifnd["Web"]).all()
        unique_publishers = df_ifnd["source_normalized"].nunique() == 29
        source_ok = source_preserved and unique_publishers
    else:
        source_ok = False
    record_check(17, "No fabricated URLs (Web correctly treated as source publisher)", source_ok)

    # Overall outcome
    all_passed = all(r[2] == "PASS" for r in results)
    print("=" * 80)
    if all_passed:
        print("ALL 17 AUTOMATED VALIDATION CHECKS PASSED: [PASS]")
        print("=" * 80)
        return 0
    else:
        failed = [f"Check {r[0]}: {r[1]}" for r in results if r[2] == "FAIL"]
        print(f"AUTOMATED VALIDATION FAILED on: {failed}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
