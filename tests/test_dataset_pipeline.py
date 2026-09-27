"""
test_dataset_pipeline.py - Unit Tests for FakeShield Dataset Preprocessing Pipeline

Verifies that:
- raw files exist
- raw hashes unchanged
- processed files exist
- IFND labels valid
- LIAR labels valid
- required columns exist
- LIAR split counts preserved
- IFND split leakage absent
- original text preserved
- processed files load
- no unexpected null labels
"""

import hashlib
from pathlib import Path
import pytest
import pandas as pd

# Paths
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
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

IFND_REQUIRED_COLUMNS = [
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

LIAR_REQUIRED_COLUMNS = [
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


def test_raw_files_exist():
    """Verify all raw dataset files exist and are non-empty."""
    for name, path in RAW_FILES.items():
        assert path.exists(), f"Raw file does not exist: {path}"
        assert path.stat().st_size > 0, f"Raw file is empty: {path}"


def test_raw_hashes_unchanged():
    """Verify that raw files match before and after hashes and live hash calculations."""
    before_csv = EVAL_DIR / "raw_file_hashes_before.csv"
    after_csv = EVAL_DIR / "raw_file_hashes_after.csv"
    assert before_csv.exists(), f"Missing {before_csv}"
    assert after_csv.exists(), f"Missing {after_csv}"

    df_before = pd.read_csv(before_csv)
    df_after = pd.read_csv(after_csv)
    assert df_before.equals(df_after), "Hashes before and after preprocessing do not match!"

    # Also compute live hashes and verify match
    for _, row in df_before.iterrows():
        fpath = RAW_DIR / ("IFND/IFND.csv" if row["dataset"] == "IFND" else f"LIAR/{row['file']}")
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        assert sha.hexdigest() == row["sha256"], f"Live hash mismatch for {row['file']}"


def test_processed_files_exist():
    """Verify all processed and split files exist on disk."""
    required = [
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
    for p in required:
        assert p.exists(), f"Required file missing: {p}"
        assert p.stat().st_size > 0, f"Required file is empty: {p}"


def test_processed_files_load():
    """Verify that processed files can be parsed and loaded with pandas."""
    df_ifnd = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", dtype={"label_original": str}, low_memory=False)
    assert len(df_ifnd) == 56714, f"Expected 56,714 IFND rows, got {len(df_ifnd)}"

    df_liar = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", low_memory=False)
    assert len(df_liar) == 12791, f"Expected 12,791 LIAR rows, got {len(df_liar)}"


def test_required_columns_exist():
    """Verify all required columns are present in processed IFND and LIAR."""
    df_ifnd = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", nrows=10)
    for col in IFND_REQUIRED_COLUMNS:
        assert col in df_ifnd.columns, f"Column '{col}' missing from ifnd_clean.csv"

    df_liar = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", nrows=10)
    for col in LIAR_REQUIRED_COLUMNS:
        assert col in df_liar.columns, f"Column '{col}' missing from liar_combined.csv"


def test_ifnd_labels_valid():
    """Verify IFND label integrity, binary mapping, and absence of unexpected labels."""
    df_ifnd = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", dtype={"label_original": str}, low_memory=False)
    expected_labels = {"TRUE", "Fake"}
    assert set(df_ifnd["label_original"].unique()) == expected_labels
    assert (df_ifnd["label_original"] == "TRUE").sum() == 37800
    assert (df_ifnd["label_original"] == "Fake").sum() == 18914

    # Binary check
    assert set(df_ifnd["label_binary"].unique()) == {0, 1}
    assert ((df_ifnd["label_original"] == "TRUE") == (df_ifnd["label_binary"] == 1)).all()
    assert ((df_ifnd["label_original"] == "Fake") == (df_ifnd["label_binary"] == 0)).all()


def test_liar_labels_valid():
    """Verify LIAR six original labels are preserved without alterations."""
    df_liar = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", low_memory=False)
    expected_labels = {"half-true", "false", "mostly-true", "barely-true", "true", "pants-fire"}
    assert set(df_liar["label"].unique()) == expected_labels
    assert (df_liar["label"] == "half-true").sum() == 2627
    assert (df_liar["label"] == "false").sum() == 2507
    assert (df_liar["label"] == "mostly-true").sum() == 2454
    assert (df_liar["label"] == "barely-true").sum() == 2103
    assert (df_liar["label"] == "true").sum() == 2053
    assert (df_liar["label"] == "pants-fire").sum() == 1047


def test_liar_split_counts_preserved():
    """Verify LIAR official benchmark split counts are perfectly preserved."""
    df_tr = pd.read_csv(PROCESSED_DIR / "liar_train_clean.csv", low_memory=False)
    df_va = pd.read_csv(PROCESSED_DIR / "liar_validation_clean.csv", low_memory=False)
    df_te = pd.read_csv(PROCESSED_DIR / "liar_test_clean.csv", low_memory=False)
    df_comb = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", low_memory=False)

    assert len(df_tr) == 10240, f"Expected 10,240 LIAR train rows, got {len(df_tr)}"
    assert len(df_va) == 1284, f"Expected 1,284 LIAR val rows, got {len(df_va)}"
    assert len(df_te) == 1267, f"Expected 1,267 LIAR test rows, got {len(df_te)}"
    assert len(df_comb) == 12791, f"Expected 12,791 LIAR combined rows, got {len(df_comb)}"


def test_ifnd_split_leakage_absent():
    """Verify that zero normalized statements leak across IFND train/val/test splits."""
    df_tr = pd.read_csv(SPLITS_DIR / "ifnd_train.csv", low_memory=False)
    df_va = pd.read_csv(SPLITS_DIR / "ifnd_validation.csv", low_memory=False)
    df_te = pd.read_csv(SPLITS_DIR / "ifnd_test.csv", low_memory=False)

    s_tr = set(df_tr["normalized_statement"])
    s_va = set(df_va["normalized_statement"])
    s_te = set(df_te["normalized_statement"])

    assert len(s_tr.intersection(s_va)) == 0, "Statement leakage between IFND train and val!"
    assert len(s_tr.intersection(s_te)) == 0, "Statement leakage between IFND train and test!"
    assert len(s_va.intersection(s_te)) == 0, "Statement leakage between IFND val and test!"

    # ID overlap
    id_tr = set(df_tr["id"])
    id_va = set(df_va["id"])
    id_te = set(df_te["id"])
    assert len(id_tr.intersection(id_va)) == 0, "ID leakage between IFND train and val!"
    assert len(id_tr.intersection(id_te)) == 0, "ID leakage between IFND train and test!"
    assert len(id_va.intersection(id_te)) == 0, "ID leakage between IFND val and test!"


def test_original_text_preserved():
    """Verify original statements in processed IFND match the raw CSV exactly."""
    raw_ifnd = pd.read_csv(RAW_FILES["IFND"], encoding="latin1")
    proc_ifnd = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", low_memory=False)
    assert (raw_ifnd["Statement"] == proc_ifnd["original_statement"]).all()


def test_no_unexpected_null_labels():
    """Verify there are zero null labels in processed IFND and LIAR."""
    proc_ifnd = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", dtype={"label_original": str}, low_memory=False)
    proc_liar = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", low_memory=False)

    assert proc_ifnd["label_original"].isnull().sum() == 0
    assert proc_ifnd["label_binary"].isnull().sum() == 0
    assert proc_liar["label"].isnull().sum() == 0
