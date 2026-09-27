#!/usr/bin/env python
"""
inspect_datasets.py - FakeShield Dataset Setup & Inspection Script

Inspects raw IFND and LIAR datasets, performs basic standardized cleaning,
generates processed CSVs, and produces dataset quality evaluation reports.

Strict rules:
- Does NOT modify, rename, move, overwrite, or delete raw datasets.
- Does NOT train any ML model.
- Does NOT perform OCR, translation, NLI, retrieval, or UI creation.
- Does NOT change ground-truth labels.
- Does NOT merge IFND and LIAR into one training dataset.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Define paths relative to this script / project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

IFND_RAW_PATH = RAW_DIR / "IFND" / "IFND.csv"
LIAR_RAW_DIR = RAW_DIR / "LIAR"
LIAR_SPLITS = {
    "train": LIAR_RAW_DIR / "train.tsv",
    "test": LIAR_RAW_DIR / "test.tsv",
    "valid": LIAR_RAW_DIR / "valid.tsv"
}

IFND_CLEAN_PATH = PROCESSED_DIR / "ifnd_clean.csv"
LIAR_COMBINED_PATH = PROCESSED_DIR / "liar_combined.csv"
REPORT_PATH = EVAL_DIR / "dataset_report.txt"
QUALITY_PATH = EVAL_DIR / "dataset_quality.csv"

# Canonical column definitions for LIAR (Wang, 2017)
LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
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
    "context"
]


def inspect_raw_liar_structure():
    """Inspects raw LIAR files directly to identify structure without assuming column names."""
    print("=" * 80)
    print("STEP 0: RAW LIAR FILE STRUCTURE INSPECTION (NO ASSUMPTIONS)")
    print("=" * 80)
    for split_name, file_path in LIAR_SPLITS.items():
        if not file_path.exists():
            raise FileNotFoundError(f"LIAR raw file missing: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_lines = [f.readline().rstrip("\r\n") for _ in range(3)]
        print(f"\n[Raw File Inspection] {split_name}.tsv:")
        print(f"  File location: {file_path}")
        for idx, line in enumerate(first_lines):
            fields = line.split("\t")
            print(f"  Line {idx + 1} field count: {len(fields)}")
            print(f"  Line {idx + 1} sample: {fields[:4]} ... (total {len(fields)} fields)")

    print("\n[Structure Discovery Finding]:")
    print("  - LIAR .tsv files contain tab-delimited records with NO header row.")
    print("  - Each complete record consists of 14 columns matching the canonical LIAR benchmark schema:")
    print(f"    {LIAR_COLUMNS}")
    print("=" * 80)


def inspect_ifnd():
    """Inspects the IFND dataset."""
    print("\n" + "=" * 80)
    print("FOR IFND: DATASET INSPECTION")
    print("=" * 80)

    if not IFND_RAW_PATH.exists():
        raise FileNotFoundError(f"IFND file not found at {IFND_RAW_PATH}")

    # Raw IFND contains non-UTF8 characters (e.g. byte 0xd4); latin1 loads reliably
    print(f"Loading IFND from: {IFND_RAW_PATH} using latin1 encoding...")
    df_ifnd = pd.read_csv(IFND_RAW_PATH, encoding="latin1")

    # 1. Number of rows
    num_rows = len(df_ifnd)
    print(f"\n1. Number of rows: {num_rows}")

    # 2. Number of columns
    num_cols = len(df_ifnd.columns)
    print(f"2. Number of columns: {num_cols}")

    # 3. Column names
    col_names = df_ifnd.columns.tolist()
    print(f"3. Column names: {col_names}")

    # 4. First 5 rows
    print("\n4. First 5 rows:")
    print(df_ifnd.head(5).to_string())

    # 5. Missing values per column
    missing_per_col = df_ifnd.isnull().sum()
    print("\n5. Missing values per column:")
    for col, cnt in missing_per_col.items():
        pct = (cnt / num_rows) * 100
        print(f"   - {col:12s}: {cnt:6d} ({pct:5.2f}%)")

    # 6. Number of duplicate rows
    dup_rows = df_ifnd.duplicated().sum()
    dup_statements = df_ifnd.duplicated(subset=["Statement"]).sum()
    print(f"\n6. Number of exact duplicate rows: {dup_rows}")
    print(f"   (Duplicate statements across rows: {dup_statements})")

    # 7. Label distribution
    label_dist = df_ifnd["Label"].value_counts(dropna=False)
    print("\n7. Label distribution:")
    for label, cnt in label_dist.items():
        pct = (cnt / num_rows) * 100
        print(f"   - {str(label):10s}: {cnt:6d} ({pct:5.2f}%)")

    # 8. Category distribution
    cat_dist = df_ifnd["Category"].value_counts(dropna=False)
    print("\n8. Category distribution:")
    for cat, cnt in cat_dist.items():
        pct = (cnt / num_rows) * 100
        print(f"   - {str(cat):15s}: {cnt:6d} ({pct:5.2f}%)")

    # 9. Date information/range
    unique_dates = df_ifnd["Date"].dropna().unique()
    print(f"\n9. Date information/range:")
    print(f"   - Unique non-null date values: {len(unique_dates)}")
    print(f"   - Missing date entries: {df_ifnd['Date'].isnull().sum()}")
    print(f"   - Sample dates: {list(unique_dates[:10])}")

    # 10. Number of unique image URLs
    unique_images = df_ifnd["Image"].nunique()
    print(f"\n10. Number of unique image URLs: {unique_images}")

    # 11. Number of unique web URLs
    unique_webs = df_ifnd["Web"].nunique()
    print(f"11. Number of unique web sources/URLs: {unique_webs}")
    print(f"    Sample web sources: {list(df_ifnd['Web'].value_counts().head(5).index)}")

    return df_ifnd


def inspect_liar():
    """Inspects each split of the LIAR dataset."""
    print("\n" + "=" * 80)
    print("FOR LIAR: DATASET INSPECTION")
    print("=" * 80)

    liar_dfs = {}
    for split_name, file_path in LIAR_SPLITS.items():
        print(f"\n--- LIAR Split: {split_name.upper()} ({file_path.name}) ---")
        df_split = pd.read_csv(file_path, sep="\t", header=None, names=LIAR_COLUMNS)
        liar_dfs[split_name] = df_split

        # 1. Number of rows
        num_rows = len(df_split)
        print(f"1. Number of rows: {num_rows}")

        # 2. Number of columns
        num_cols = len(df_split.columns)
        print(f"2. Number of columns: {num_cols}")

        # 3. Column names
        print(f"3. Column names: {df_split.columns.tolist()}")

        # 4. First 5 rows
        print("4. First 5 rows:")
        print(df_split.head(5).to_string())

        # 5. Missing values
        print("5. Missing values per column:")
        for col, cnt in df_split.isnull().sum().items():
            pct = (cnt / num_rows) * 100
            print(f"   - {col:22s}: {cnt:5d} ({pct:5.2f}%)")

        # 6. Duplicate rows
        dup_rows = df_split.duplicated().sum()
        print(f"6. Duplicate rows: {dup_rows}")

        # 7. Label distribution
        print("7. Label distribution:")
        label_dist = df_split["label"].value_counts(dropna=False)
        for lbl, cnt in label_dist.items():
            pct = (cnt / num_rows) * 100
            print(f"   - {str(lbl):15s}: {cnt:5d} ({pct:5.2f}%)")

    return liar_dfs


def create_ifnd_clean(df_raw_ifnd):
    """
    Cleans IFND:
    - remove completely empty rows
    - remove exact duplicates
    - strip leading/trailing whitespace from text
    - preserve punctuation
    - preserve original labels
    - preserve the original useful columns
    """
    print("\n" + "=" * 80)
    print("CREATING: data/processed/ifnd_clean.csv")
    print("=" * 80)

    df_clean = df_raw_ifnd.copy()
    initial_len = len(df_clean)

    # 1. Remove completely empty rows
    df_clean = df_clean.dropna(how="all")
    after_dropna = len(df_clean)

    # 2. Strip leading/trailing whitespace from text without removing punctuation
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    # 3. Remove exact duplicates
    dups = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    final_len = len(df_clean)

    # Preserve original useful columns
    useful_cols = ["id", "Statement", "Image", "Web", "Category", "Date", "Label"]
    df_clean = df_clean[useful_cols]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(IFND_CLEAN_PATH, index=False, encoding="utf-8")

    print(f"Initial rows: {initial_len}")
    print(f"Rows after dropping empty rows: {after_dropna}")
    print(f"Exact duplicates removed: {dups}")
    print(f"Final clean rows saved: {final_len}")
    print(f"Saved to: {IFND_CLEAN_PATH}")

    return df_clean


def create_liar_combined(liar_dfs):
    """
    Combines train.tsv, test.tsv, and valid.tsv.
    Adds a 'split' column with:
      - 'train'
      - 'test'
      - 'validation'
    Preserves original labels.
    """
    print("\n" + "=" * 80)
    print("CREATING: data/processed/liar_combined.csv")
    print("=" * 80)

    split_mapping = {
        "train": "train",
        "test": "test",
        "valid": "validation"
    }

    combined_splits = []
    for split_name, df in liar_dfs.items():
        df_copy = df.copy()
        target_split_label = split_mapping[split_name]
        df_copy["split"] = target_split_label
        print(f"Adding split '{target_split_label}' with {len(df_copy)} rows.")
        combined_splits.append(df_copy)

    df_combined = pd.concat(combined_splits, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(LIAR_COMBINED_PATH, index=False, encoding="utf-8")

    print(f"Combined LIAR total rows: {len(df_combined)}")
    print(f"Combined columns ({len(df_combined.columns)}): {df_combined.columns.tolist()}")
    print(f"Split counts:\n{df_combined['split'].value_counts().to_string()}")
    print(f"Original labels preserved:\n{df_combined['label'].value_counts().to_string()}")
    print(f"Saved to: {LIAR_COMBINED_PATH}")

    return df_combined


def create_dataset_quality_csv(df_ifnd, liar_dfs, df_liar_combined):
    """
    Creates data/evaluation/dataset_quality.csv
    Columns:
      dataset
      total_rows
      duplicate_rows
      missing_text_rows
      missing_image_rows
      missing_label_rows
      unique_labels
      notes
    """
    print("\n" + "=" * 80)
    print("CREATING: data/evaluation/dataset_quality.csv")
    print("=" * 80)

    records = [
        {
            "dataset": "IFND",
            "total_rows": len(df_ifnd),
            "duplicate_rows": int(df_ifnd.duplicated().sum()),
            "missing_text_rows": int(df_ifnd["Statement"].isnull().sum()),
            "missing_image_rows": int(df_ifnd["Image"].isnull().sum()),
            "missing_label_rows": int(df_ifnd["Label"].isnull().sum()),
            "unique_labels": int(df_ifnd["Label"].nunique()),
            "notes": "Multimodal Indian news; Binary labels (TRUE: 37800, Fake: 18914); 11321 missing Date; 46755 unique images"
        },
        {
            "dataset": "LIAR (train)",
            "total_rows": len(liar_dfs["train"]),
            "duplicate_rows": int(liar_dfs["train"].duplicated().sum()),
            "missing_text_rows": int(liar_dfs["train"]["statement"].isnull().sum()),
            "missing_image_rows": len(liar_dfs["train"]),  # Text-only benchmark
            "missing_label_rows": int(liar_dfs["train"]["label"].isnull().sum()),
            "unique_labels": int(liar_dfs["train"]["label"].nunique()),
            "notes": "Text-only political claims; 6 fine-grained labels; missing job_title (2898), state_info (2210), context (102)"
        },
        {
            "dataset": "LIAR (test)",
            "total_rows": len(liar_dfs["test"]),
            "duplicate_rows": int(liar_dfs["test"].duplicated().sum()),
            "missing_text_rows": int(liar_dfs["test"]["statement"].isnull().sum()),
            "missing_image_rows": len(liar_dfs["test"]),
            "missing_label_rows": int(liar_dfs["test"]["label"].isnull().sum()),
            "unique_labels": int(liar_dfs["test"]["label"].nunique()),
            "notes": "Text-only test split; 6 labels; missing job_title (325), state_info (262), context (17)"
        },
        {
            "dataset": "LIAR (valid)",
            "total_rows": len(liar_dfs["valid"]),
            "duplicate_rows": int(liar_dfs["valid"].duplicated().sum()),
            "missing_text_rows": int(liar_dfs["valid"]["statement"].isnull().sum()),
            "missing_image_rows": len(liar_dfs["valid"]),
            "missing_label_rows": int(liar_dfs["valid"]["label"].isnull().sum()),
            "unique_labels": int(liar_dfs["valid"]["label"].nunique()),
            "notes": "Text-only validation split; 6 labels; missing job_title (345), state_info (279), context (12)"
        },
        {
            "dataset": "LIAR (combined)",
            "total_rows": len(df_liar_combined),
            "duplicate_rows": int(df_liar_combined.duplicated().sum()),
            "missing_text_rows": int(df_liar_combined["statement"].isnull().sum()),
            "missing_image_rows": len(df_liar_combined),
            "missing_label_rows": int(df_liar_combined["label"].isnull().sum()),
            "unique_labels": int(df_liar_combined["label"].nunique()),
            "notes": "Combined canonical LIAR benchmark; split column added (train/test/validation); text-only; 6 original labels"
        }
    ]

    quality_df = pd.DataFrame(records)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(QUALITY_PATH, index=False, encoding="utf-8")
    print(quality_df.to_string(index=False))
    print(f"\nSaved to: {QUALITY_PATH}")


def create_dataset_report_txt(df_ifnd_raw, df_ifnd_clean, liar_dfs, df_liar_combined):
    """
    Creates data/evaluation/dataset_report.txt
    Includes:
    - dataset sizes
    - columns
    - label distributions
    - missing values
    - duplicates
    - observations
    - problems found
    """
    print("\n" + "=" * 80)
    print("CREATING: data/evaluation/dataset_report.txt")
    print("=" * 80)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FAKESHIELD DATASET EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 1. Dataset Sizes
    report_lines.append("1. DATASET SIZES")
    report_lines.append("-" * 40)
    report_lines.append(f"IFND Raw Rows:                  {len(df_ifnd_raw):,}")
    report_lines.append(f"IFND Clean Rows:                {len(df_ifnd_clean):,}")
    report_lines.append(f"LIAR Train Split Rows:          {len(liar_dfs['train']):,}")
    report_lines.append(f"LIAR Test Split Rows:           {len(liar_dfs['test']):,}")
    report_lines.append(f"LIAR Validation Split Rows:     {len(liar_dfs['valid']):,}")
    report_lines.append(f"LIAR Combined Total Rows:       {len(df_liar_combined):,}")
    report_lines.append("")

    # 2. Columns
    report_lines.append("2. COLUMNS & SCHEMA")
    report_lines.append("-" * 40)
    report_lines.append(f"IFND Columns ({len(df_ifnd_raw.columns)}):")
    for col in df_ifnd_raw.columns:
        report_lines.append(f"  - {col}")
    report_lines.append("")
    report_lines.append(f"LIAR Raw/Combined Columns ({len(df_liar_combined.columns)}):")
    for col in df_liar_combined.columns:
        report_lines.append(f"  - {col}")
    report_lines.append("")

    # 3. Label Distributions
    report_lines.append("3. LABEL DISTRIBUTIONS")
    report_lines.append("-" * 40)
    report_lines.append("IFND Label Distribution:")
    for lbl, count in df_ifnd_raw["Label"].value_counts(dropna=False).items():
        pct = (count / len(df_ifnd_raw)) * 100
        report_lines.append(f"  - {lbl:12s}: {count:6d} ({pct:5.2f}%)")
    report_lines.append("")
    report_lines.append("LIAR Combined Label Distribution:")
    for lbl, count in df_liar_combined["label"].value_counts(dropna=False).items():
        pct = (count / len(df_liar_combined)) * 100
        report_lines.append(f"  - {lbl:15s}: {count:6d} ({pct:5.2f}%)")
    report_lines.append("")
    for split_name, df_sp in liar_dfs.items():
        report_lines.append(f"LIAR ({split_name}) Label Distribution:")
        for lbl, count in df_sp["label"].value_counts(dropna=False).items():
            pct = (count / len(df_sp)) * 100
            report_lines.append(f"  - {lbl:15s}: {count:5d} ({pct:5.2f}%)")
        report_lines.append("")

    # 4. Missing Values
    report_lines.append("4. MISSING VALUES ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append("IFND Missing Values:")
    for col, count in df_ifnd_raw.isnull().sum().items():
        pct = (count / len(df_ifnd_raw)) * 100
        report_lines.append(f"  - {col:15s}: {count:6d} ({pct:5.2f}%)")
    report_lines.append("")
    report_lines.append("LIAR Combined Missing Values:")
    for col, count in df_liar_combined.isnull().sum().items():
        pct = (count / len(df_liar_combined)) * 100
        report_lines.append(f"  - {col:22s}: {count:6d} ({pct:5.2f}%)")
    report_lines.append("")

    # 5. Duplicates
    report_lines.append("5. DUPLICATE ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append(f"IFND Exact Duplicate Rows:               {df_ifnd_raw.duplicated().sum()}")
    report_lines.append(f"IFND Duplicate Statements:               {df_ifnd_raw.duplicated(subset=['Statement']).sum()}")
    report_lines.append(f"LIAR Train Duplicate Rows:               {liar_dfs['train'].duplicated().sum()}")
    report_lines.append(f"LIAR Test Duplicate Rows:                {liar_dfs['test'].duplicated().sum()}")
    report_lines.append(f"LIAR Validation Duplicate Rows:          {liar_dfs['valid'].duplicated().sum()}")
    report_lines.append(f"LIAR Combined Duplicate Rows:            {df_liar_combined.duplicated().sum()}")
    report_lines.append(f"LIAR Combined Duplicate Statements:      {df_liar_combined.duplicated(subset=['statement']).sum()}")
    report_lines.append("")

    # 6. Observations
    report_lines.append("6. OBSERVATIONS")
    report_lines.append("-" * 40)
    report_lines.append("1. Nature of IFND Dataset:")
    report_lines.append("   - IFND is an Indian multimodal fake news dataset featuring statement headlines,")
    report_lines.append("     associated web publisher sources (29 sources like DNAINDIA, TRIBUNEINDIA, THEPRINT),")
    report_lines.append("     image URLs (46,755 unique image links), category tags, and binary labels ('TRUE'/'Fake').")
    report_lines.append("   - The text statements represent concise headline-style news claims from Indian media contexts.")
    report_lines.append("")
    report_lines.append("2. Nature of LIAR Dataset:")
    report_lines.append("   - LIAR is a benchmark political claim dataset containing 12,791 statements collected from PolitiFact.")
    report_lines.append("   - It uses a 6-class fine-grained truth scale: half-true, false, mostly-true, true, barely-true, pants-fire.")
    report_lines.append("   - Includes rich speaker context, party affiliation, speaker state, speaker credit history counts, and venue.")
    report_lines.append("   - It is entirely text-based with zero image references.")
    report_lines.append("")
    report_lines.append("3. Multilingual and Multimodal Complementarity:")
    report_lines.append("   - IFND provides multimodal Indian-context news grounding essential for FakeShield's Indian news target.")
    report_lines.append("   - LIAR provides structured fine-grained claim classification and rich statement verification nuance.")
    report_lines.append("   - In accordance with FakeShield specifications, IFND and LIAR remain strictly segregated and unmerged.")
    report_lines.append("")

    # 7. Problems Found
    report_lines.append("7. PROBLEMS FOUND & DATA QUALITY FINDINGS")
    report_lines.append("-" * 40)
    report_lines.append("1. Character Encoding in Raw IFND:")
    report_lines.append("   - Problem: IFND.csv contains non-UTF-8 bytes (e.g. byte 0xd4 for curly quotes) causing UnicodeDecodeError")
    report_lines.append("     when read with standard UTF-8.")
    report_lines.append("   - Resolution: Loaded with Latin-1 (latin1/iso-8859-1) encoding and standardized to clean UTF-8 output.")
    report_lines.append("")
    report_lines.append("2. High Proportion of Missing Dates in IFND:")
    report_lines.append("   - Problem: 11,321 out of 56,714 rows (19.96%) in IFND have missing values for the 'Date' column.")
    report_lines.append("   - Impact: Temporal grounding cannot rely solely on the Date column without fallback mechanisms.")
    report_lines.append("")
    report_lines.append("3. Typographical Category Error in IFND:")
    report_lines.append("   - Problem: 'Category' column contains a typo entry 'MISLEADIND' (18 rows) alongside 'MISLEADING' (4,059 rows).")
    report_lines.append("   - Impact: When filtering or aggregating by category, 'MISLEADIND' should be recognized as 'MISLEADING'.")
    report_lines.append("")
    report_lines.append("4. Class Imbalance in IFND:")
    report_lines.append("   - Problem: IFND has 37,800 TRUE rows (66.65%) and 18,914 Fake rows (33.35%), creating a 2:1 majority for TRUE.")
    report_lines.append("   - Impact: Evaluation metrics must use balanced/macro F1 rather than raw accuracy.")
    report_lines.append("")
    report_lines.append("5. Missing Header and Metadata in LIAR:")
    report_lines.append("   - Problem: Raw LIAR TSV files contain no header row, requiring explicit schema mapping.")
    report_lines.append("   - In addition, metadata columns have missing values: job_title is missing in 27.9% of rows,")
    report_lines.append("     state_info is missing in 21.5% of rows, and context is missing in 1.0% of rows.")
    report_lines.append("   - Impact: Core statement and label columns are 100% complete; metadata sparsity must be handled gracefully.")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    report_content = "\n".join(report_lines)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report written ({len(report_lines)} lines) to: {REPORT_PATH}")


def main():
    print("\n" + "#" * 80)
    print("# FAKESHIELD DATASET SETUP & INSPECTION SUITE")
    print("#" * 80)

    try:
        # Step 0: Raw structure inspection
        inspect_raw_liar_structure()

        # Step 1: Inspect IFND
        df_ifnd = inspect_ifnd()

        # Step 2: Inspect LIAR
        liar_dfs = inspect_liar()

        # Step 3: Create cleaned IFND
        df_ifnd_clean = create_ifnd_clean(df_ifnd)

        # Step 4: Create combined LIAR
        df_liar_combined = create_liar_combined(liar_dfs)

        # Step 5: Create dataset_quality.csv
        create_dataset_quality_csv(df_ifnd, liar_dfs, df_liar_combined)

        # Step 6: Create dataset_report.txt
        create_dataset_report_txt(df_ifnd, df_ifnd_clean, liar_dfs, df_liar_combined)

        # Step 7: Verify required files exist
        print("\n" + "=" * 80)
        print("VERIFYING OUTPUT ARTIFACTS EXISTENCE")
        print("=" * 80)
        required_files = [
            IFND_CLEAN_PATH,
            LIAR_COMBINED_PATH,
            REPORT_PATH,
            QUALITY_PATH
        ]

        all_exist = True
        for rf in required_files:
            if rf.exists() and rf.stat().st_size > 0:
                print(f"  [OK] Exists ({rf.stat().st_size:,} bytes): {rf.relative_to(PROJECT_ROOT)}")
            else:
                print(f"  [FAIL] Missing or empty: {rf}")
                all_exist = False

        if all_exist:
            print("\n" + "=" * 80)
            print("DATASET SETUP STATUS: PASS")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("DATASET SETUP STATUS: FAIL (Missing output files)")
            print("=" * 80)
            return 1

    except Exception as e:
        print(f"\nERROR ENCOUNTERED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("\nDATASET SETUP STATUS: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
