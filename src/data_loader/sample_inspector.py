"""
sample_inspector.py - Helper to display samples for the manual verification checklist.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

def main():
    df = pd.read_csv(PROCESSED_DIR / "ifnd_clean.csv", dtype={"label_original": str}, low_memory=False)

    print("=" * 80)
    print("MANUAL CHECK 4 & 5: 20 RANDOM IFND SAMPLES (ORIGINAL vs NORMALIZED)")
    print("=" * 80)
    sample20 = df.sample(20, random_state=42)
    for _, r in sample20.iterrows():
        print(f"ID: {r['id']} | Label: {r['label_original']} (binary: {r['label_binary']}) | Date: {r['date_original']} -> {r['date_normalized']} ({r['date_parse_status']}) | Cat: {r['category_original']} ({r['category_quality_flag']})")
        print(f"  ORIG: {r['original_statement']}")
        print(f"  NORM: {r['normalized_statement']}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("MANUAL CHECK 6: 10 DUPLICATE STATEMENT EXAMPLES (LABEL AGREEMENT / CONFLICT)")
    print("=" * 80)
    dup_df = pd.read_csv(EVAL_DIR / "duplicate_analysis.csv")
    for _, r in dup_df.head(10).iterrows():
        print(f"Type: {r['duplicate_type']} | Count: {r['occurrences']} | Labels: {r['labels_found']} | IDs: {r['ids']} | Sources: {r['sources']}")
        print(f"  Statement: {r['normalized_statement']}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("MANUAL CHECK 7 & 8: 20 RANDOM LIAR SAMPLES (STATEMENT & LABEL ALIGNMENT)")
    print("=" * 80)
    liar = pd.read_csv(PROCESSED_DIR / "liar_combined.csv", low_memory=False)
    for _, r in liar.sample(20, random_state=42).iterrows():
        print(f"ID: {r['id']} | Split: {r['split']} | Label: {r['label']} | Speaker: {r['speaker']} | Venue: {r['context']}")
        print(f"  ORIG: {r['statement_original']}")
        print(f"  NORM: {r['statement_normalized']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
