#!/usr/bin/env python
"""
run_dataset_pipeline.py - Master Pipeline Execution Script for FakeShield

Coordinates and executes:
1. Raw dataset hashing before processing
2. Safe preprocessing and normalization for IFND and LIAR
3. Quality tagging, duplicate analysis, and date normalization
4. Grouped stratified train/validation/test splitting
5. Generation of all audits, reports, and hashes after processing
6. Integrity checks comparing before and after hashes
"""

import sys
from pathlib import Path

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from data_loader.preprocess_datasets import run_preprocessing


def main():
    print("#" * 80)
    print("# FAKESHIELD: MASTER DATASET PREPROCESSING PIPELINE")
    print("#" * 80)
    try:
        run_preprocessing()
        print("\nMaster dataset preprocessing pipeline completed successfully.")
        return 0
    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
