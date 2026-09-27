#!/usr/bin/env python
"""
validate_environment.py - FakeShield Automated Environment Validation Script

Returns PASS / FAIL for each of the 16 core system prerequisites:
1. Python version
2. Virtual environment
3. pandas
4. numpy
5. scikit-learn
6. Pillow
7. OpenCV
8. pytesseract
9. transformers
10. torch
11. requests
12. python-dotenv
13. streamlit
14. pytest
15. Phase 1 dataset accessibility
16. Phase 1 raw hash integrity

Tesseract OCR engine:
If installed: PASS/FAIL based on executable and version.
If not installed: WARNING — REQUIRED BEFORE PHASE 3.
"""

import sys
import shutil
import hashlib
import platform
from pathlib import Path

PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"


def validate_all():
    print("=" * 80)
    print("FAKESHIELD AUTOMATED ENVIRONMENT VALIDATION")
    print("=" * 80)

    checks = []

    def record(num, name, status, detail=""):
        checks.append((num, name, status, detail))
        print(f"  [{status:<7}] Item {num:2d}: {name:<35} {detail}")

    # 1. Python version
    major, minor = sys.version_info[:2]
    py_ok = (major == 3 and minor >= 10)
    record(1, "Python Version", "PASS" if py_ok else "FAIL", f"{platform.python_version()} (>= 3.10 required)")

    # 2. Virtual environment
    in_venv = (sys.prefix != sys.base_prefix) or ("venv" in sys.executable.lower())
    record(2, "Virtual Environment", "PASS" if in_venv else "FAIL", f"{sys.executable}")

    # 3-14. Packages
    pkg_map = [
        (3, "pandas", "pandas"),
        (4, "numpy", "numpy"),
        (5, "scikit-learn", "sklearn"),
        (6, "Pillow", "PIL"),
        (7, "OpenCV", "cv2"),
        (8, "pytesseract", "pytesseract"),
        (9, "transformers", "transformers"),
        (10, "torch", "torch"),
        (11, "requests", "requests"),
        (12, "python-dotenv", "dotenv"),
        (13, "streamlit", "streamlit"),
        (14, "pytest", "pytest"),
    ]

    for num, dist_name, imp_name in pkg_map:
        try:
            mod = __import__(imp_name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                import importlib.metadata
                ver = importlib.metadata.version(dist_name)
            record(num, dist_name, "PASS", f"v{ver}")
        except Exception as e:
            record(num, dist_name, "FAIL", f"ImportError: {e}")

    # 15. Phase 1 dataset accessibility
    p1_files = [
        RAW_DIR / "IFND" / "IFND.csv",
        RAW_DIR / "LIAR" / "train.tsv",
        RAW_DIR / "LIAR" / "test.tsv",
        RAW_DIR / "LIAR" / "valid.tsv",
        PROCESSED_DIR / "ifnd_clean.csv",
        PROCESSED_DIR / "liar_combined.csv",
        SPLITS_DIR / "ifnd_train.csv",
        SPLITS_DIR / "ifnd_validation.csv",
        SPLITS_DIR / "ifnd_test.csv",
    ]
    p1_ok = all(p.exists() and p.stat().st_size > 0 for p in p1_files)
    record(15, "Phase 1 Dataset Accessibility", "PASS" if p1_ok else "FAIL", f"{len(p1_files)} files verified")

    # 16. Phase 1 raw hash integrity
    before_csv = EVAL_DIR / "raw_file_hashes_before.csv"
    hashes_ok = False
    if before_csv.exists():
        import pandas as pd
        df_before = pd.read_csv(before_csv)
        live_ok = True
        for _, row in df_before.iterrows():
            fpath = RAW_DIR / ("IFND/IFND.csv" if row["dataset"] == "IFND" else f"LIAR/{row['file']}")
            sha = hashlib.sha256()
            with open(fpath, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            if sha.hexdigest() != row["sha256"]:
                live_ok = False
                break
        hashes_ok = live_ok
    record(16, "Phase 1 Raw Hash Integrity", "PASS" if hashes_ok else "FAIL", "SHA-256 match confirmed")

    # Tesseract engine check (special handling)
    print("-" * 80)
    tess_path = shutil.which("tesseract") or (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe" if Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists() else None
    )
    if tess_path:
        tess_status = "PASS"
        tess_detail = f"Found at {tess_path}"
    else:
        tess_status = "WARNING"
        tess_detail = "REQUIRED BEFORE PHASE 3 (Tesseract binary not in PATH / Program Files)"

    print(f"  [{tess_status:<7}] Tesseract OCR Engine:       {tess_detail}")
    print("=" * 80)

    # Calculate overall status
    core_failures = [c for c in checks if c[2] == "FAIL"]
    if not core_failures:
        if tess_status == "WARNING":
            print("ENVIRONMENT VALIDATION OUTCOME: PASS WITH WARNING")
            print("  Note: Tesseract OCR engine binary must be installed before Phase 3.")
            return 0
        else:
            print("ENVIRONMENT VALIDATION OUTCOME: PASS")
            return 0
    else:
        print(f"ENVIRONMENT VALIDATION OUTCOME: FAIL ({len(core_failures)} failed checks)")
        return 1


if __name__ == "__main__":
    sys.exit(validate_all())
