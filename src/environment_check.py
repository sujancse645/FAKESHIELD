#!/usr/bin/env python
"""
environment_check.py - FakeShield Package Import and Version Verification Script

Tests importing all required FakeShield packages and prints their installed versions
and import status (PASS/FAIL).
"""

import sys

PACKAGES_TO_TEST = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("pytesseract", "pytesseract"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("sentencepiece", "sentencepiece"),
    ("jiwer", "jiwer"),
    ("requests", "requests"),
    ("dotenv", "python-dotenv"),
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
    ("langdetect", "langdetect"),
]


def test_package_imports():
    print("=" * 65)
    print(f"{'PACKAGE':<18} {'VERSION':<18} {'IMPORT STATUS'}")
    print("=" * 65)

    all_passed = True
    results = []

    for import_name, dist_name in PACKAGES_TO_TEST:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", None)
            if version is None:
                try:
                    import importlib.metadata
                    version = importlib.metadata.version(dist_name)
                except Exception:
                    version = "UNKNOWN"
            status = "PASS"
            print(f"{import_name:<18} {str(version):<18} {status}")
            results.append((import_name, str(version), status))
        except Exception as e:
            status = "FAIL"
            print(f"{import_name:<18} {'N/A':<18} {status} ({e})")
            results.append((import_name, "N/A", status))
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("ALL REQUIRED PACKAGES IMPORTED SUCCESSFULLY: PASS")
    else:
        print("ONE OR MORE PACKAGE IMPORTS FAILED: FAIL")
    print("=" * 65)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(test_package_imports())
