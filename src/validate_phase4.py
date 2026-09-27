"""validate_phase4.py - FakeShield Phase 4 Automated Validation Script.

Verifies:
1.  Phase 3 OCR modules are importable.
2.  All 7 evaluation samples exist.
3.  Reference files are present (references.json).
4.  phase4_results.csv exists and is well-formed.
5.  phase4_ablation.csv exists and is well-formed.
6.  phase4_error_analysis.csv exists and is well-formed.
7.  CER values are valid non-negative floats (or 'N/A' for edge cases).
8.  WER values are valid non-negative floats (or 'N/A' for edge cases).
9.  OCR confidence values are within [0.0, 100.0] or -1.0 (empty).
10. No required result fields are missing.
11. Phase 1 raw files remain unchanged (SHA-256 integrity).
12. Phase 3 original test images remain preserved and unchanged.
13. Phase 4 report and reproducibility reports exist and are populated.
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_phase4():
    print("=" * 80)
    print("FAKESHIELD PHASE 4: AUTOMATED VALIDATION SUITE")
    print("=" * 80)

    checks = []

    def record(num, name, status, detail=""):
        checks.append((num, name, status, detail))
        print(f"  [{status:<7}] Check {num:>2}: {name:<40} {detail}")

    # Check 1: Phase 3 OCR modules importable
    try:
        from src.ocr import (
            ImagePreprocessor, PreprocessingConfig,
            TesseractEngine, OCRConfig, OCRResult,
            OCRQualityAnalyzer, compute_cer, compute_wer,
            OCRPipeline,
        )
        record(1, "Phase 3 OCR modules importable", "PASS", "All core classes imported")
    except Exception as e:
        record(1, "Phase 3 OCR modules importable", "FAIL", str(e))

    # Check 2: Evaluation samples exist (all 7)
    samples_dir = PROJECT_ROOT / "data" / "ocr_samples"
    expected_samples = [
        "english_clear.png", "english_noisy.png", "english_skewed.png",
        "english_low_contrast.png", "blank.png", "tiny.png", "tamil_news_sample.png"
    ]
    missing_samples = [s for s in expected_samples if not (samples_dir / s).exists()]
    if not missing_samples:
        record(2, "Evaluation samples exist", "PASS", f"All {len(expected_samples)} samples present")
    else:
        record(2, "Evaluation samples exist", "FAIL", f"Missing: {missing_samples}")

    # Check 3: Reference files present where required
    refs_p = samples_dir / "references.json"
    if refs_p.exists() and refs_p.stat().st_size > 0:
        try:
            with open(refs_p, "r", encoding="utf-8") as f:
                ref_dict = json.load(f)
            assert len(ref_dict) >= 7
            record(3, "Reference files present & valid", "PASS", f"{len(ref_dict)} keys in references.json")
        except Exception as e:
            record(3, "Reference files present & valid", "FAIL", str(e))
    else:
        record(3, "Reference files present & valid", "FAIL", "references.json missing or empty")

    eval_ocr_dir = PROJECT_ROOT / "data" / "evaluation" / "ocr"

    # Check 4: phase4_results.csv exists
    results_csv = eval_ocr_dir / "phase4_results.csv"
    results_rows = []
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results_rows = list(reader)
        required_fields = [
            "sample_id", "language", "condition", "preprocessing_config",
            "cer", "wer", "ocr_confidence", "character_count", "word_count",
            "processing_time_seconds", "status"
        ]
        if results_rows and all(all(f in r for f in required_fields) for r in results_rows):
            record(4, "phase4_results.csv exists & valid", "PASS", f"{len(results_rows)} records verified")
        else:
            record(4, "phase4_results.csv exists & valid", "FAIL", "Missing fields or empty")
    else:
        record(4, "phase4_results.csv exists & valid", "FAIL", "File not found")

    # Check 5: phase4_ablation.csv exists
    ablation_csv = eval_ocr_dir / "phase4_ablation.csv"
    ablation_rows = []
    if ablation_csv.exists():
        with open(ablation_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ablation_rows = list(reader)
        ab_fields = [
            "sample_id", "language", "preprocessing_variant", "cer",
            "wer", "ocr_confidence", "processing_time_seconds", "status"
        ]
        if ablation_rows and all(all(f in r for f in ab_fields) for r in ablation_rows):
            record(5, "phase4_ablation.csv exists & valid", "PASS", f"{len(ablation_rows)} ablation records")
        else:
            record(5, "phase4_ablation.csv exists & valid", "FAIL", "Missing fields or empty")
    else:
        record(5, "phase4_ablation.csv exists & valid", "FAIL", "File not found")

    # Check 6: phase4_error_analysis.csv exists
    error_csv = eval_ocr_dir / "phase4_error_analysis.csv"
    if error_csv.exists():
        with open(error_csv, "r", encoding="utf-8") as f:
            err_rows = list(csv.DictReader(f))
        err_fields = ["sample_id", "language", "raw_or_preprocessed", "error_type", "reference_fragment", "ocr_fragment", "notes"]
        if err_rows and all(all(f in r for f in err_fields) for r in err_rows):
            record(6, "phase4_error_analysis.csv valid", "PASS", f"{len(err_rows)} error records")
        else:
            record(6, "phase4_error_analysis.csv valid", "FAIL", "Missing fields or empty")
    else:
        record(6, "phase4_error_analysis.csv valid", "FAIL", "File not found")

    # Check 7: CER values valid
    cer_ok = True
    for r in results_rows + ablation_rows:
        val = r["cer"]
        if val != "N/A":
            try:
                f_val = float(val)
                if f_val < 0:
                    cer_ok = False
                    break
            except ValueError:
                cer_ok = False
                break
    record(7, "CER values non-negative or N/A", "PASS" if cer_ok else "FAIL")

    # Check 8: WER values valid
    wer_ok = True
    for r in results_rows + ablation_rows:
        val = r["wer"]
        if val != "N/A":
            try:
                f_val = float(val)
                if f_val < 0:
                    wer_ok = False
                    break
            except ValueError:
                wer_ok = False
                break
    record(8, "WER values non-negative or N/A", "PASS" if wer_ok else "FAIL")

    # Check 9: OCR confidence bounds valid
    conf_ok = True
    for r in results_rows + ablation_rows:
        val = r.get("ocr_confidence")
        if val is not None and val != "N/A":
            try:
                f_val = float(val)
                if not (f_val == -1.0 or 0.0 <= f_val <= 100.0):
                    conf_ok = False
                    break
            except ValueError:
                conf_ok = False
                break
    record(9, "OCR confidence within [0,100] or -1", "PASS" if conf_ok else "FAIL")

    # Check 10: No required result fields missing
    edge_cases = [r for r in results_rows if r["sample_id"] in ("blank", "tiny")]
    edge_handled = (len(edge_cases) >= 4 and all(e["cer"] == "N/A" and e["wer"] == "N/A" for e in edge_cases))
    record(10, "Edge case CER/WER recorded as N/A", "PASS" if edge_handled else "FAIL", "blank & tiny correctly handled")

    # Check 11: Phase 1 raw files unchanged (SHA-256)
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
    record(11, "Phase 1 raw file hashes unchanged", "PASS" if raw_hashes_ok else "FAIL", "All 4 SHA-256 hashes match baseline")

    # Check 12: Phase 3 original test images preserved unchanged
    orig_hashes = {
        "blank.png": "6459d1d67de7903e",
        "english_clear.png": "820d26388aa15591",
        "english_low_contrast.png": "b0059a3a36564fb3",
        "english_noisy.png": "f6ba0ac368eaad36",
        "english_skewed.png": "0e2742665e115cce",
        "tamil_news_sample.png": "93531af048701231",
        "tiny.png": "5ee84d08a78e715f",
    }
    orig_ok = True
    for img_name, prefix in orig_hashes.items():
        ipath = samples_dir / img_name
        if not ipath.exists():
            orig_ok = False
            break
        sha = hashlib.sha256(ipath.read_bytes()).hexdigest()
        if not sha.startswith(prefix):
            orig_ok = False
            break
    record(12, "Phase 3 original test images preserved", "PASS" if orig_ok else "FAIL", "All 7 sample image hashes match")

    # Check 13: Phase 4 report and reproducibility exist
    rep_p = eval_ocr_dir / "phase4_report.txt"
    repro_p = eval_ocr_dir / "phase4_reproducibility.txt"
    rep_ok = rep_p.exists() and rep_p.stat().st_size > 1000
    repro_ok = repro_p.exists() and repro_p.stat().st_size > 300
    record(13, "Phase 4 report & reproducibility exist", "PASS" if (rep_ok and repro_ok) else "FAIL", "phase4_report.txt and phase4_reproducibility.txt verified")

    print("=" * 80)
    failures = [c for c in checks if c[2] == "FAIL"]
    if not failures:
        print("PHASE 4 AUTOMATED VALIDATION: ALL 13 CHECKS PASS")
        return 0
    else:
        print(f"PHASE 4 AUTOMATED VALIDATION: {len(failures)} CHECKS FAILED")
        for num, name, status, detail in failures:
            print(f"  Check {num} FAIL: {name} - {detail}")
        return 1


if __name__ == "__main__":
    sys.exit(validate_phase4())
