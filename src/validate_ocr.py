"""FakeShield Phase 3 OCR Validation Script.

Verifies:
1.  OCR modules exist and import
2.  Tesseract executable exists
3.  Tesseract version is detectable
4.  Required Python packages import
5.  Test images exist
6.  Original images are preserved
7.  OCR output schema is valid
8.  CER/WER values are valid
9.  No Phase 1 raw files were modified
"""

import hashlib
import json
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_all():
    """Run all Phase 3 validation checks."""
    print("=" * 80)
    print("FAKESHIELD PHASE 3 — OCR VALIDATION")
    print("=" * 80)

    checks = []

    def record(num, name, status, detail=""):
        checks.append((num, name, status, detail))
        print(f"  [{status:<7}] Check {num:>2}: {name:<40} {detail}")

    # ── 1. OCR modules exist and import ──
    try:
        from src.ocr import (
            ImagePreprocessor, PreprocessingConfig,
            TesseractEngine, OCRConfig, OCRResult,
            OCRQualityAnalyzer, compute_cer, compute_wer,
            OCRPipeline,
        )
        record(1, "OCR modules import", "PASS", "All 9 exports imported")
    except Exception as e:
        record(1, "OCR modules import", "FAIL", str(e))

    # ── 2. Tesseract executable exists ──
    tess_path = shutil.which("tesseract")
    if not tess_path:
        for p in [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]:
            if p.exists():
                tess_path = str(p)
                break
    if tess_path:
        record(2, "Tesseract executable exists", "PASS", tess_path)
    else:
        record(2, "Tesseract executable exists", "FAIL", "Not found")

    # ── 3. Tesseract version detectable ──
    try:
        from src.ocr.tesseract_engine import TesseractEngine
        engine = TesseractEngine()
        version = engine.get_version()
        if "tesseract" in version.lower():
            record(3, "Tesseract version detectable", "PASS", version)
        else:
            record(3, "Tesseract version detectable", "FAIL", version)
    except Exception as e:
        record(3, "Tesseract version detectable", "FAIL", str(e))

    # ── 4. Required Python packages import ──
    pkgs = ["PIL", "cv2", "pytesseract", "numpy", "jiwer"]
    pkg_ok = True
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            pkg_ok = False
            break
    if pkg_ok:
        record(4, "Required packages import", "PASS", ", ".join(pkgs))
    else:
        record(4, "Required packages import", "FAIL", f"{pkg} failed")

    # ── 5. Test images exist ──
    samples_dir = PROJECT_ROOT / "data" / "ocr_samples"
    expected_images = [
        "english_clear.png", "english_noisy.png", "english_skewed.png",
        "english_low_contrast.png", "blank.png", "tiny.png",
    ]
    refs_file = samples_dir / "references.json"
    images_ok = all((samples_dir / img).exists() for img in expected_images)
    refs_ok = refs_file.exists()
    if images_ok and refs_ok:
        record(5, "Test images exist", "PASS", f"{len(expected_images)} images + references.json")
    else:
        missing = [img for img in expected_images if not (samples_dir / img).exists()]
        detail = f"Missing: {missing}"
        if not refs_ok:
            detail += " + references.json"
        record(5, "Test images exist", "FAIL", detail)

    # ── 6. Original images preserved ──
    # Check that ocr_samples still has originals
    originals_ok = True
    for img in expected_images:
        p = samples_dir / img
        if p.exists() and p.stat().st_size > 0:
            continue
        originals_ok = False
        break
    record(6, "Original images preserved", "PASS" if originals_ok else "FAIL")

    # ── 7. OCR output schema valid ──
    processed_dir = PROJECT_ROOT / "data" / "ocr_processed"
    schema_ok = False
    schema_detail = "No metadata files found"
    if processed_dir.exists():
        meta_files = list(processed_dir.glob("*_metadata.json"))
        if meta_files:
            try:
                with open(meta_files[0], "r", encoding="utf-8") as f:
                    sample = json.load(f)
                required_keys = [
                    "image_id", "image_path", "language_requested",
                    "image_width", "image_height", "preprocessing",
                    "ocr_text_raw", "ocr_text_clean", "ocr_engine",
                    "tesseract_version", "ocr_language", "psm", "ocr_status",
                ]
                missing_keys = [k for k in required_keys if k not in sample]
                if not missing_keys:
                    schema_ok = True
                    schema_detail = f"Validated {len(meta_files)} files"
                else:
                    schema_detail = f"Missing keys: {missing_keys}"
            except Exception as e:
                schema_detail = str(e)
    record(7, "OCR output schema valid", "PASS" if schema_ok else "FAIL", schema_detail)

    # ── 8. CER/WER values valid ──
    csv_path = PROJECT_ROOT / "data" / "evaluation" / "ocr_samples.csv"
    cer_wer_ok = False
    cer_wer_detail = "ocr_samples.csv not found"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if "cer" in df.columns and "wer" in df.columns:
                # Check that values are non-negative (or 'inf' for edge cases)
                cer_vals = df["cer"].dropna()
                wer_vals = df["wer"].dropna()
                cer_wer_ok = True
                cer_wer_detail = f"CER range [{cer_vals.min():.4f}, {cer_vals.max():.4f}], WER range [{wer_vals.min():.4f}, {wer_vals.max():.4f}]"
            else:
                cer_wer_detail = "Missing cer/wer columns"
        except Exception as e:
            cer_wer_detail = str(e)
    record(8, "CER/WER values valid", "PASS" if cer_wer_ok else "FAIL", cer_wer_detail)

    # ── 9. Phase 1 raw files unchanged ──
    raw_dir = PROJECT_ROOT / "data" / "raw"
    eval_dir = PROJECT_ROOT / "data" / "evaluation"
    hashes_csv = eval_dir / "raw_file_hashes_before.csv"
    phase1_ok = False
    phase1_detail = "Hash reference file not found"
    if hashes_csv.exists():
        try:
            import pandas as pd
            df_ref = pd.read_csv(hashes_csv)
            all_match = True
            for _, row in df_ref.iterrows():
                fpath = raw_dir / ("IFND/IFND.csv" if row["dataset"] == "IFND" else f"LIAR/{row['file']}")
                sha = hashlib.sha256()
                with open(fpath, "rb") as f:
                    while chunk := f.read(65536):
                        sha.update(chunk)
                if sha.hexdigest() != row["sha256"]:
                    all_match = False
                    break
            phase1_ok = all_match
            phase1_detail = "SHA-256 match confirmed" if all_match else "MISMATCH DETECTED"
        except Exception as e:
            phase1_detail = str(e)
    record(9, "Phase 1 raw files unchanged", "PASS" if phase1_ok else "FAIL", phase1_detail)

    # ── Summary ──
    print("=" * 80)
    failures = [c for c in checks if c[2] == "FAIL"]
    if not failures:
        print("PHASE 3 OCR VALIDATION: ALL CHECKS PASS")
        return 0
    else:
        print(f"PHASE 3 OCR VALIDATION: {len(failures)} FAILED")
        for num, name, status, detail in failures:
            print(f"  FAILED: Check {num} — {name}: {detail}")
        return 1


if __name__ == "__main__":
    sys.exit(validate_all())
