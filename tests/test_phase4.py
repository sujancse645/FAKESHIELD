"""test_phase4.py - Automated Pytest Suite for FakeShield Phase 4.

Verifies:
1. Result CSV schema (phase4_results.csv)
2. Ablation CSV schema (phase4_ablation.csv)
3. Error analysis CSV schema (phase4_error_analysis.csv)
4. Valid non-negative CER values
5. Valid non-negative WER values
6. Valid OCR confidence ranges ([0, 100] or -1.0)
7. Required sample IDs evaluated
8. Raw and preprocessed conditions present
9. Tamil result exists for both tam and tam+eng
10. Edge-case handling (blank and tiny recorded as N/A)
11. Phase 1 raw hash integrity
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_OCR_DIR = PROJECT_ROOT / "data" / "evaluation" / "ocr"
SAMPLES_DIR = PROJECT_ROOT / "data" / "ocr_samples"


@pytest.fixture(scope="session")
def results_rows():
    csv_p = EVAL_OCR_DIR / "phase4_results.csv"
    assert csv_p.exists(), f"{csv_p} does not exist"
    with open(csv_p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def ablation_rows():
    csv_p = EVAL_OCR_DIR / "phase4_ablation.csv"
    assert csv_p.exists(), f"{csv_p} does not exist"
    with open(csv_p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def error_rows():
    csv_p = EVAL_OCR_DIR / "phase4_error_analysis.csv"
    assert csv_p.exists(), f"{csv_p} does not exist"
    with open(csv_p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_phase4_results_schema(results_rows):
    """Test 1: phase4_results.csv schema and completeness."""
    required = [
        "sample_id", "language", "condition", "preprocessing_config",
        "cer", "wer", "ocr_confidence", "character_count", "word_count",
        "processing_time_seconds", "status"
    ]
    assert len(results_rows) >= 14, f"Expected at least 14 results rows, got {len(results_rows)}"
    for r in results_rows:
        for f in required:
            assert f in r, f"Missing column '{f}' in results row: {r}"


def test_phase4_ablation_schema(ablation_rows):
    """Test 2: phase4_ablation.csv schema and completeness."""
    required = [
        "sample_id", "language", "preprocessing_variant", "cer",
        "wer", "ocr_confidence", "processing_time_seconds", "status"
    ]
    assert len(ablation_rows) >= 20, f"Expected at least 20 ablation rows, got {len(ablation_rows)}"
    for r in ablation_rows:
        for f in required:
            assert f in r, f"Missing column '{f}' in ablation row: {r}"


def test_phase4_error_analysis_schema(error_rows):
    """Test 3: phase4_error_analysis.csv schema and content."""
    required = [
        "sample_id", "language", "raw_or_preprocessed", "error_type",
        "reference_fragment", "ocr_fragment", "notes"
    ]
    assert len(error_rows) >= 1
    for r in error_rows:
        for f in required:
            assert f in r, f"Missing column '{f}' in error row: {r}"


def test_cer_valid_ranges(results_rows, ablation_rows):
    """Test 4: Valid non-negative CER values across results and ablation."""
    for r in results_rows + ablation_rows:
        val = r["cer"]
        if val != "N/A":
            c = float(val)
            assert c >= 0.0, f"Negative CER {c} found in row {r}"


def test_wer_valid_ranges(results_rows, ablation_rows):
    """Test 5: Valid non-negative WER values across results and ablation."""
    for r in results_rows + ablation_rows:
        val = r["wer"]
        if val != "N/A":
            w = float(val)
            assert w >= 0.0, f"Negative WER {w} found in row {r}"


def test_confidence_range(results_rows, ablation_rows):
    """Test 6: Valid OCR confidence ranges."""
    for r in results_rows + ablation_rows:
        val = r.get("ocr_confidence")
        if val is not None and val != "N/A":
            conf = float(val)
            assert conf == -1.0 or (0.0 <= conf <= 100.0), f"Invalid confidence {conf} in row {r}"


def test_required_sample_ids(results_rows):
    """Test 7: Required sample IDs are present in evaluation results."""
    sids = {r["sample_id"] for r in results_rows}
    expected = {
        "english_clear", "english_noisy", "english_skewed",
        "english_low_contrast", "blank", "tiny", "tamil_news_sample"
    }
    assert expected.issubset(sids), f"Missing sample IDs: {expected - sids}"


def test_raw_preprocessed_conditions(results_rows):
    """Test 8: Both raw and preprocessed conditions exist for each sample."""
    samples = set(r["sample_id"] for r in results_rows)
    for s in samples:
        conds = {r["condition"] for r in results_rows if r["sample_id"] == s}
        assert "raw" in conds, f"Missing 'raw' condition for {s}"
        assert "preprocessed" in conds, f"Missing 'preprocessed' condition for {s}"


def test_tamil_result_exists(results_rows):
    """Test 9: Tamil results exist under both 'tam' and 'tam+eng' configurations."""
    tam_rows = [r for r in results_rows if r["sample_id"] == "tamil_news_sample"]
    langs = {r["language"] for r in tam_rows}
    assert "tam" in langs, "Missing 'tam' evaluation for tamil_news_sample"
    assert "tam+eng" in langs, "Missing 'tam+eng' evaluation for tamil_news_sample"
    for r in tam_rows:
        assert r["status"] == "success"
        assert float(r["ocr_confidence"]) > 80.0


def test_edge_case_handling(results_rows):
    """Test 10: Blank and tiny images are handled safely without crashing and with N/A CER/WER."""
    blank_rows = [r for r in results_rows if r["sample_id"] == "blank"]
    tiny_rows = [r for r in results_rows if r["sample_id"] == "tiny"]
    assert len(blank_rows) == 2
    assert len(tiny_rows) == 2
    for r in blank_rows + tiny_rows:
        assert r["cer"] == "N/A"
        assert r["wer"] == "N/A"
        assert int(r["character_count"]) == 0
        assert int(r["word_count"]) == 0
        assert r["status"] in ("empty", "success")


def test_phase1_hash_integrity():
    """Test 11: Phase 1 raw dataset files have unchanged SHA-256 hashes."""
    stored_hashes = [
        ("data/raw/IFND/IFND.csv", "3728c7b785ea93c9"),
        ("data/raw/LIAR/train.tsv", "23ef8cb7ff57b25c"),
        ("data/raw/LIAR/test.tsv", "ee41e2c741e00192"),
        ("data/raw/LIAR/valid.tsv", "8dd3d7ed1a11e78e"),
    ]
    for frel, prefix in stored_hashes:
        fpath = PROJECT_ROOT / frel
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        assert sha.hexdigest().startswith(prefix), f"Hash mismatch for {frel}"
