"""Automated tests for FakeShield OCR pipeline (Phase 3).

12 test cases covering:
1.  Image loading
2.  Invalid image handling
3.  Grayscale conversion
4.  Preprocessing output dimensions
5.  OCR wrapper import
6.  Tesseract executable discovery
7.  OCR output structure
8.  Empty OCR handling
9.  OCR quality metrics
10. CER calculation
11. WER calculation
12. Original image preservation

All tests use locally-generated synthetic images — no external data.
"""

import json
import os
import sys
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.image_preprocessing import ImagePreprocessor, PreprocessingConfig
from src.ocr.tesseract_engine import TesseractEngine, OCRConfig, OCRResult
from src.ocr.ocr_quality import OCRQualityAnalyzer, compute_cer, compute_wer
from src.ocr.ocr_pipeline import OCRPipeline


# ─── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_image_dir(tmp_path_factory):
    """Create temporary test images for the session."""
    d = tmp_path_factory.mktemp("ocr_test_images")

    # Clean text image
    img = Image.new("L", (400, 100), color=255)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Hello World", fill=0)
    img.save(str(d / "clean.png"))

    # Blank image
    blank = Image.new("L", (200, 100), color=255)
    blank.save(str(d / "blank.png"))

    # Tiny image
    tiny = Image.new("L", (5, 5), color=255)
    tiny.save(str(d / "tiny.png"))

    # Color image (RGB)
    color = Image.new("RGB", (200, 100), color=(128, 64, 32))
    from PIL import ImageDraw as ID2
    draw2 = ID2.Draw(color)
    draw2.text((10, 10), "Color", fill=(255, 255, 255))
    color.save(str(d / "color.png"))

    return d


@pytest.fixture
def preprocessor():
    """Default preprocessor instance."""
    return ImagePreprocessor(PreprocessingConfig())


@pytest.fixture
def engine():
    """Default Tesseract engine instance."""
    return TesseractEngine(OCRConfig(language="eng"))


@pytest.fixture
def quality_analyzer():
    """Default quality analyzer instance."""
    return OCRQualityAnalyzer()


# ─── Test 1: Image Loading ──────────────────────────────────────────────

def test_image_loading(preprocessor, test_image_dir):
    """Test 1: Valid image loads successfully as numpy array."""
    img = preprocessor.load_image(str(test_image_dir / "clean.png"))
    assert isinstance(img, np.ndarray)
    assert img.size > 0
    assert len(img.shape) >= 2


# ─── Test 2: Invalid Image Handling ─────────────────────────────────────

def test_invalid_image_handling(preprocessor, test_image_dir):
    """Test 2: Corrupt/missing path raises appropriate error."""
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        preprocessor.load_image(str(test_image_dir / "nonexistent.png"))

    # Create a file that is not a valid image
    bad_path = test_image_dir / "corrupt.png"
    bad_path.write_text("this is not an image")
    with pytest.raises(ValueError):
        preprocessor.load_image(str(bad_path))


# ─── Test 3: Grayscale Conversion ──────────────────────────────────────

def test_grayscale_conversion(preprocessor, test_image_dir):
    """Test 3: RGB image converts to single-channel grayscale."""
    color_img = preprocessor.load_image(str(test_image_dir / "color.png"))
    assert len(color_img.shape) == 3  # BGR
    gray = preprocessor.to_grayscale(color_img)
    assert len(gray.shape) == 2  # single channel


# ─── Test 4: Preprocessing Output Dimensions ───────────────────────────

def test_preprocessing_output_dimensions(preprocessor, test_image_dir):
    """Test 4: Preprocessed output has valid, non-zero dimensions."""
    original, processed, meta = preprocessor.preprocess(
        str(test_image_dir / "clean.png")
    )

    assert processed.shape[0] > 0  # height
    assert processed.shape[1] > 0  # width
    assert len(processed.shape) == 2  # grayscale after preprocessing

    # Border adds pixels
    config = preprocessor.config
    if config.border:
        orig_h, orig_w = original.shape[:2]
        # Processed should be at least as large (border added)
        assert processed.shape[0] >= orig_h
        assert processed.shape[1] >= orig_w


# ─── Test 5: OCR Wrapper Import ────────────────────────────────────────

def test_ocr_wrapper_import():
    """Test 5: All OCR modules import cleanly."""
    from src.ocr import (
        ImagePreprocessor,
        PreprocessingConfig,
        TesseractEngine,
        OCRConfig,
        OCRResult,
        OCRQualityAnalyzer,
        compute_cer,
        compute_wer,
        OCRPipeline,
    )

    assert ImagePreprocessor is not None
    assert TesseractEngine is not None
    assert OCRPipeline is not None


# ─── Test 6: Tesseract Executable Discovery ────────────────────────────

def test_tesseract_executable_discovery(engine):
    """Test 6: Finds Tesseract at known path."""
    assert engine.tesseract_found, "Tesseract executable not found"
    assert engine.tesseract_path is not None
    version = engine.get_version()
    assert "tesseract" in version.lower() or "v" in version.lower()


# ─── Test 7: OCR Output Structure ──────────────────────────────────────

def test_ocr_output_structure(test_image_dir):
    """Test 7: OCR output dict has all required schema keys."""
    pipeline = OCRPipeline()
    result = pipeline.process_image(str(test_image_dir / "clean.png"))

    required_keys = [
        "image_id",
        "image_path",
        "language_requested",
        "image_width",
        "image_height",
        "preprocessing",
        "ocr_text_raw",
        "ocr_text_clean",
        "ocr_engine",
        "tesseract_version",
        "ocr_language",
        "psm",
        "ocr_status",
        "quality_metrics",
    ]

    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert result["ocr_engine"] == "tesseract"
    assert isinstance(result["preprocessing"], dict)
    assert isinstance(result["quality_metrics"], dict)


# ─── Test 8: Empty OCR Handling ────────────────────────────────────────

def test_empty_ocr_handling(test_image_dir):
    """Test 8: Blank image returns empty text without crash."""
    pipeline = OCRPipeline()
    result = pipeline.process_image(str(test_image_dir / "blank.png"))

    assert result["ocr_status"] in ("empty", "success")
    assert result["quality_metrics"]["is_empty"] or result["ocr_text_clean"] == ""


# ─── Test 9: OCR Quality Metrics ───────────────────────────────────────

def test_ocr_quality_metrics(quality_analyzer):
    """Test 9: Metrics dict has expected keys and valid ranges."""
    text = "Hello World 123! This is a test."
    metrics = quality_analyzer.analyze(text)

    expected_keys = [
        "char_count",
        "word_count",
        "alphabetic_ratio",
        "digit_ratio",
        "whitespace_ratio",
        "suspicious_symbol_ratio",
        "avg_word_length",
        "is_empty",
        "line_count",
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"

    assert metrics["char_count"] == len(text)
    assert metrics["word_count"] == 7
    assert 0.0 <= metrics["alphabetic_ratio"] <= 1.0
    assert 0.0 <= metrics["digit_ratio"] <= 1.0
    assert 0.0 <= metrics["whitespace_ratio"] <= 1.0
    assert metrics["is_empty"] is False

    # Test empty text
    empty_metrics = quality_analyzer.analyze("")
    assert empty_metrics["is_empty"] is True
    assert empty_metrics["char_count"] == 0


# ─── Test 10: CER Calculation ──────────────────────────────────────────

def test_cer_calculation():
    """Test 10: CER against known reference/hypothesis pairs."""
    # Perfect match
    assert compute_cer("hello", "hello") == 0.0

    # Known edit distance: 1 substitution in 5 chars = 0.2
    cer = compute_cer("hello", "hallo")
    assert abs(cer - 0.2) < 0.001

    # Both empty
    assert compute_cer("", "") == 0.0

    # Reference empty, hypothesis not
    assert compute_cer("", "text") == float("inf")

    # Completely different
    cer = compute_cer("abc", "xyz")
    assert cer == 1.0  # 3 substitutions / 3 chars


# ─── Test 11: WER Calculation ──────────────────────────────────────────

def test_wer_calculation():
    """Test 11: WER against known reference/hypothesis pairs."""
    # Perfect match
    assert compute_wer("hello world", "hello world") == 0.0

    # Both empty
    assert compute_wer("", "") == 0.0

    # Reference empty
    assert compute_wer("", "text") == float("inf")

    # One word wrong out of two
    wer = compute_wer("hello world", "hello earth")
    assert 0.0 < wer <= 1.0

    # All words wrong
    wer = compute_wer("hello world", "goodbye earth")
    assert wer == 1.0


# ─── Test 12: Original Image Preservation ──────────────────────────────

def test_original_image_preservation(test_image_dir, tmp_path):
    """Test 12: Original file unchanged after pipeline run."""
    # Copy image to temp location
    import shutil
    src = test_image_dir / "clean.png"
    test_copy = tmp_path / "preserve_test.png"
    shutil.copy2(str(src), str(test_copy))

    # Hash before
    hash_before = hashlib.sha256(test_copy.read_bytes()).hexdigest()

    # Run pipeline
    pipeline = OCRPipeline()
    result = pipeline.process_image(
        str(test_copy),
        output_dir=str(tmp_path / "output"),
    )

    # Hash after
    hash_after = hashlib.sha256(test_copy.read_bytes()).hexdigest()

    assert hash_before == hash_after, "Original image was modified!"
