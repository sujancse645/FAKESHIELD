"""FakeShield OCR Pipeline Package.

Provides image preprocessing, Tesseract OCR wrapper, quality metrics,
and end-to-end OCR pipeline for news image text extraction.
"""

from .image_preprocessing import ImagePreprocessor, PreprocessingConfig
from .tesseract_engine import TesseractEngine, OCRConfig, OCRResult
from .ocr_quality import OCRQualityAnalyzer, compute_cer, compute_wer
from .ocr_pipeline import OCRPipeline

__all__ = [
    "ImagePreprocessor",
    "PreprocessingConfig",
    "TesseractEngine",
    "OCRConfig",
    "OCRResult",
    "OCRQualityAnalyzer",
    "compute_cer",
    "compute_wer",
    "OCRPipeline",
]
