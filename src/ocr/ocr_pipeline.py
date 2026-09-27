"""End-to-end OCR pipeline orchestrator.

Coordinates: image loading -> preprocessing -> Tesseract OCR -> quality metrics.
Produces structured JSON-compatible output for each processed image.

Scope: IMAGE -> PREPROCESSING -> OCR -> TEXT OUTPUT
Does NOT perform: translation, claim extraction, NLI, verdict generation.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

from .image_preprocessing import ImagePreprocessor, PreprocessingConfig
from .tesseract_engine import TesseractEngine, OCRConfig, OCRResult
from .ocr_quality import OCRQualityAnalyzer


class OCRPipeline:
    """End-to-end OCR pipeline.

    Usage:
        pipeline = OCRPipeline()
        result = pipeline.process_image("path/to/image.png")
    """

    def __init__(
        self,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        ocr_config: Optional[OCRConfig] = None,
    ):
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.ocr_config = ocr_config or OCRConfig()
        self.preprocessor = ImagePreprocessor(self.preprocessing_config)
        self.engine = TesseractEngine(self.ocr_config)
        self.quality_analyzer = OCRQualityAnalyzer()

    def process_image(
        self,
        image_path: str,
        image_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        reference_text: Optional[str] = None,
        skip_preprocessing: bool = False,
    ) -> dict:
        """Process a single image through the full OCR pipeline.

        Args:
            image_path: Path to input image.
            image_id: Identifier for this image. Defaults to filename stem.
            output_dir: Directory to save processed outputs. If None, no files saved.
            reference_text: Ground-truth text for CER/WER evaluation.
            skip_preprocessing: If True, run OCR on raw image (for comparison).

        Returns:
            Dictionary with complete OCR output matching the project schema.
        """
        path = Path(image_path)
        if image_id is None:
            image_id = path.stem

        total_start = time.time()

        # Load and optionally preprocess
        if skip_preprocessing:
            original = self.preprocessor.load_image(image_path)
            processed = original.copy()
            preprocess_meta = {
                "steps_applied": [],
                "config": {"skip_preprocessing": True},
                "preprocessing_time_seconds": 0.0,
            }
        else:
            original, processed, preprocess_meta = self.preprocessor.preprocess(
                image_path
            )

        # Get image dimensions
        orig_h, orig_w = original.shape[:2]
        proc_h, proc_w = processed.shape[:2]

        # Run OCR
        ocr_result = self.engine.run_ocr(processed)

        # Quality metrics
        quality = self.quality_analyzer.analyze(ocr_result.text_clean)

        # Reference comparison if available
        evaluation = {}
        if reference_text is not None:
            evaluation = self.quality_analyzer.evaluate_against_reference(
                reference_text, ocr_result.text_clean
            )

        total_time = round(time.time() - total_start, 4)

        # Build output schema
        output = {
            "image_id": image_id,
            "image_path": str(path.resolve()),
            "language_requested": self.ocr_config.language,
            "image_width": orig_w,
            "image_height": orig_h,
            "processed_width": proc_w,
            "processed_height": proc_h,
            "preprocessing": {
                "grayscale": self.preprocessing_config.grayscale,
                "resize": self.preprocessing_config.resize,
                "threshold": self.preprocessing_config.threshold,
                "deskew": self.preprocessing_config.deskew,
                "noise_reduction": self.preprocessing_config.noise_reduction,
                "contrast_enhancement": self.preprocessing_config.contrast_enhancement,
                "border": self.preprocessing_config.border,
                "steps_applied": preprocess_meta.get("steps_applied", []),
                "preprocessing_time_seconds": preprocess_meta.get(
                    "preprocessing_time_seconds", 0.0
                ),
            },
            "ocr_text_raw": ocr_result.text_raw,
            "ocr_text_clean": ocr_result.text_clean,
            "ocr_engine": "tesseract",
            "tesseract_version": ocr_result.tesseract_version,
            "ocr_language": self.ocr_config.language,
            "psm": self.ocr_config.psm,
            "oem": self.ocr_config.oem,
            "ocr_confidence_mean": ocr_result.confidence_mean,
            "ocr_status": ocr_result.status,
            "ocr_error": ocr_result.error_message,
            "ocr_processing_time_seconds": ocr_result.processing_time_seconds,
            "quality_metrics": quality,
            "total_processing_time_seconds": total_time,
        }

        if evaluation:
            output["evaluation"] = evaluation

        # Save outputs if output_dir specified
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save original image copy
            orig_save = out_path / f"{image_id}_original.png"
            self.preprocessor.save_image(original, str(orig_save))

            # Save preprocessed image
            proc_save = out_path / f"{image_id}_preprocessed.png"
            self.preprocessor.save_image(processed, str(proc_save))

            # Save metadata JSON
            meta_save = out_path / f"{image_id}_metadata.json"
            with open(meta_save, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=str)

            output["saved_files"] = {
                "original": str(orig_save),
                "preprocessed": str(proc_save),
                "metadata": str(meta_save),
            }

        return output

    def process_batch(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        reference_texts: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Process multiple images through the pipeline.

        Args:
            image_paths: List of image file paths.
            image_ids: Optional list of IDs (same length as image_paths).
            output_dir: Directory to save all outputs.
            reference_texts: Dict mapping image_id -> reference text.

        Returns:
            List of output dictionaries.
        """
        results = []
        for i, img_path in enumerate(image_paths):
            img_id = image_ids[i] if image_ids else None
            ref_text = None
            if reference_texts and img_id and img_id in reference_texts:
                ref_text = reference_texts[img_id]

            try:
                result = self.process_image(
                    image_path=img_path,
                    image_id=img_id,
                    output_dir=output_dir,
                    reference_text=ref_text,
                )
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "image_id": img_id or Path(img_path).stem,
                        "image_path": img_path,
                        "ocr_status": "error",
                        "ocr_error": str(e),
                    }
                )

        return results

    def run_comparison_experiment(
        self,
        image_path: str,
        image_id: Optional[str] = None,
        reference_text: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Run raw vs preprocessed OCR comparison on a single image.

        Args:
            image_path: Path to input image.
            image_id: Identifier for this image.
            reference_text: Ground-truth text for evaluation.
            output_dir: Directory to save outputs.

        Returns:
            Dictionary with 'raw' and 'preprocessed' results.
        """
        if image_id is None:
            image_id = Path(image_path).stem

        # Run A: Raw (no preprocessing)
        raw_result = self.process_image(
            image_path=image_path,
            image_id=f"{image_id}_raw",
            output_dir=output_dir,
            reference_text=reference_text,
            skip_preprocessing=True,
        )

        # Run B: Preprocessed
        preprocessed_result = self.process_image(
            image_path=image_path,
            image_id=f"{image_id}_preprocessed",
            output_dir=output_dir,
            reference_text=reference_text,
            skip_preprocessing=False,
        )

        return {
            "image_id": image_id,
            "raw": raw_result,
            "preprocessed": preprocessed_result,
        }
