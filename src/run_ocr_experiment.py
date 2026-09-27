"""FakeShield Phase 3 — OCR Experiment Runner.

Generates test images, runs the OCR pipeline, performs raw-vs-preprocessed
comparison, and generates all evaluation reports.

Scope: IMAGE -> PREPROCESSING -> OCR -> TEXT OUTPUT -> QUALITY EVALUATION
Does NOT perform: translation, claim extraction, NLI, retrieval, verdict.
"""

import csv
import json
import platform
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.image_preprocessing import ImagePreprocessor, PreprocessingConfig
from src.ocr.tesseract_engine import TesseractEngine, OCRConfig
from src.ocr.ocr_quality import OCRQualityAnalyzer, compute_cer, compute_wer
from src.ocr.ocr_pipeline import OCRPipeline
from src.ocr.generate_test_images import generate_all, REFERENCES


def main():
    print("#" * 80)
    print("# FAKESHIELD PHASE 3 — OCR EXPERIMENT RUNNER")
    print("#" * 80)

    samples_dir = PROJECT_ROOT / "data" / "ocr_samples"
    processed_dir = PROJECT_ROOT / "data" / "ocr_processed"
    eval_dir = PROJECT_ROOT / "data" / "evaluation"
    ocr_eval_dir = eval_dir / "ocr"

    processed_dir.mkdir(parents=True, exist_ok=True)
    ocr_eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Verify Tesseract ──
    print("\n" + "=" * 80)
    print("STEP 1 — TESSERACT VERIFICATION")
    print("=" * 80)
    engine = TesseractEngine()
    print(f"Tesseract found:     {engine.tesseract_found}")
    print(f"Tesseract path:      {engine.tesseract_path}")
    tess_version = engine.get_version()
    print(f"Tesseract version:   {tess_version}")
    available_langs = engine.get_available_languages()
    print(f"Available languages: {available_langs}")
    tam_available = "tam" in available_langs
    print(f"Tamil (tam) available: {tam_available}")

    if not engine.tesseract_found:
        print("FATAL: Tesseract not found. Cannot proceed.")
        return 1

    # ── Step 2: Generate test images ──
    print("\n" + "=" * 80)
    print("STEP 2 — GENERATE SYNTHETIC TEST IMAGES")
    print("=" * 80)
    image_paths = generate_all(str(samples_dir))

    # Load references
    with open(samples_dir / "references.json", "r", encoding="utf-8") as f:
        references = json.load(f)

    # ── Step 3: Run OCR pipeline on all samples ──
    print("\n" + "=" * 80)
    print("STEP 3 — RUN OCR PIPELINE ON ALL SAMPLES")
    print("=" * 80)

    pipeline = OCRPipeline(
        preprocessing_config=PreprocessingConfig(),
        ocr_config=OCRConfig(language="eng", psm=6),
    )

    results = []
    for image_id, img_path in image_paths.items():
        ref_text = references.get(image_id, None)
        print(f"\nProcessing: {image_id}")
        result = pipeline.process_image(
            image_path=img_path,
            image_id=image_id,
            output_dir=str(processed_dir),
            reference_text=ref_text if ref_text else None,
        )
        results.append(result)
        print(f"  Status:     {result['ocr_status']}")
        print(f"  Text (clean): {result['ocr_text_clean'][:80]}..." if len(result.get('ocr_text_clean', '')) > 80 else f"  Text (clean): {result.get('ocr_text_clean', '')}")
        if "evaluation" in result:
            print(f"  CER: {result['evaluation'].get('cer', 'N/A')}")
            print(f"  WER: {result['evaluation'].get('wer', 'N/A')}")

    # ── Step 4: Raw vs Preprocessed Comparison ──
    print("\n" + "=" * 80)
    print("STEP 4 — RAW VS PREPROCESSED COMPARISON EXPERIMENT")
    print("=" * 80)

    # Only compare images that have reference text (not blank/tiny)
    comparison_ids = ["english_clear", "english_noisy", "english_skewed", "english_low_contrast"]
    comparison_results = []

    for image_id in comparison_ids:
        img_path = image_paths[image_id]
        ref_text = references[image_id]
        print(f"\nComparing: {image_id}")

        comp = pipeline.run_comparison_experiment(
            image_path=img_path,
            image_id=image_id,
            reference_text=ref_text,
            output_dir=str(processed_dir),
        )
        comparison_results.append(comp)

        raw_eval = comp["raw"].get("evaluation", {})
        pre_eval = comp["preprocessed"].get("evaluation", {})
        print(f"  Raw       — CER: {raw_eval.get('cer', 'N/A')}, WER: {raw_eval.get('wer', 'N/A')}, chars: {comp['raw']['quality_metrics']['char_count']}, words: {comp['raw']['quality_metrics']['word_count']}")
        print(f"  Preproc   — CER: {pre_eval.get('cer', 'N/A')}, WER: {pre_eval.get('wer', 'N/A')}, chars: {comp['preprocessed']['quality_metrics']['char_count']}, words: {comp['preprocessed']['quality_metrics']['word_count']}")

    # ── Step 5: Generate ocr_samples.csv ──
    print("\n" + "=" * 80)
    print("STEP 5 — GENERATE EVALUATION CSV")
    print("=" * 80)

    csv_path = eval_dir / "ocr_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "language", "reference_text", "ocr_text",
            "cer", "wer", "preprocessing", "status",
        ])
        writer.writeheader()

        for result in results:
            img_id = result["image_id"]
            ref = references.get(img_id, "")
            ocr_text = result.get("ocr_text_clean", "")
            eval_data = result.get("evaluation", {})

            cer_val = eval_data.get("cer", "") if ref else ""
            wer_val = eval_data.get("wer", "") if ref else ""

            writer.writerow({
                "image_id": img_id,
                "language": "eng",
                "reference_text": ref,
                "ocr_text": ocr_text,
                "cer": cer_val,
                "wer": wer_val,
                "preprocessing": "default",
                "status": result.get("ocr_status", "unknown"),
            })

    print(f"  Generated: {csv_path.relative_to(PROJECT_ROOT)}")

    # ── Step 6: Generate ocr_report.txt ──
    print("\n" + "=" * 80)
    print("STEP 6 — GENERATE OCR REPORT")
    print("=" * 80)

    import importlib.metadata
    import PIL
    import cv2
    import pytesseract
    import jiwer
    jiwer_version = getattr(jiwer, "__version__", None) or importlib.metadata.version("jiwer")

    report_lines = [
        "=" * 80,
        "FAKESHIELD PHASE 3 — OCR & IMAGE PROCESSING REPORT",
        "=" * 80,
        "",
        "1. OBJECTIVE",
        "-" * 40,
        "Build a reproducible OCR pipeline: Image -> Preprocessing -> OCR -> Text Output.",
        "Evaluate OCR quality using CER and WER against synthetic reference images.",
        "",
        "2. ENVIRONMENT",
        "-" * 40,
        f"Python Version:      {platform.python_version()}",
        f"Operating System:    {platform.platform()}",
        f"Tesseract Version:   {tess_version}",
        f"pytesseract Version: {pytesseract.__version__}",
        f"Pillow Version:      {PIL.__version__}",
        f"OpenCV Version:      {cv2.__version__}",
        f"jiwer Version:       {jiwer_version}",
        "",
        "3. TESSERACT VERIFICATION",
        "-" * 40,
        f"Executable:          {engine.tesseract_path}",
        f"Version:             {tess_version}",
        f"Status:              PASS",
        "",
        "4. AVAILABLE OCR LANGUAGES",
        "-" * 40,
        f"Languages:           {available_langs}",
        f"Tamil (tam):         {'AVAILABLE' if tam_available else 'NOT INSTALLED'}",
        "",
        "5. IMAGE PREPROCESSING METHODS",
        "-" * 40,
        "Grayscale conversion (cv2.cvtColor BGR2GRAY)",
        "Noise reduction (cv2.medianBlur, kernel=3)",
        "Contrast enhancement (CLAHE, clip=2.0, grid=8x8)",
        "Binarization (Otsu's thresholding)",
        "Deskew (minAreaRect rotation correction) [optional]",
        "Border addition (10px white margin)",
        "Resize/upscale [optional]",
        "",
        "6. OCR CONFIGURATION",
        "-" * 40,
        "Language:            eng",
        "PSM:                 6 (single uniform block)",
        "OEM:                 3 (LSTM + legacy)",
        "Timeout:             60 seconds",
        "",
        "7. EVALUATION DATASET",
        "-" * 40,
        "Type:                Synthetic (programmatically generated)",
        "Images:              6 total",
        "  english_clear      Clean English text",
        "  english_noisy      English text with Gaussian noise",
        "  english_skewed     English text rotated ~3 degrees",
        "  english_low_contrast  English text with reduced contrast",
        "  blank              Empty white image (edge case)",
        "  tiny               5x5 pixel image (edge case)",
        "",
    ]

    # 8. CER results
    report_lines.extend([
        "8. CER RESULTS (Preprocessed)",
        "-" * 40,
    ])
    for r in results:
        e = r.get("evaluation", {})
        cer = e.get("cer", "N/A")
        report_lines.append(f"  {r['image_id']:<25} CER: {cer}")
    report_lines.append("")

    # 9. WER results
    report_lines.extend([
        "9. WER RESULTS (Preprocessed)",
        "-" * 40,
    ])
    for r in results:
        e = r.get("evaluation", {})
        wer = e.get("wer", "N/A")
        report_lines.append(f"  {r['image_id']:<25} WER: {wer}")
    report_lines.append("")

    # 10. Raw vs preprocessed comparison
    report_lines.extend([
        "10. RAW VS PREPROCESSED COMPARISON",
        "-" * 40,
        f"{'Image':<25} {'Raw CER':<12} {'Pre CER':<12} {'Raw WER':<12} {'Pre WER':<12}",
        "-" * 73,
    ])
    for comp in comparison_results:
        img_id = comp["image_id"]
        raw_e = comp["raw"].get("evaluation", {})
        pre_e = comp["preprocessed"].get("evaluation", {})
        report_lines.append(
            f"  {img_id:<23} {str(raw_e.get('cer', 'N/A')):<12} "
            f"{str(pre_e.get('cer', 'N/A')):<12} "
            f"{str(raw_e.get('wer', 'N/A')):<12} "
            f"{str(pre_e.get('wer', 'N/A')):<12}"
        )
    report_lines.append("")

    # 11. Tamil OCR status
    report_lines.extend([
        "11. TAMIL OCR STATUS",
        "-" * 40,
    ])
    if tam_available:
        report_lines.append("Tamil OCR language data (tam): AVAILABLE")
    else:
        report_lines.extend([
            "Tamil OCR language data (tam): NOT INSTALLED",
            "Tamil OCR demonstration cannot be validated in Phase 3.",
            "Only eng and osd language data files are present.",
        ])
    report_lines.append("")

    # 12. Failure cases
    report_lines.extend([
        "12. FAILURE CASES",
        "-" * 40,
    ])
    failures = [r for r in results if r.get("ocr_status") in ("error", "timeout")]
    if failures:
        for f in failures:
            report_lines.append(f"  {f['image_id']}: {f.get('ocr_error', 'Unknown')}")
    else:
        report_lines.append("  No failures detected in evaluation samples.")
    report_lines.append("")

    # 13. Limitations
    report_lines.extend([
        "13. LIMITATIONS",
        "-" * 40,
        "- Synthetic test images only; real newspaper images not tested.",
        "- Tamil (tam) language data not installed; Tamil OCR not evaluated.",
        "- OCR confidence measures character recognition quality, NOT factual accuracy.",
        "- CER/WER depend on font rendering matching Tesseract's trained models.",
        "- No real-world newspaper scan/photo noise patterns tested.",
        "",
    ])

    # 14. Reproducibility
    report_lines.extend([
        "14. REPRODUCIBILITY",
        "-" * 40,
        f"Python:              {platform.python_version()}",
        f"Tesseract:           {tess_version}",
        f"pytesseract:         {pytesseract.__version__}",
        f"Pillow:              {PIL.__version__}",
        f"OpenCV:              {cv2.__version__}",
        f"jiwer:               {jiwer_version}",
        f"OS:                  {platform.platform()}",
        "Random seed:         42 (for noise generation)",
        "PSM:                 6",
        "OEM:                 3",
        "",
    ])

    # 15. Conclusion
    report_lines.extend([
        "15. PHASE 3 CONCLUSION",
        "-" * 40,
        "OCR pipeline implemented and evaluated successfully.",
        "Preprocessing pipeline is configurable and preserves original images.",
        "CER and WER calculated against known reference texts.",
        "Raw vs preprocessed comparison completed.",
        f"Tamil OCR status: {'AVAILABLE' if tam_available else 'NOT INSTALLED — documented as limitation'}",
        "",
        "=" * 80,
        "END OF OCR REPORT",
        "=" * 80,
    ])

    report_path = eval_dir / "ocr_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Generated: {report_path.relative_to(PROJECT_ROOT)}")

    # ── Step 7: Generate reproducibility file ──
    repro_lines = [
        "=" * 80,
        "FAKESHIELD PHASE 3 — OCR REPRODUCIBILITY SPECIFICATION",
        "=" * 80,
        f"Python Version:      {platform.python_version()}",
        f"Tesseract Version:   {tess_version}",
        f"pytesseract Version: {pytesseract.__version__}",
        f"Pillow Version:      {PIL.__version__}",
        f"OpenCV Version:      {cv2.__version__}",
        f"jiwer Version:       {jiwer_version}",
        f"Operating System:    {platform.platform()}",
        f"OCR Language:        eng",
        f"PSM:                 6",
        f"OEM:                 3",
        f"Preprocessing:       grayscale, noise_reduction, contrast_enhancement, threshold, border",
        f"Evaluation Samples:  {list(image_paths.keys())}",
        "",
        "CER/WER Results:",
    ]
    for r in results:
        e = r.get("evaluation", {})
        repro_lines.append(f"  {r['image_id']}: CER={e.get('cer', 'N/A')}, WER={e.get('wer', 'N/A')}")
    repro_lines.extend([
        "",
        "=" * 80,
    ])

    repro_path = ocr_eval_dir / "ocr_reproducibility.txt"
    with open(repro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(repro_lines))
    print(f"  Generated: {repro_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("PHASE 3 OCR EXPERIMENT COMPLETED")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
