"""Tesseract OCR engine wrapper.

Provides a clean interface to pytesseract with configurable language,
page segmentation mode (PSM), OCR engine mode (OEM), timeout handling,
and structured output.
"""

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import pytesseract
from PIL import Image


# Standard Windows install paths for Tesseract
_WINDOWS_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


@dataclass
class OCRConfig:
    """Configuration for Tesseract OCR.

    Attributes:
        language: Tesseract language code(s), e.g. 'eng', 'tam', 'tam+eng'.
        psm: Page Segmentation Mode (0-13). Default 6 = single uniform block.
        oem: OCR Engine Mode (0-3). Default 3 = LSTM + legacy.
        timeout: Seconds before OCR times out. None = no timeout.
    """

    language: str = "eng"
    psm: int = 6
    oem: int = 3
    timeout: Optional[int] = 60


@dataclass
class OCRResult:
    """Structured result from an OCR run.

    Attributes:
        text_raw: Raw OCR output text.
        text_clean: Cleaned OCR text (stripped, collapsed whitespace).
        language_requested: Language code that was requested.
        language_detected: Language reported by Tesseract (if available).
        confidence_mean: Mean word-level confidence (0-100), or -1 if unavailable.
        confidence_per_word: List of (word, confidence) tuples.
        tesseract_version: Version string of Tesseract engine.
        psm: PSM used.
        oem: OEM used.
        processing_time_seconds: Time taken for OCR (excluding preprocessing).
        status: 'success', 'empty', 'error', or 'timeout'.
        error_message: Error details if status is 'error'.
    """

    text_raw: str = ""
    text_clean: str = ""
    language_requested: str = ""
    language_detected: str = ""
    confidence_mean: float = -1.0
    confidence_per_word: list = field(default_factory=list)
    tesseract_version: str = ""
    psm: int = 6
    oem: int = 3
    processing_time_seconds: float = 0.0
    status: str = "success"
    error_message: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "text_raw": self.text_raw,
            "text_clean": self.text_clean,
            "language_requested": self.language_requested,
            "language_detected": self.language_detected,
            "confidence_mean": self.confidence_mean,
            "confidence_per_word": self.confidence_per_word,
            "tesseract_version": self.tesseract_version,
            "psm": self.psm,
            "oem": self.oem,
            "processing_time_seconds": self.processing_time_seconds,
            "status": self.status,
            "error_message": self.error_message,
        }


class TesseractEngine:
    """Wrapper around pytesseract for Tesseract OCR.

    Handles executable discovery, language validation, and structured output.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._tesseract_path = self._find_tesseract()
        if self._tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_path

    def _find_tesseract(self) -> Optional[str]:
        """Discover Tesseract executable on the system."""
        # Check PATH first
        path = shutil.which("tesseract")
        if path:
            return path

        # Check standard Windows locations
        for p in _WINDOWS_PATHS:
            if p.exists():
                return str(p)

        return None

    @property
    def tesseract_found(self) -> bool:
        """Whether Tesseract executable was found."""
        return self._tesseract_path is not None

    @property
    def tesseract_path(self) -> Optional[str]:
        """Path to Tesseract executable."""
        return self._tesseract_path

    def get_version(self) -> str:
        """Get Tesseract version string."""
        if not self.tesseract_found:
            return "NOT FOUND"
        try:
            output = subprocess.check_output(
                [self._tesseract_path, "--version"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            return output.splitlines()[0].strip()
        except Exception as e:
            return f"ERROR: {e}"

    def get_available_languages(self) -> List[str]:
        """List available OCR languages."""
        if not self.tesseract_found:
            return []
        try:
            output = subprocess.check_output(
                [self._tesseract_path, "--list-langs"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            lines = output.strip().splitlines()
            # First line is header, rest are language codes
            return [ln.strip() for ln in lines[1:] if ln.strip()]
        except Exception:
            return []

    def is_language_available(self, lang: str) -> bool:
        """Check if a specific language is installed.

        Handles composite language strings like 'tam+eng'.
        """
        available = self.get_available_languages()
        for single_lang in lang.split("+"):
            if single_lang.strip() not in available:
                return False
        return True

    def _build_custom_config(self) -> str:
        """Build Tesseract custom config string."""
        return f"--psm {self.config.psm} --oem {self.config.oem}"

    def _clean_text(self, raw_text: str) -> str:
        """Clean OCR output text.

        Strips leading/trailing whitespace, collapses multiple blank lines,
        normalizes whitespace within lines.
        """
        if not raw_text:
            return ""
        lines = raw_text.splitlines()
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not prev_blank:
                    cleaned_lines.append("")
                prev_blank = True
            else:
                # Collapse internal multiple spaces to single space
                cleaned_lines.append(" ".join(stripped.split()))
                prev_blank = False
        result = "\n".join(cleaned_lines).strip()
        return result

    def run_ocr(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a preprocessed image array.

        Args:
            image: Image as numpy array (grayscale or BGR).

        Returns:
            OCRResult with extracted text and metadata.
        """
        result = OCRResult(
            language_requested=self.config.language,
            psm=self.config.psm,
            oem=self.config.oem,
            tesseract_version=self.get_version(),
        )

        if not self.tesseract_found:
            result.status = "error"
            result.error_message = "Tesseract executable not found"
            return result

        # Check language availability
        if not self.is_language_available(self.config.language):
            result.status = "error"
            result.error_message = (
                f"Language '{self.config.language}' not available. "
                f"Available: {self.get_available_languages()}"
            )
            return result

        # Convert numpy array to PIL Image for pytesseract
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image, mode="L")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        custom_config = self._build_custom_config()
        start_time = time.time()

        try:
            # Extract text
            raw_text = pytesseract.image_to_string(
                pil_img,
                lang=self.config.language,
                config=custom_config,
                timeout=self.config.timeout,
            )

            result.text_raw = raw_text
            result.text_clean = self._clean_text(raw_text)
            result.processing_time_seconds = round(time.time() - start_time, 4)

            if not result.text_clean:
                result.status = "empty"
            else:
                result.status = "success"

            # Try to get word-level confidence
            try:
                data = pytesseract.image_to_data(
                    pil_img,
                    lang=self.config.language,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT,
                    timeout=self.config.timeout,
                )
                confidences = []
                word_confs = []
                for i, conf in enumerate(data["conf"]):
                    conf_val = int(conf)
                    word = data["text"][i].strip()
                    if conf_val >= 0 and word:
                        confidences.append(conf_val)
                        word_confs.append((word, conf_val))

                if confidences:
                    result.confidence_mean = round(
                        sum(confidences) / len(confidences), 2
                    )
                    result.confidence_per_word = word_confs
            except Exception:
                pass  # confidence data is optional

        except RuntimeError as e:
            if "timeout" in str(e).lower():
                result.status = "timeout"
                result.error_message = f"OCR timed out after {self.config.timeout}s"
            else:
                result.status = "error"
                result.error_message = str(e)
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.processing_time_seconds = round(time.time() - start_time, 4)

        return result

    def run_ocr_on_file(self, image_path: str) -> OCRResult:
        """Run OCR directly on an image file.

        Args:
            image_path: Path to image file.

        Returns:
            OCRResult with extracted text and metadata.
        """
        path = Path(image_path)
        if not path.exists():
            result = OCRResult(
                language_requested=self.config.language,
                status="error",
                error_message=f"File not found: {image_path}",
            )
            return result

        # Load via OpenCV to get numpy array
        raw = path.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            result = OCRResult(
                language_requested=self.config.language,
                status="error",
                error_message=f"Cannot decode image: {image_path}",
            )
            return result

        return self.run_ocr(img)


# Need cv2 import at module level for run_ocr_on_file
import cv2
