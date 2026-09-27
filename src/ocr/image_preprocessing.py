"""Image preprocessing pipeline for OCR.

Provides configurable preprocessing operations using Pillow and OpenCV
to improve OCR quality. Each operation is individually toggleable.
The original image is NEVER modified or overwritten.

Operations and their purposes:
- Grayscale conversion: Reduces color complexity for OCR engines
- Resize/upscale: Ensures sufficient resolution for character recognition
- Noise reduction: Removes scanner/camera artifacts that confuse OCR
- Contrast enhancement (CLAHE): Improves text-background separation
- Thresholding/binarization: Creates clean black-text-on-white for OCR
- Deskew: Corrects rotation from scanning/photography
- Border addition: Provides margin space for OCR engines
"""

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing operations.

    Each boolean flag enables/disables the corresponding operation.
    Parameters control operation-specific behavior.
    """

    grayscale: bool = True
    resize: bool = False
    resize_target_height: int = 2000  # pixels; width scaled proportionally
    noise_reduction: bool = True
    noise_kernel_size: int = 3  # must be odd
    contrast_enhancement: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    threshold: bool = True
    threshold_method: str = "otsu"  # "otsu" or "adaptive"
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    deskew: bool = False
    border: bool = True
    border_size: int = 10  # pixels
    border_color: int = 255  # white

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return {
            "grayscale": self.grayscale,
            "resize": self.resize,
            "resize_target_height": self.resize_target_height,
            "noise_reduction": self.noise_reduction,
            "noise_kernel_size": self.noise_kernel_size,
            "contrast_enhancement": self.contrast_enhancement,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid": list(self.clahe_tile_grid),
            "threshold": self.threshold,
            "threshold_method": self.threshold_method,
            "deskew": self.deskew,
            "border": self.border,
            "border_size": self.border_size,
        }


class ImagePreprocessor:
    """Configurable image preprocessing pipeline for OCR.

    Applies a sequence of image transformations to improve OCR quality.
    The original image is never modified.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as numpy array (BGR format).

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read raw bytes to handle unicode paths on Windows
        raw = path.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        return img

    def get_image_info(self, img: np.ndarray) -> dict:
        """Extract basic image metadata."""
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        return {
            "height": h,
            "width": w,
            "channels": channels,
            "dtype": str(img.dtype),
        }

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        if len(img.shape) == 2:
            return img  # already grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target height, preserving aspect ratio."""
        h, w = img.shape[:2]
        target_h = self.config.resize_target_height
        if h >= target_h:
            return img  # no upscale needed
        scale = target_h / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    def reduce_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply median blur for noise reduction."""
        k = self.config.noise_kernel_size
        if k % 2 == 0:
            k += 1  # kernel must be odd
        return cv2.medianBlur(img, k)

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Ensure input is grayscale
        if len(img.shape) == 3:
            img = self.to_grayscale(img)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )
        return clahe.apply(img)

    def apply_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply binarization via Otsu's or adaptive thresholding."""
        if len(img.shape) == 3:
            img = self.to_grayscale(img)

        if self.config.threshold_method == "otsu":
            _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.config.threshold_method == "adaptive":
            result = cv2.adaptiveThreshold(
                img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_block_size,
                self.config.adaptive_c,
            )
        else:
            raise ValueError(f"Unknown threshold method: {self.config.threshold_method}")
        return result

    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Correct image skew using minAreaRect on contours."""
        if len(img.shape) == 3:
            gray = self.to_grayscale(img)
        else:
            gray = img

        # Invert for contour detection (text = white on black)
        inverted = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inverted > 0))

        if len(coords) < 10:
            return img  # not enough content to deskew

        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct small angles (< 15 degrees)
        if abs(angle) > 15 or abs(angle) < 0.1:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def add_border(self, img: np.ndarray) -> np.ndarray:
        """Add white border/margin around the image."""
        bs = self.config.border_size
        color = self.config.border_color
        return cv2.copyMakeBorder(
            img, bs, bs, bs, bs,
            cv2.BORDER_CONSTANT,
            value=color,
        )

    def validate_output(self, img: np.ndarray) -> bool:
        """Verify output is a valid image array."""
        if img is None:
            return False
        if not isinstance(img, np.ndarray):
            return False
        if img.size == 0:
            return False
        return True

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run the full preprocessing pipeline.

        Args:
            image_path: Path to input image.

        Returns:
            Tuple of (original_image, preprocessed_image, metadata_dict).
        """
        start_time = time.time()

        # Load original
        original = self.load_image(image_path)
        original_info = self.get_image_info(original)

        img = original.copy()
        steps_applied = []

        # 1. Grayscale
        if self.config.grayscale:
            img = self.to_grayscale(img)
            steps_applied.append("grayscale")

        # 2. Resize
        if self.config.resize:
            img = self.resize_image(img)
            steps_applied.append("resize")

        # 3. Noise reduction
        if self.config.noise_reduction:
            img = self.reduce_noise(img)
            steps_applied.append("noise_reduction")

        # 4. Contrast enhancement
        if self.config.contrast_enhancement:
            img = self.enhance_contrast(img)
            steps_applied.append("contrast_enhancement")

        # 5. Thresholding
        if self.config.threshold:
            img = self.apply_threshold(img)
            steps_applied.append("threshold")

        # 6. Deskew
        if self.config.deskew:
            img = self.deskew_image(img)
            steps_applied.append("deskew")

        # 7. Border
        if self.config.border:
            img = self.add_border(img)
            steps_applied.append("border")

        # Validate
        if not self.validate_output(img):
            raise RuntimeError("Preprocessing produced invalid output image")

        elapsed = time.time() - start_time
        processed_info = self.get_image_info(img)

        metadata = {
            "original": original_info,
            "processed": processed_info,
            "steps_applied": steps_applied,
            "config": self.config.to_dict(),
            "preprocessing_time_seconds": round(elapsed, 4),
        }

        return original, img, metadata

    def save_image(self, img: np.ndarray, output_path: str) -> str:
        """Save image to disk.

        Args:
            img: Image array to save.
            output_path: Destination path.

        Returns:
            Absolute path of saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Encode to bytes then write (handles unicode paths on Windows)
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        else:
            encode_param = []

        success, buf = cv2.imencode(ext if ext else ".png", img, encode_param)
        if not success:
            raise RuntimeError(f"Failed to encode image to {ext}")
        path.write_bytes(buf.tobytes())

        return str(path.resolve())
