"""Generate synthetic test images with known reference text for OCR evaluation.

Creates programmatic test images using Pillow — no external downloads needed.
Each image has a corresponding reference text stored in references.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Reference texts for each sample
REFERENCES = {
    "english_clear": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
    "english_noisy": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
    "english_skewed": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
    "english_low_contrast": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
    "blank": "",
    "tiny": "",
}


def get_font(size: int = 32):
    """Get a font, falling back to default if system fonts unavailable."""
    # Try common Windows fonts
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    # Fallback to default
    return ImageFont.load_default()


def create_text_image(
    text: str,
    width: int = 800,
    height: int = 200,
    font_size: int = 32,
    bg_color: int = 255,
    text_color: int = 0,
) -> Image.Image:
    """Create an image with rendered text."""
    img = Image.new("L", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    # Draw text with margin
    margin = 30
    y_offset = margin
    for line in text.split("\n"):
        draw.text((margin, y_offset), line, fill=text_color, font=font)
        y_offset += font_size + 10

    return img


def create_english_clear(output_dir: Path) -> str:
    """Create clean English text image."""
    text = REFERENCES["english_clear"]
    img = create_text_image(text, width=800, height=200)
    path = output_dir / "english_clear.png"
    img.save(str(path))
    return str(path)


def create_english_noisy(output_dir: Path) -> str:
    """Create English text image with Gaussian noise."""
    text = REFERENCES["english_noisy"]
    img = create_text_image(text, width=800, height=200)

    # Add noise
    arr = np.array(img, dtype=np.float32)
    noise = np.random.RandomState(42).normal(0, 25, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img_noisy = Image.fromarray(arr, mode="L")

    path = output_dir / "english_noisy.png"
    img_noisy.save(str(path))
    return str(path)


def create_english_skewed(output_dir: Path) -> str:
    """Create English text image rotated ~3 degrees."""
    text = REFERENCES["english_skewed"]
    img = create_text_image(text, width=900, height=250)

    # Rotate 3 degrees
    img_rotated = img.rotate(-3, expand=True, fillcolor=255)

    path = output_dir / "english_skewed.png"
    img_rotated.save(str(path))
    return str(path)


def create_english_low_contrast(output_dir: Path) -> str:
    """Create English text image with reduced contrast."""
    text = REFERENCES["english_low_contrast"]
    # Light gray text on lighter gray background
    img = create_text_image(text, width=800, height=200, bg_color=220, text_color=140)

    path = output_dir / "english_low_contrast.png"
    img.save(str(path))
    return str(path)


def create_blank(output_dir: Path) -> str:
    """Create blank white image (edge case)."""
    img = Image.new("L", (400, 200), color=255)
    path = output_dir / "blank.png"
    img.save(str(path))
    return str(path)


def create_tiny(output_dir: Path) -> str:
    """Create very small image (edge case)."""
    img = Image.new("L", (10, 10), color=255)
    path = output_dir / "tiny.png"
    img.save(str(path))
    return str(path)


def generate_all(output_dir: str = "data/ocr_samples") -> dict:
    """Generate all synthetic test images and save references.json.

    Returns:
        Dictionary mapping image_id -> file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic OCR test images...")

    paths = {}
    paths["english_clear"] = create_english_clear(out)
    print(f"  [OK] english_clear.png")

    paths["english_noisy"] = create_english_noisy(out)
    print(f"  [OK] english_noisy.png")

    paths["english_skewed"] = create_english_skewed(out)
    print(f"  [OK] english_skewed.png")

    paths["english_low_contrast"] = create_english_low_contrast(out)
    print(f"  [OK] english_low_contrast.png")

    paths["blank"] = create_blank(out)
    print(f"  [OK] blank.png")

    paths["tiny"] = create_tiny(out)
    print(f"  [OK] tiny.png")

    # Save references
    refs_path = out / "references.json"
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump(REFERENCES, f, indent=2, ensure_ascii=False)
    print(f"  [OK] references.json")

    print(f"\nGenerated {len(paths)} test images in {out}")
    return paths


if __name__ == "__main__":
    generate_all()
