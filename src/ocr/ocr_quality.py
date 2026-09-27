"""OCR quality metrics and error rate calculations.

Provides text-level quality indicators for OCR output and
ground-truth comparison via CER (Character Error Rate) and
WER (Word Error Rate).

IMPORTANT: OCR confidence measures how well the engine recognized
characters. It does NOT measure factual correctness, truth, or
fake-news likelihood. These are entirely different concepts.
"""

import time
from typing import Optional

import jiwer


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Uses dynamic programming (Wagner-Fischer algorithm).
    This is a transparent, auditable implementation for CER calculation.
    """
    len1, len2 = len(s1), len(s2)

    # Create distance matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )

    return dp[len1][len2]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER).

    CER = edit_distance(reference, hypothesis) / len(reference)

    Args:
        reference: Ground-truth text.
        hypothesis: OCR-produced text.

    Returns:
        CER as a float. 0.0 = perfect match.
        Returns 0.0 if both strings are empty.
        Returns float('inf') if reference is empty but hypothesis is not.
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return float("inf")

    distance = _levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using jiwer.

    WER = (substitutions + deletions + insertions) / words_in_reference

    Args:
        reference: Ground-truth text.
        hypothesis: OCR-produced text.

    Returns:
        WER as a float. 0.0 = perfect match.
        Returns 0.0 if both strings are empty.
        Returns float('inf') if reference is empty but hypothesis is not.
    """
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return float("inf")
    if not hypothesis.strip():
        # All words are deletions
        return 1.0

    return jiwer.wer(reference, hypothesis)


class OCRQualityAnalyzer:
    """Analyze OCR output quality metrics.

    These metrics measure OCR text characteristics, NOT factual accuracy.
    """

    def analyze(self, ocr_text: str) -> dict:
        """Compute quality metrics for OCR output text.

        Args:
            ocr_text: The text produced by OCR.

        Returns:
            Dictionary of quality metrics.
        """
        if not ocr_text:
            return {
                "char_count": 0,
                "word_count": 0,
                "alphabetic_ratio": 0.0,
                "digit_ratio": 0.0,
                "whitespace_ratio": 0.0,
                "suspicious_symbol_ratio": 0.0,
                "avg_word_length": 0.0,
                "is_empty": True,
                "line_count": 0,
            }

        char_count = len(ocr_text)
        words = ocr_text.split()
        word_count = len(words)
        line_count = len(ocr_text.splitlines())

        alpha_count = sum(1 for c in ocr_text if c.isalpha())
        digit_count = sum(1 for c in ocr_text if c.isdigit())
        whitespace_count = sum(1 for c in ocr_text if c.isspace())

        # Suspicious symbols: characters that are neither alpha, digit,
        # whitespace, nor common punctuation
        common_punct = set(".,;:!?'\"-()[]{}/@#$%&*+=<>_~`^|\\")
        suspicious_count = sum(
            1 for c in ocr_text
            if not c.isalnum() and not c.isspace() and c not in common_punct
        )

        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )

        return {
            "char_count": char_count,
            "word_count": word_count,
            "alphabetic_ratio": round(alpha_count / char_count, 4) if char_count else 0.0,
            "digit_ratio": round(digit_count / char_count, 4) if char_count else 0.0,
            "whitespace_ratio": round(whitespace_count / char_count, 4) if char_count else 0.0,
            "suspicious_symbol_ratio": round(suspicious_count / char_count, 4) if char_count else 0.0,
            "avg_word_length": round(avg_word_length, 2),
            "is_empty": False,
            "line_count": line_count,
        }

    def evaluate_against_reference(
        self, reference: str, hypothesis: str
    ) -> dict:
        """Compare OCR output against ground-truth reference text.

        Args:
            reference: Known correct text.
            hypothesis: OCR-produced text.

        Returns:
            Dictionary with CER, WER, and text statistics.
        """
        cer = compute_cer(reference, hypothesis)
        wer = compute_wer(reference, hypothesis)
        quality = self.analyze(hypothesis)

        return {
            "cer": round(cer, 4) if cer != float("inf") else "inf",
            "wer": round(wer, 4) if wer != float("inf") else "inf",
            "reference_char_count": len(reference),
            "reference_word_count": len(reference.split()),
            "hypothesis_char_count": quality["char_count"],
            "hypothesis_word_count": quality["word_count"],
            **quality,
        }
