"""translator.py - Production-grade translation layer for FakeShield Phase 6.

Translates Indic languages to an English pivot representation using IndicTrans2.
English inputs skip translation with status='skipped'.
Short, numeric, or punctuation-only inputs are guarded with status='insufficient_text'.
"""

import os
import sys
import re
import time
import unicodedata
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForSeq2SeqLM

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesPunctNormalizer

from src.translation.translation_config import TranslationConfig, LANGUAGE_MAPPING
from src.language_detection.language_detector import LanguageDetector


@dataclass
class TranslationResult:
    """Structured result from the translation layer."""
    raw_text: str
    normalized_text: str
    source_language: Optional[str]
    source_language_name: str
    target_language: str
    english_pivot: str
    status: str
    character_count_in: int
    character_count_out: int
    model_name: str
    is_pivot_english: bool
    execution_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return asdict(self)


class Translator:
    """FakeShield Multilingual Translator using IndicTrans2 for English pivot generation."""

    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._model = None
        self._tokenizer = None
        self._detector = None
        self._punct_normalizer = MosesPunctNormalizer()
        self._normalizers: Dict[str, Any] = {}
        self._transliterator = UnicodeIndicTransliterator()

    @property
    def detector(self) -> LanguageDetector:
        """Lazy-loaded language detector for automatic language identification."""
        if self._detector is None:
            self._detector = LanguageDetector()
        return self._detector

    def _ensure_model_loaded(self):
        """Lazy load model and tokenizer on first use."""
        if self._model is not None and self._tokenizer is not None:
            return

        tokenizer_dir = None
        # Check standard huggingface hub cache location
        from huggingface_hub import snapshot_download
        tokenizer_dir = snapshot_download(
            self.config.tokenizer_identifier,
            allow_patterns=["*tokenization*", "*dict*", "*model.*", "config.json"]
        )

        if tokenizer_dir not in sys.path:
            sys.path.insert(0, tokenizer_dir)

        import tokenization_indictrans
        self._tokenizer = tokenization_indictrans.IndicTransTokenizer.from_pretrained(tokenizer_dir)

        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_identifier,
            trust_remote_code=True
        )
        self._model.to(self.config.device)
        self._model.eval()

    def _get_indic_normalizer(self, iso_lang: str):
        """Get or initialize language-specific normalizer."""
        if iso_lang not in self._normalizers:
            try:
                self._normalizers[iso_lang] = IndicNormalizerFactory().get_normalizer(iso_lang)
            except Exception:
                self._normalizers[iso_lang] = None
        return self._normalizers[iso_lang]

    def _normalize_text(self, text: str) -> str:
        """Apply Unicode NFKC normalization and collapse extra whitespace."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", str(text))
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def _is_insufficient_text(self, text: str) -> bool:
        """Check if text is insufficient for meaningful linguistic translation."""
        if not text or len(text.strip()) == 0:
            return True
        stripped = text.strip()
        # Less than 3 characters
        if len(stripped) < 3:
            return True
        # Check if text has any linguistic/letter characters
        letters = [c for c in stripped if unicodedata.category(c).startswith("L")]
        if len(letters) < 3:
            return True
        # Pure digits or punctuation (using standard regex categories)
        non_punct_digits = [c for c in stripped if not (c.isdigit() or unicodedata.category(c).startswith("P") or c.isspace())]
        if not non_punct_digits:
            return True
        return False

    def preprocess_indic_text(self, text: str, src_lang_code: str) -> str:
        """Preprocess Indic text: punctuation normalization, Indic normalization, and script transliteration."""
        # 1. Punctuation normalization
        cleaned = self._punct_normalizer.normalize(text)

        # 2. Language-specific normalization
        normalizer = self._get_indic_normalizer(src_lang_code)
        if normalizer is not None:
            cleaned = normalizer.normalize(cleaned)

        # 3. Trivial tokenization
        try:
            tokens = indic_tokenize.trivial_tokenize(cleaned, src_lang_code)
            joined = " ".join(tokens)
        except Exception:
            joined = cleaned

        # 4. Transliterate to Devanagari script for IndicTrans2 (except Hindi, Marathi, Urdu)
        # Urdu uses Perso-Arabic script; Hindi/Marathi already use Devanagari
        if src_lang_code not in ("hi", "mr", "ur"):
            try:
                xlated = self._transliterator.transliterate(joined, src_lang_code, "hi")
                joined = xlated.replace(" ् ", "्")
            except Exception:
                pass

        return joined

    def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translate input text into English pivot representation.

        Args:
            text: Input text to translate.
            source_language: Optional 2-letter language code (e.g., 'ta', 'hi', 'en').
                             If None, language is automatically detected.
            sample_id: Optional identifier for logging/tracking.

        Returns:
            Dict containing raw_text, normalized_text, english_pivot, status, etc.
        """
        start_time = time.time()
        raw_text = "" if text is None else str(text)
        normalized_text = self._normalize_text(raw_text)

        # 1. Short text / edge-case guard
        if self._is_insufficient_text(normalized_text):
            elapsed = round(time.time() - start_time, 4)
            return TranslationResult(
                raw_text=raw_text,
                normalized_text=normalized_text,
                source_language=source_language,
                source_language_name="Unknown" if not source_language else self.config.get_name(source_language),
                target_language=self.config.target_language,
                english_pivot="",
                status="insufficient_text",
                character_count_in=len(raw_text),
                character_count_out=0,
                model_name=self.config.model_identifier,
                is_pivot_english=False,
                execution_time_sec=elapsed
            ).to_dict()

        # 2. Resolve source language if not provided
        if not source_language:
            detection = self.detector.detect(normalized_text)
            if detection.status == "success":
                source_language = detection.language_code
            else:
                source_language = "unknown"

        source_lang_code = source_language.lower().strip()

        # 3. English skip logic (English is already the pivot language)
        if source_lang_code in ("en", "english"):
            elapsed = round(time.time() - start_time, 4)
            return TranslationResult(
                raw_text=raw_text,
                normalized_text=normalized_text,
                source_language="en",
                source_language_name="English",
                target_language=self.config.target_language,
                english_pivot=normalized_text,
                status="skipped",
                character_count_in=len(raw_text),
                character_count_out=len(normalized_text),
                model_name=self.config.model_identifier,
                is_pivot_english=True,
                execution_time_sec=elapsed
            ).to_dict()

        # 4. Check language support
        if not self.config.is_supported(source_lang_code):
            elapsed = round(time.time() - start_time, 4)
            return TranslationResult(
                raw_text=raw_text,
                normalized_text=normalized_text,
                source_language=source_lang_code,
                source_language_name=f"Unsupported ({source_lang_code})",
                target_language=self.config.target_language,
                english_pivot="",
                status="unsupported_language",
                character_count_in=len(raw_text),
                character_count_out=0,
                model_name=self.config.model_identifier,
                is_pivot_english=False,
                execution_time_sec=elapsed
            ).to_dict()

        # 5. Perform IndicTrans2 translation
        try:
            self._ensure_model_loaded()

            src_tag = self.config.get_tag(source_lang_code)
            tgt_tag = self.config.target_tag

            # Preprocess text (normalization, tokenization, transliteration)
            preprocessed_text = self.preprocess_indic_text(normalized_text, source_lang_code)
            tagged_input = f"{src_tag} {tgt_tag} {preprocessed_text}"

            inputs = self._tokenizer(
                tagged_input,
                truncation=True,
                max_length=self.config.max_input_length,
                return_tensors="pt"
            ).to(self.config.device)

            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    use_cache=False,
                    early_stopping=True,
                )

            decoded = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            translated_text = decoded[0].strip() if decoded else ""
            elapsed = round(time.time() - start_time, 4)

            return TranslationResult(
                raw_text=raw_text,
                normalized_text=normalized_text,
                source_language=source_lang_code,
                source_language_name=self.config.get_name(source_lang_code),
                target_language=self.config.target_language,
                english_pivot=translated_text,
                status="success",
                character_count_in=len(raw_text),
                character_count_out=len(translated_text),
                model_name=self.config.model_identifier,
                is_pivot_english=True,
                execution_time_sec=elapsed
            ).to_dict()
        except Exception as e:
            elapsed = round(time.time() - start_time, 4)
            return TranslationResult(
                raw_text=raw_text,
                normalized_text=normalized_text,
                source_language=source_lang_code,
                source_language_name=self.config.get_name(source_lang_code),
                target_language=self.config.target_language,
                english_pivot="",
                status="failed",
                character_count_in=len(raw_text),
                character_count_out=0,
                model_name=self.config.model_identifier,
                is_pivot_english=False,
                execution_time_sec=elapsed
            ).to_dict()


def translate_to_english(
    text: str,
    source_language: Optional[str] = None,
    translator: Optional[Translator] = None
) -> Dict[str, Any]:
    """Convenience functional wrapper for English pivot translation."""
    t = translator or Translator()
    return t.translate(text, source_language=source_language)
