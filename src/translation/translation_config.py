"""translation_config.py - Centralized configuration for FakeShield Phase 6 translation layer.

Contains model identifiers, language code mappings, token tags, and device settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Mapping from FakeShield 2-letter language codes to IndicTrans2 FLORES-style tags
# Format: code -> (FLORES_tag, Human_Readable_Name)
LANGUAGE_MAPPING: Dict[str, Dict[str, str]] = {
    "en": {"tag": "eng_Latn", "name": "English"},
    "ta": {"tag": "tam_Taml", "name": "Tamil"},
    "hi": {"tag": "hin_Deva", "name": "Hindi"},
    "te": {"tag": "tel_Telu", "name": "Telugu"},
    "ml": {"tag": "mal_Mlym", "name": "Malayalam"},
    "kn": {"tag": "kan_Knda", "name": "Kannada"},
    "bn": {"tag": "ben_Beng", "name": "Bengali"},
    "mr": {"tag": "mar_Deva", "name": "Marathi"},
    "gu": {"tag": "guj_Gujr", "name": "Gujarati"},
    "pa": {"tag": "pan_Guru", "name": "Punjabi"},
    "ur": {"tag": "urd_Arab", "name": "Urdu"},
    "as": {"tag": "asm_Beng", "name": "Assamese"},
    "or": {"tag": "ory_Orya", "name": "Odia"},
}

TARGET_LANGUAGE = "en"
TARGET_TAG = "eng_Latn"


@dataclass
class TranslationConfig:
    """Central configuration for IndicTrans2 translation layer."""
    model_identifier: str = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
    tokenizer_identifier: str = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
    device: str = "cpu"
    max_input_length: int = 512
    max_new_tokens: int = 256
    num_beams: int = 4
    batch_size: int = 1
    target_language: str = TARGET_LANGUAGE
    target_tag: str = TARGET_TAG
    supported_languages: Dict[str, Dict[str, str]] = field(default_factory=lambda: LANGUAGE_MAPPING)

    def is_supported(self, lang_code: str) -> bool:
        """Check if language code is supported by the translation layer."""
        return lang_code.lower() in self.supported_languages

    def get_tag(self, lang_code: str) -> Optional[str]:
        """Get the FLORES language tag for a given language code."""
        info = self.supported_languages.get(lang_code.lower())
        return info["tag"] if info else None

    def get_name(self, lang_code: str) -> str:
        """Get the human-readable language name for a given language code."""
        info = self.supported_languages.get(lang_code.lower())
        return info["name"] if info else f"Unknown ({lang_code})"
