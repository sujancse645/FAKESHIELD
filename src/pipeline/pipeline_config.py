"""pipeline_config.py - Configuration for the FakeShield End-to-End Pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml

from src.ocr.tesseract_engine import OCRConfig
from src.ocr.image_preprocessing import PreprocessingConfig
from src.translation.translation_config import TranslationConfig
from src.nli.nli_config import NLIConfig
from src.evidence_fusion.fusion_config import FusionConfig


@dataclass
class PipelineConfig:
    """Master configuration for FakeShield multimodal verification pipeline."""

    max_claims: int = 5
    max_evidence_per_claim: int = 5
    supported_input_types: List[str] = field(
        default_factory=lambda: ["text", "image", "url"]
    )
    url_request_timeout: float = 5.0
    url_max_chars: int = 15000
    min_claim_words: int = 3
    preserve_full_provenance: bool = True

    # Sub-component configs
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    preprocessing_config: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    translation_config: TranslationConfig = field(default_factory=TranslationConfig)
    nli_config: NLIConfig = field(default_factory=NLIConfig)
    fusion_config: FusionConfig = field(default_factory=FusionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file if available."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        cfg = cls()
        if "max_claims" in data:
            cfg.max_claims = int(data["max_claims"])
        if "max_evidence_per_claim" in data:
            cfg.max_evidence_per_claim = int(data["max_evidence_per_claim"])
        if "url_request_timeout" in data:
            cfg.url_request_timeout = float(data["url_request_timeout"])
        return cfg
