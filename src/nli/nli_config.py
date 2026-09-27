"""nli_config.py - Configuration dataclass for FakeShield NLI component."""

from dataclasses import dataclass, field
from typing import List, Dict

DEFAULT_NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"

# Canonical NLI labels used across FakeShield
LABEL_CONTRADICTION = "CONTRADICTION"
LABEL_ENTAILMENT = "ENTAILMENT"
LABEL_NEUTRAL = "NEUTRAL"

CANONICAL_NLI_LABELS = [LABEL_CONTRADICTION, LABEL_ENTAILMENT, LABEL_NEUTRAL]


@dataclass
class NLIConfig:
    """Configuration for NLI model loading, inference, and evidence preprocessing."""

    model_id: str = DEFAULT_NLI_MODEL_ID
    device: int = -1  # -1 for CPU, 0+ for GPU if available
    max_length: int = 512
    batch_size: int = 16
    labels: List[str] = field(default_factory=lambda: list(CANONICAL_NLI_LABELS))
    clean_html: bool = True
    normalize_whitespace: bool = True
    max_evidence_chars: int = 1500  # Cap overly verbose evidence snippets
