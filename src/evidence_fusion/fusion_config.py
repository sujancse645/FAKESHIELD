"""fusion_config.py - Configuration dataclass for FakeShield Evidence Fusion."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
import yaml


@dataclass
class FusionConfig:
    """Configurable weights, thresholds, and temporal rules for evidence fusion."""

    source_tier_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.40}
    )
    min_evidence_strength: float = 0.40
    support_margin: float = 0.30
    contradiction_margin: float = 0.30
    conflict_threshold: float = 0.35
    nli_score_threshold: float = 0.50
    max_temporal_gap_years: float = 1.5  # 18 months
    currentness_indicators: List[str] = field(
        default_factory=lambda: [
            "today",
            "now",
            "currently",
            "this year",
            "this week",
            "this month",
            "just announced",
            "breaking",
            "recent",
            "recently",
            "latest",
        ]
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FusionConfig":
        """Loads configuration from YAML file, with fallback to default heuristic values."""
        if not os.path.exists(yaml_path):
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        tier_weights = data.get("source_tier_weights", {})
        # Convert keys to integer
        int_tier_weights = {int(k): float(v) for k, v in tier_weights.items()} if tier_weights else {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.40}

        thresh = data.get("thresholds", {})
        ooc = data.get("out_of_context", {})

        return cls(
            source_tier_weights=int_tier_weights,
            min_evidence_strength=float(thresh.get("min_evidence_strength", 0.40)),
            support_margin=float(thresh.get("support_margin", 0.30)),
            contradiction_margin=float(thresh.get("contradiction_margin", 0.30)),
            conflict_threshold=float(thresh.get("conflict_threshold", 0.35)),
            nli_score_threshold=float(thresh.get("nli_score_threshold", 0.50)),
            max_temporal_gap_years=float(ooc.get("max_temporal_gap_years", 2.0)),
            currentness_indicators=list(ooc.get("currentness_indicators", [
                "today", "now", "currently", "this year", "this week", "this month", "just announced", "breaking", "recent", "recently", "latest"
            ])),
        )
