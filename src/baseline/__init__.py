"""Baseline package for FakeShield Phase 11 comparative evaluation.

Exposes conventional TF-IDF + Logistic Regression text classification baseline
for benchmarking against FakeShield evidence-grounded verification pipeline.
"""

from src.baseline.tfidf_baseline import (
    TfidfBaseline,
    BaselineMetrics,
    compute_classification_metrics,
)

__all__ = [
    "TfidfBaseline",
    "BaselineMetrics",
    "compute_classification_metrics",
]
