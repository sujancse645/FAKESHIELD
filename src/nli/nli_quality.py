"""nli_quality.py - Quality evaluation and error analysis for NLI predictions."""

from typing import List, Dict, Any, Tuple
import pandas as pd
from src.nli.nli_config import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    CANONICAL_NLI_LABELS,
)


class NLIQualityAnalyzer:
    """Calculates accuracy, per-class metrics, confusion matrix, and error analysis."""

    def __init__(self, labels: List[str] = None):
        self.labels = labels or CANONICAL_NLI_LABELS

    def compute_metrics(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, Any]:
        """Calculates accuracy, per-class and macro precision, recall, and F1."""
        assert len(y_true) == len(y_pred), "y_true and y_pred must have equal length"
        n = len(y_true)
        if n == 0:
            return {}

        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = correct / n

        per_class = {}
        macro_p = 0.0
        macro_r = 0.0
        macro_f1 = 0.0

        for label in self.labels:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": sum(1 for yt in y_true if yt == label),
            }

            macro_p += precision
            macro_r += recall
            macro_f1 += f1

        num_labels = len(self.labels)
        macro_metrics = {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_p / num_labels, 4),
            "macro_recall": round(macro_r / num_labels, 4),
            "macro_f1": round(macro_f1 / num_labels, 4),
            "total_samples": n,
        }

        return {
            "summary": macro_metrics,
            "per_class": per_class,
        }

    def compute_confusion_matrix(
        self, y_true: List[str], y_pred: List[str]
    ) -> pd.DataFrame:
        """Constructs a confusion matrix DataFrame indexed by true label and columned by pred label."""
        matrix = {
            pred_lbl: {true_lbl: 0 for true_lbl in self.labels}
            for pred_lbl in self.labels
        }
        for yt, yp in zip(y_true, y_pred):
            if yt in matrix and yp in matrix[yt]:
                matrix[yp][yt] += 1

        df_cm = pd.DataFrame(matrix, index=self.labels)
        df_cm.index.name = "True_Label"
        return df_cm

    def analyze_errors(
        self, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identifies misclassifications and categorizes potential linguistic error causes."""
        errors = []
        for s in samples:
            y_true = s.get("expected_label")
            y_pred = s.get("predicted_label")
            score = s.get("predicted_score", 0.0)

            if y_true != y_pred:
                claim = s.get("claim", "").lower()
                evidence = s.get("evidence", "").lower()

                error_cat = "ambiguous evidence"
                if any(w in claim for w in ["not", "no", "never", "refuted", "denied", "false"]):
                    error_cat = "negation failure"
                elif any(char.isdigit() for char in claim):
                    error_cat = "numerical mismatch"
                elif any(w in claim for w in ["today", "yesterday", "tomorrow", "year", "month"]):
                    error_cat = "temporal mismatch"
                elif len(evidence.split()) > 35:
                    error_cat = "long/complex sentence"
                elif y_true == LABEL_CONTRADICTION and y_pred == LABEL_NEUTRAL:
                    error_cat = "insufficient context"
                elif y_true == LABEL_NEUTRAL and y_pred == LABEL_ENTAILMENT:
                    error_cat = "lexical mismatch"

                errors.append({
                    "sample_id": s.get("sample_id"),
                    "claim": s.get("claim"),
                    "evidence": s.get("evidence"),
                    "expected_label": y_true,
                    "predicted_label": y_pred,
                    "predicted_score": round(score, 4),
                    "error_category": error_cat,
                    "notes": s.get("notes", ""),
                })
        return errors
