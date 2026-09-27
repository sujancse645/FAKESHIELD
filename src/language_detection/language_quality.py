"""language_quality.py - Evaluation metrics for FakeShield Language Detection.

Computes accuracy, macro precision, macro recall, macro F1, and
confusion matrices for language identification experiments.
"""

from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple


class LanguageQualityAnalyzer:
    """Computes standard classification evaluation metrics for language detection."""

    @staticmethod
    def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predictions against ground truth.

        Args:
            records: List of dicts, each with 'expected_language' and 'detected_language',
                     plus 'status'.

        Returns:
            Dict containing accuracy, macro metrics, per-language breakdown,
            and confusion matrix.
        """
        # Filter samples that have a valid ground truth label
        eval_records = [
            r for r in records
            if r.get("expected_language") and r["expected_language"] != "N/A"
        ]

        total_evaluated = len(eval_records)
        if total_evaluated == 0:
            return {"error": "No valid ground truth records found for evaluation."}

        classes = sorted(list(set(r["expected_language"] for r in eval_records)))

        # Build confusion counts
        # confusion[expected][detected]
        confusion: Dict[str, Dict[str, int]] = {c: defaultdict(int) for c in classes}
        correct = 0
        insufficient_count = 0

        for r in eval_records:
            exp = r["expected_language"]
            det = r.get("detected_language") or "unclassified"
            status = r.get("status", "")

            if status == "insufficient_text":
                insufficient_count += 1

            confusion[exp][det] += 1
            if exp == det:
                correct += 1

        accuracy = correct / total_evaluated

        # Compute per-class precision, recall, F1
        per_class: Dict[str, Dict[str, float]] = {}
        precisions = []
        recalls = []
        f1s = []

        for c in classes:
            # TP: expected == c and detected == c
            tp = confusion[c][c]
            # FN: expected == c and detected != c
            fn = sum(confusion[c][d] for d in confusion[c] if d != c)
            # FP: expected != c and detected == c
            fp = sum(confusion[other][c] for other in classes if other != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            per_class[c] = {
                "support": tp + fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }

        macro_prec = sum(precisions) / len(classes) if classes else 0.0
        macro_rec = sum(recalls) / len(classes) if classes else 0.0
        macro_f1 = sum(f1s) / len(classes) if classes else 0.0

        # Build clean confusion matrix representation
        detected_labels = sorted(list(set(
            det for c in classes for det in confusion[c].keys()
        )))

        confusion_matrix_data = []
        for exp in classes:
            row = {"expected_language": exp}
            for det in detected_labels:
                row[det] = confusion[exp][det]
            confusion_matrix_data.append(row)

        return {
            "total_evaluated": total_evaluated,
            "correct_detections": correct,
            "incorrect_detections": total_evaluated - correct,
            "insufficient_text_cases": insufficient_count,
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "per_language": per_class,
            "confusion_matrix": confusion_matrix_data,
            "classes": classes,
            "detected_labels": detected_labels,
        }
