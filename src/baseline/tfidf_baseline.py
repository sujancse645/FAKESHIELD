"""tfidf_baseline.py - Conventional TF-IDF + Logistic Regression Baseline.

This baseline serves as a standard text-classification comparator against
FakeShield's evidence-grounded claim verification pipeline.

Design Constraints:
- Substantially simpler than FakeShield.
- Zero OCR, zero external evidence, zero Google Fact Check API, zero NLI,
  zero source reliability tiering, zero translation, zero temporal reasoning.
- Strictly fits on training data; validation data used only for tuning/selection;
  test split evaluated once for final reporting.
"""

import os
import re
import pickle
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class BaselineMetrics:
    """Container for baseline classification metrics."""
    dataset_name: str
    split_name: str
    num_samples: int
    num_classes: int
    classes: List[str]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class_precision": {k: round(v, 4) for k, v in self.per_class_precision.items()},
            "per_class_recall": {k: round(v, 4) for k, v in self.per_class_recall.items()},
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "confusion_matrix": self.confusion_matrix,
        }


def safe_normalize_text(text: str) -> str:
    """Safe text normalization that preserves semantic numbers and negations."""
    if text is None:
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def compute_classification_metrics(
    y_true: Union[List[Any], np.ndarray],
    y_pred: Union[List[Any], np.ndarray],
    dataset_name: str = "dataset",
    split_name: str = "test",
    class_names: Optional[List[str]] = None,
) -> BaselineMetrics:
    """Compute comprehensive classification metrics from ground truth and predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if class_names is None:
        unique_labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))
        class_names = [str(x) for x in unique_labels]
    else:
        unique_labels = class_names

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)

    p_dict = {str(c): float(p) for c, p in zip(unique_labels, prec_per_class)}
    r_dict = {str(c): float(r) for c, r in zip(unique_labels, rec_per_class)}
    f_dict = {str(c): float(f) for c, f in zip(unique_labels, f1_per_class)}

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()

    return BaselineMetrics(
        dataset_name=dataset_name,
        split_name=split_name,
        num_samples=len(y_true),
        num_classes=len(unique_labels),
        classes=[str(c) for c in unique_labels],
        accuracy=acc,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        weighted_f1=weighted_f,
        per_class_precision=p_dict,
        per_class_recall=r_dict,
        per_class_f1=f_dict,
        confusion_matrix=cm,
    )


class TfidfBaseline:
    """Conventional TF-IDF + Logistic Regression Baseline Classifier."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        multi_class: str = "auto",
        solver: str = "lbfgs",
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.multi_class = multi_class
        self.solver = solver

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            preprocessor=safe_normalize_text,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[Any]) -> "TfidfBaseline":
        """Fit vectorizer and classifier strictly on training split."""
        if len(texts) == 0:
            raise ValueError("Cannot fit baseline on empty text collection.")
        if len(texts) != len(labels):
            raise ValueError(f"Mismatch between text count ({len(texts)}) and labels ({len(labels)}).")

        X_tfidf = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X_tfidf, labels)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for input texts."""
        if not self.is_fitted:
            raise RuntimeError("TfidfBaseline must be fitted before calling predict().")
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict(X_tfidf)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for input texts."""
        if not self.is_fitted:
            raise RuntimeError("TfidfBaseline must be fitted before calling predict_proba().")
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_tfidf)

    def evaluate(
        self,
        texts: List[str],
        labels: List[Any],
        dataset_name: str = "dataset",
        split_name: str = "test",
    ) -> BaselineMetrics:
        """Evaluate fitted model on a specific split."""
        preds = self.predict(texts)
        return compute_classification_metrics(
            y_true=labels,
            y_pred=preds,
            dataset_name=dataset_name,
            split_name=split_name,
            class_names=[str(c) for c in self.classes_],
        )

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return baseline configuration for reproducibility reporting."""
        return {
            "model_type": "TF-IDF + Logistic Regression",
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "solver": self.solver,
            "is_fitted": self.is_fitted,
            "vocab_size": len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
        }

    def save(self, filepath: str) -> None:
        """Save fitted baseline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "hyperparams": self.get_hyperparameters(),
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "is_fitted": self.is_fitted,
                "classes_": self.classes_,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "TfidfBaseline":
        """Load fitted baseline from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.is_fitted = data["is_fitted"]
        instance.classes_ = data["classes_"]
        return instance
