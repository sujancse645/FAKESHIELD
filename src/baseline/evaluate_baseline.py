"""evaluate_baseline.py - Standalone evaluation for IFND and LIAR baselines."""

import os
import sys
import json
import pandas as pd
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline.tfidf_baseline import TfidfBaseline, BaselineMetrics


def run_baseline_evaluation(
    splits_dir: str = os.path.join(PROJECT_ROOT, "data", "splits"),
    output_dir: str = os.path.join(PROJECT_ROOT, "data", "evaluation", "benchmark"),
) -> Dict[str, Any]:
    """Train and evaluate conventional baselines on IFND and LIAR datasets."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # 1. IFND Binary Classification
    print("[Baseline Track 1] Evaluating IFND Binary Classification...")
    ifnd_train_path = os.path.join(splits_dir, "ifnd_train.csv")
    ifnd_val_path = os.path.join(splits_dir, "ifnd_validation.csv")
    ifnd_test_path = os.path.join(splits_dir, "ifnd_test.csv")

    df_ifnd_train = pd.read_csv(ifnd_train_path)
    df_ifnd_val = pd.read_csv(ifnd_val_path)
    df_ifnd_test = pd.read_csv(ifnd_test_path)

    X_train_ifnd = df_ifnd_train["normalized_statement"].fillna("").astype(str).tolist()
    y_train_ifnd = df_ifnd_train["label_binary"].astype(int).tolist()

    X_val_ifnd = df_ifnd_val["normalized_statement"].fillna("").astype(str).tolist()
    y_val_ifnd = df_ifnd_val["label_binary"].astype(int).tolist()

    X_test_ifnd = df_ifnd_test["normalized_statement"].fillna("").astype(str).tolist()
    y_test_ifnd = df_ifnd_test["label_binary"].astype(int).tolist()

    baseline_ifnd = TfidfBaseline(random_state=42, max_iter=1000)
    baseline_ifnd.fit(X_train_ifnd, y_train_ifnd)

    val_metrics_ifnd = baseline_ifnd.evaluate(X_val_ifnd, y_val_ifnd, "IFND", "validation")
    test_metrics_ifnd = baseline_ifnd.evaluate(X_test_ifnd, y_test_ifnd, "IFND", "test")

    print(f"  IFND Test Accuracy: {test_metrics_ifnd.accuracy:.4f} | Macro F1: {test_metrics_ifnd.macro_f1:.4f}")

    # 2. LIAR 6-Class Multiclass Classification
    print("[Baseline Track 2] Evaluating LIAR 6-Class Multiclass Classification...")
    liar_train_path = os.path.join(splits_dir, "liar_train.csv")
    liar_val_path = os.path.join(splits_dir, "liar_validation.csv")
    liar_test_path = os.path.join(splits_dir, "liar_test.csv")

    df_liar_train = pd.read_csv(liar_train_path)
    df_liar_val = pd.read_csv(liar_val_path)
    df_liar_test = pd.read_csv(liar_test_path)

    X_train_liar = df_liar_train["statement_normalized"].fillna("").astype(str).tolist()
    y_train_liar = df_liar_train["label"].astype(str).tolist()

    X_val_liar = df_liar_val["statement_normalized"].fillna("").astype(str).tolist()
    y_val_liar = df_liar_val["label"].astype(str).tolist()

    X_test_liar = df_liar_test["statement_normalized"].fillna("").astype(str).tolist()
    y_test_liar = df_liar_test["label"].astype(str).tolist()

    baseline_liar = TfidfBaseline(random_state=42, max_iter=1000)
    baseline_liar.fit(X_train_liar, y_train_liar)

    val_metrics_liar = baseline_liar.evaluate(X_val_liar, y_val_liar, "LIAR_6class", "validation")
    test_metrics_liar = baseline_liar.evaluate(X_test_liar, y_test_liar, "LIAR_6class", "test")

    print(f"  LIAR 6-Class Test Accuracy: {test_metrics_liar.accuracy:.4f} | Macro F1: {test_metrics_liar.macro_f1:.4f}")

    # 3. LIAR Binary Mapping Classification (Documented)
    # Mapping: pants-fire, false, barely-true -> 0 (False); half-true, mostly-true, true -> 1 (True)
    print("[Baseline Track 3] Evaluating LIAR Binary Classification (Documented Mapping)...")
    binary_map = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 1,
        "mostly-true": 1,
        "true": 1,
    }
    y_train_liar_bin = [binary_map[l] for l in y_train_liar]
    y_val_liar_bin = [binary_map[l] for l in y_val_liar]
    y_test_liar_bin = [binary_map[l] for l in y_test_liar]

    baseline_liar_bin = TfidfBaseline(random_state=42, max_iter=1000)
    baseline_liar_bin.fit(X_train_liar, y_train_liar_bin)
    test_metrics_liar_bin = baseline_liar_bin.evaluate(X_test_liar, y_test_liar_bin, "LIAR_binary", "test")
    print(f"  LIAR Binary Test Accuracy: {test_metrics_liar_bin.accuracy:.4f} | Macro F1: {test_metrics_liar_bin.macro_f1:.4f}")

    results = {
        "ifnd_val": val_metrics_ifnd.to_dict(),
        "ifnd_test": test_metrics_ifnd.to_dict(),
        "liar_val": val_metrics_liar.to_dict(),
        "liar_test": test_metrics_liar.to_dict(),
        "liar_bin_test": test_metrics_liar_bin.to_dict(),
        "hyperparams": baseline_ifnd.get_hyperparameters(),
    }
    return results


if __name__ == "__main__":
    run_baseline_evaluation()
