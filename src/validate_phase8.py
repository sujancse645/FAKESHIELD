"""validate_phase8.py - Automated validation suite for FakeShield Phase 8.

Performs at least 18 rigorous verification checks covering:
1. NLI module imports cleanly
2. NLI model loads
3. NLI inference works on representative pairs
4. NLI labels correctly normalized (ENTAILMENT, CONTRADICTION, NEUTRAL)
5. Evaluation dataset valid (30 claim-evidence pairs)
6. Expected columns present in evaluation dataset
7. NLI metrics calculated and valid
8. Confusion matrix generated and valid
9. Source-tier YAML configuration valid (Tiers 1-4)
10. Evidence fusion engine initializes and works
11. Support score correctly calculated with tier weighting
12. Contradiction score correctly calculated with tier weighting
13. Neutral evidence handled correctly (0.0 support, 0.0 contradiction)
14. Abstention works (UNVERIFIED produced on weak/missing evidence)
15. Out-of-context logic works (temporal mismatch flagged)
16. Reasoning output schema valid with confidence and explanation
17. 20-section Phase 8 technical report exists
18. Phase 1 raw dataset SHA-256 hashes unchanged
"""

import os
import sys
import hashlib
import yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "nli")
NLI_SAMPLES_CSV = os.path.join(DATA_DIR, "nli_samples.csv")
NLI_RESULTS_CSV = os.path.join(DATA_DIR, "nli_results.csv")
NLI_METRICS_CSV = os.path.join(DATA_DIR, "nli_metrics.csv")
NLI_CM_CSV = os.path.join(DATA_DIR, "nli_confusion_matrix.csv")
NLI_DETERMINISM_CSV = os.path.join(DATA_DIR, "nli_determinism.csv")
REASONING_RESULTS_CSV = os.path.join(DATA_DIR, "reasoning_results.csv")
REPORT_PATH = os.path.join(DATA_DIR, "phase8_report.txt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "evidence_fusion.yaml")

RAW_FILES = {
    "IFND.csv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "IFND", "IFND.csv"),
        "3728c7b785ea93c9dadcc61a4c549c0eddd71a5fb6a1a31be08b5235ee4f5bfa",
    ),
    "train.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "train.tsv"),
        "23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36",
    ),
    "test.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "test.tsv"),
        "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f",
    ),
    "valid.tsv": (
        os.path.join(PROJECT_ROOT, "data", "raw", "LIAR", "valid.tsv"),
        "8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16",
    ),
}


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_validation():
    print("=" * 80)
    print("FAKESHIELD PHASE 8: AUTOMATED VALIDATION SUITE")
    print("=" * 80)

    checks = []

    def record_check(num: int, title: str, passed: bool, detail: str = ""):
        status_str = "PASS" if passed else "FAIL"
        checks.append((num, title, passed, detail))
        print(f"Check {num:02d}: [{status_str}] {title} {detail}")

    # Check 1: NLI module imports cleanly
    try:
        from src.nli import NLIModel, NLIConfig, NLIQualityAnalyzer
        record_check(1, "NLI module imports cleanly", True)
    except Exception as e:
        record_check(1, "NLI module imports cleanly", False, str(e))

    # Check 2: NLI model loads
    try:
        cfg = NLIConfig()
        nli = NLIModel(cfg)
        record_check(2, f"NLI model loads successfully ('{cfg.model_id}')", True)
    except Exception as e:
        record_check(2, "NLI model loads successfully", False, str(e))

    # Check 3: NLI inference works on representative pairs
    try:
        c = "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent."
        e = "The Monetary Policy Committee (MPC) of the Reserve Bank of India decided to keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent."
        res = nli.predict(c, e)
        inf_ok = res["label"] == "ENTAILMENT" and res["score"] > 0.8
        record_check(3, f"NLI inference works (score={res['score']:.4f})", inf_ok)
    except Exception as e:
        record_check(3, "NLI inference works", False, str(e))

    # Check 4: NLI labels correctly normalized
    try:
        labels_set = set(res["probabilities"].keys())
        expected_labels = {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}
        norm_ok = labels_set == expected_labels and res["label"] in expected_labels
        record_check(4, "NLI labels normalized to canonical set", norm_ok)
    except Exception as e:
        record_check(4, "NLI labels normalized", False, str(e))

    # Check 5: Evaluation dataset valid (30 claim-evidence pairs)
    try:
        df_samples = pd.read_csv(NLI_SAMPLES_CSV)
        count_ok = len(df_samples) == 30
        record_check(5, f"Evaluation dataset valid ({len(df_samples)} pairs)", count_ok)
    except Exception as e:
        record_check(5, "Evaluation dataset valid", False, str(e))

    # Check 6: All expected columns present in evaluation dataset
    try:
        has_id = "sample_id" in df_samples.columns
        has_claim = "claim" in df_samples.columns
        has_evidence = ("evidence_text" in df_samples.columns) or ("evidence" in df_samples.columns)
        has_label = ("expected_label" in df_samples.columns) or ("expected_nli_label" in df_samples.columns)
        has_source = ("publisher" in df_samples.columns) or ("source" in df_samples.columns) or ("source_tier" in df_samples.columns)
        has_notes = "notes" in df_samples.columns
        cols_ok = all([has_id, has_claim, has_evidence, has_label, has_source, has_notes])
        record_check(6, "Expected columns present in NLI evaluation dataset", cols_ok)
    except Exception as e:
        record_check(6, "Expected columns present", False, str(e))

    # Check 7: NLI metrics calculated and valid
    try:
        df_metrics = pd.read_csv(NLI_METRICS_CSV)
        acc_row = df_metrics[(df_metrics["metric"] == "accuracy") & (df_metrics["class"] == "overall")]
        metrics_ok = len(acc_row) > 0 and float(acc_row["value"].values[0]) > 0.5
        record_check(7, f"NLI metrics valid (accuracy={acc_row['value'].values[0]*100:.1f}%)", metrics_ok)
    except Exception as e:
        record_check(7, "NLI metrics valid", False, str(e))

    # Check 8: Confusion matrix valid
    try:
        df_cm = pd.read_csv(NLI_CM_CSV, index_col=0)
        cm_ok = df_cm.shape == (3, 3) and df_cm.sum().sum() == 30
        record_check(8, f"Confusion matrix valid (shape={df_cm.shape}, total={df_cm.sum().sum()})", cm_ok)
    except Exception as e:
        record_check(8, "Confusion matrix valid", False, str(e))

    # Check 9: Source-tier YAML configuration valid
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f)
        has_tiers = all(t in cfg_yaml.get("source_tier_weights", {}) for t in [1, 2, 3, 4])
        has_thresh = "thresholds" in cfg_yaml and "min_evidence_strength" in cfg_yaml["thresholds"]
        record_check(9, "Source-tier configuration valid in YAML", has_tiers and has_thresh)
    except Exception as e:
        record_check(9, "Source-tier configuration valid", False, str(e))

    # Check 10: Evidence fusion engine initializes and works
    try:
        from src.evidence_fusion import EvidenceFusionEngine, FusionConfig
        f_cfg = FusionConfig.from_yaml(CONFIG_PATH)
        engine = EvidenceFusionEngine(config=f_cfg, nli_model=nli)
        record_check(10, "Evidence fusion engine initialized cleanly", True)
    except Exception as e:
        record_check(10, "Evidence fusion engine initialized", False, str(e))

    # Check 11: Support score correctly calculated with tier weighting
    try:
        ev_t1 = [{"text": e, "tier": 1, "publisher": "RBI", "date": "2024-04-05"}]
        r_sup = engine.fuse(c, ev_t1, "c_sup")
        sup_ok = r_sup.support_score > 0.7 and r_sup.verdict == "VERIFIED"
        record_check(11, f"Support score calculated with tier weighting (sup={r_sup.support_score:.2f})", sup_ok)
    except Exception as e:
        record_check(11, "Support score calculated", False, str(e))

    # Check 12: Contradiction score correctly calculated with tier weighting
    try:
        c_con = "The Reserve Bank of India reduced the repo rate to 5.0 percent."
        r_con = engine.fuse(c_con, ev_t1, "c_con")
        con_ok = r_con.contradiction_score > 0.7 and r_con.verdict == "CONTRADICTED"
        record_check(12, f"Contradiction score calculated with tier weighting (con={r_con.contradiction_score:.2f})", con_ok)
    except Exception as e:
        record_check(12, "Contradiction score calculated", False, str(e))

    # Check 13: Neutral evidence handled correctly
    try:
        c_neut = "The stock market rose by 2 percent today."
        r_neut = engine.fuse(c_neut, ev_t1, "c_neut")
        neut_ok = r_neut.support_score == 0.0 and r_neut.contradiction_score == 0.0
        record_check(13, "Neutral evidence correctly contributes 0.0 support and 0.0 contradiction", neut_ok)
    except Exception as e:
        record_check(13, "Neutral evidence handled", False, str(e))

    # Check 14: Abstention works (UNVERIFIED on weak/no evidence)
    try:
        r_abst = engine.fuse("Unretrieved claim about an unknown event.", [], "c_abst")
        abst_ok = r_abst.verdict == "UNVERIFIED"
        record_check(14, "Abstention logic produces UNVERIFIED on empty evidence", abst_ok)
    except Exception as e:
        record_check(14, "Abstention logic produces UNVERIFIED", False, str(e))

    # Check 15: Out-of-context logic works
    try:
        c_ooc = "Today the Prime Minister announced a nationwide lockdown to contain COVID-19."
        ev_ooc = [{"text": "The Prime Minister today announced a nationwide lockdown to contain COVID-19 transmission.", "tier": 1, "publisher": "PIB", "date": "2020-03-24"}]
        r_ooc = engine.fuse(c_ooc, ev_ooc, "c_ooc")
        ooc_ok = r_ooc.temporal_status == "out_of_context" and r_ooc.verdict == "OUT-OF-CONTEXT"
        record_check(15, f"Out-of-context detected (temporal_status='{r_ooc.temporal_status}')", ooc_ok)
    except Exception as e:
        record_check(15, "Out-of-context detected", False, str(e))

    # Check 16: Reasoning output schema valid
    try:
        df_reasoning = pd.read_csv(REASONING_RESULTS_CSV)
        req_r_cols = {"sample_id", "claim", "evidence_count", "support_score", "contradiction_score", "temporal_status", "verdict", "confidence", "explanation"}
        schema_ok = req_r_cols.issubset(set(df_reasoning.columns)) and len(df_reasoning) > 0
        record_check(16, f"Reasoning output schema valid ({len(df_reasoning)} rows)", schema_ok)
    except Exception as e:
        record_check(16, "Reasoning output schema valid", False, str(e))

    # Check 17: 20-section Phase 8 technical report exists
    try:
        exists_rep = os.path.exists(REPORT_PATH)
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        all_sec = all(f"{i}. " in content for i in range(1, 21))
        record_check(17, "Technical report exists with all 20 sections", exists_rep and all_sec)
    except Exception as e:
        record_check(17, "Technical report exists", False, str(e))

    # Check 18: Phase 1 raw hashes unchanged
    try:
        hashes_match = True
        for name, (path, expected) in RAW_FILES.items():
            actual = compute_sha256(path)
            if actual != expected:
                hashes_match = False
                break
        record_check(18, "Phase 1 raw dataset SHA-256 hashes match baseline", hashes_match)
    except Exception as e:
        record_check(18, "Phase 1 raw dataset SHA-256 hashes match baseline", False, str(e))

    print("\n" + "=" * 80)
    passed_count = sum(1 for c in checks if c[2])
    total_count = len(checks)
    print(f"VALIDATION SUMMARY: {passed_count}/{total_count} CHECKS PASSED")
    print("=" * 80)

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(run_validation())
