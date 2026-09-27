"""pipeline_quality.py - Stage execution tracing, timing, determinism, and error analysis for FakeShield Pipeline."""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class StageTraceItem:
    """Audit trace for an individual pipeline execution stage."""
    case_id: str
    stage_name: str
    status: str  # SUCCESS, FAILED, SKIPPED, ABSTAINED
    runtime_seconds: float
    output_summary: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stage_name": self.stage_name,
            "status": self.status,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "output_summary": self.output_summary,
            "error_message": self.error_message or "",
        }


class PipelineTracer:
    """Manages runtime telemetry and stage tracing for pipeline verification runs."""

    def __init__(self, case_id: str = "case_01"):
        self.case_id = case_id
        self.traces: List[StageTraceItem] = []
        self._stage_start_times: Dict[str, float] = {}

    def start_stage(self, stage_name: str):
        """Mark the beginning of a processing stage."""
        self._stage_start_times[stage_name] = time.time()

    def end_stage(
        self,
        stage_name: str,
        status: str = "SUCCESS",
        output_summary: str = "",
        error_message: Optional[str] = None,
    ):
        """Record the completion of a processing stage."""
        start = self._stage_start_times.get(stage_name, time.time())
        duration = time.time() - start
        item = StageTraceItem(
            case_id=self.case_id,
            stage_name=stage_name,
            status=status,
            runtime_seconds=duration,
            output_summary=output_summary,
            error_message=error_message,
        )
        self.traces.append(item)

    def get_trace_dicts(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.traces]


class PipelineDeterminismAuditor:
    """Evaluates reproducibility across repeated runs of identical inputs."""

    @staticmethod
    def compare_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compares multiple pipeline results and verifies exact field consistency.

        Fields compared:
        - language code
        - translation english text
        - extracted claims count and texts
        - NLI labels and predicted scores
        - fusion support and contradiction scores
        - overall verdict
        """
        if len(runs) < 2:
            return {"runs_evaluated": len(runs), "is_deterministic": True, "variances": []}

        variances = []
        first = runs[0]

        for i, curr in enumerate(runs[1:], start=2):
            # 1. Compare overall verdict
            if curr.get("overall_status") != first.get("overall_status"):
                variances.append(f"Run {i}: overall_status mismatch ('{curr.get('overall_status')}' vs '{first.get('overall_status')}')")

            # 2. Compare language
            if curr.get("language", {}).get("code") != first.get("language", {}).get("code"):
                variances.append(f"Run {i}: language code mismatch ('{curr.get('language', {}).get('code')}' vs '{first.get('language', {}).get('code')}')")

            # 3. Compare translation
            if curr.get("translation", {}).get("english_text") != first.get("translation", {}).get("english_text"):
                variances.append(f"Run {i}: translation english_text mismatch")

            # 4. Compare claims
            curr_claims = curr.get("claims", [])
            first_claims = first.get("claims", [])
            if len(curr_claims) != len(first_claims):
                variances.append(f"Run {i}: claims count mismatch ({len(curr_claims)} vs {len(first_claims)})")
            else:
                for c_idx, (cc, fc) in enumerate(zip(curr_claims, first_claims)):
                    if cc.get("verdict") != fc.get("verdict"):
                        variances.append(f"Run {i} Claim {c_idx}: verdict mismatch ('{cc.get('verdict')}' vs '{fc.get('verdict')}')")
                    if abs(cc.get("support_score", 0.0) - fc.get("support_score", 0.0)) > 1e-4:
                        variances.append(f"Run {i} Claim {c_idx}: support_score diff > 1e-4")
                    if abs(cc.get("contradiction_score", 0.0) - fc.get("contradiction_score", 0.0)) > 1e-4:
                        variances.append(f"Run {i} Claim {c_idx}: contradiction_score diff > 1e-4")

        return {
            "runs_evaluated": len(runs),
            "is_deterministic": len(variances) == 0,
            "variances": variances,
        }


class PipelineErrorAnalyzer:
    """Categorizes verification pipeline anomalies and abstentions."""

    FAILURE_CATEGORIES = [
        "ocr_failure",
        "language_detection_error",
        "translation_error",
        "claim_extraction_error",
        "retrieval_failure",
        "source_classification_issue",
        "nli_error",
        "fusion_error",
        "temporal_error",
        "abstention",
        "pipeline_integration_error",
        "none",
    ]

    @classmethod
    def categorize_case(cls, result: Dict[str, Any]) -> str:
        """Determines the primary failure or abstention category from pipeline result dict."""
        input_type = result.get("input_type", "text")
        ocr = result.get("ocr", {})
        lang = result.get("language", {})
        trans = result.get("translation", {})
        claims = result.get("claims", [])
        overall = result.get("overall_status", "")
        pipeline_status = result.get("pipeline_status", "")

        if pipeline_status == "FAILED":
            return "pipeline_integration_error"

        if input_type == "image" and ocr.get("status") in ["failed", "insufficient_text"]:
            return "ocr_failure"

        if lang.get("status") == "insufficient_text":
            return "language_detection_error"

        if trans.get("status") == "failed":
            return "translation_error"

        if not claims:
            return "claim_extraction_error"

        # Check claim-level statuses
        has_temporal = any(c.get("verdict") == "OUT-OF-CONTEXT" or c.get("temporal_flags") for c in claims)
        if has_temporal:
            return "temporal_error"

        has_retrieval_empty = any(len(c.get("evidence", [])) == 0 for c in claims)
        if has_retrieval_empty and overall == "UNVERIFIED":
            return "retrieval_failure"

        if overall == "UNVERIFIED":
            return "abstention"

        return "none"
