"""Classify PG-36 formal-candidate failure without changing the model.

This is a read-only experiment-vs-engineering triage record.  It deliberately
does not authorize a checkpoint update, data promotion, or infrastructure
scaling; it only preserves the evidence-backed next hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiment_engineering_triage import triage_failure  # noqa: E402


FORMAL_REPORT = ROOT / "research" / "pg36_formal_rule_ir_candidate_report_v1.json"
REGRESSION_MARKER = "266 passed"
REPORT_PATH = ROOT / "research" / "pg36_formal_rule_ir_triage_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg36_formal_rule_ir_triage_v1.md"


def main() -> int:
    report = json.loads(FORMAL_REPORT.read_text(encoding="utf-8"))
    triage = triage_failure(
        experiment_signals=["failure_reproduces_at_small_scale", "family_holdout_regression", "seed_or_split_sensitive"],
        engineering_signals=[],
        experiment_gate_passed=False,
        engineering_gate_passed=True,
        evidence={
            "formal_report_sha256": hashlib.sha256(FORMAL_REPORT.read_bytes()).hexdigest(),
            "full_regression": REGRESSION_MARKER,
            "family_holdout_typed_recall": report["splits"]["family_holdout"]["typed_recall"],
            "ood_source_typed_recall": report["splits"]["ood_source"]["typed_recall"],
            "source_holdout_typed_recall": report["source_holdout"]["typed_recall"],
            "unknown_false_positive_rate": report["unknown_abstain"]["false_positive_rate"],
            "capability_gate_status": report["capability_gate"]["status"],
        },
    )
    result = {
        "schema_version": "pg-pk-36-formal-rule-ir-triage-v1",
        "protocol_id": "PG-36",
        "status": "experiment_problem_no_model_change",
        "classification": triage["classification"],
        "triage": triage,
        "decision": {
            "model_change_authorized_by_triage": False,
            "checkpoint_promotion": False,
            "memory_promotion": False,
            "infrastructure_scale": False,
            "next_experiment": "PG-37 representation-gap ablation with paired response-shape counterfactuals",
            "reason": "family-general detection fails while local regression and lineage checks pass; do not mask the failure with more compute",
        },
        "raw_data_retained": False,
        "local_only": True,
    }
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-36 formal Rule IR 失败分流\n\n"
        f"分类：`{result['classification']}`。\n\n"
        "完整回归通过，问题复现于小规模和族外留出，因此归类为实验/表示泛化问题，不是工程资源问题。\n\n"
        "禁止模型晋升、长期记忆写入和扩容；下一轮先做 PG-37 表示缺口消融。\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol_id": result["protocol_id"], "classification": result["classification"], "decision": result["decision"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
