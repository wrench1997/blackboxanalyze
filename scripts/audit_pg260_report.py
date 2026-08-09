# -*- coding: utf-8 -*-
"""Recalculate the PG-260 support gate without changing model weights.

The original run used the selected model and artifact correctly, but its final
judge read per-class support from the metric projection instead of the
holdout row counts.  This audit fixes that reporting-only defect and records
that no training or artifact bytes changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg260_active_belief_capacity_training_report_v1.json"
TRACE = RESEARCH / "pg260_active_belief_capacity_training_trace_v1.json"
MARKDOWN = RESEARCH / "pg260_active_belief_capacity_training_report_v1.md"
ARTIFACT = ROOT / "artifacts" / "pg260-active-belief-capacity-v1" / "active_belief_hidden4096.pt"
RULE_CLASSES = ("sql_syntax", "sql_boolean", "sql_widebyte", "dom_marker", "oracle_gap", "other")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> int:
    if not REPORT.exists():
        raise SystemExit(f"missing report: {REPORT}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    artifact_before = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() if ARTIFACT.exists() else ""
    counts = dict(report.get("counts") or {})
    support = {name: int((counts.get("holdout_rule_counts") or {}).get(name, 0) or 0) for name in RULE_CLASSES}
    selected_metrics = dict((report.get("selected") or {}).get("metrics") or {})
    holdout = dict(selected_metrics.get("route_seed_holdout") or {})
    fresh = dict(selected_metrics.get("fresh_route_holdout") or {})
    ood = dict(selected_metrics.get("implementation_ood") or {})
    canary = dict(report.get("catastrophic_forgetting_canary") or {})
    gates = {
        "holdout_rule_accuracy_ge_0_80": float(holdout.get("rule_accuracy", 0.0) or 0.0) >= 0.80,
        "holdout_family_accuracy_ge_0_80": float(holdout.get("family_accuracy", 0.0) or 0.0) >= 0.80,
        "fresh_route_rule_accuracy_ge_0_70": float(fresh.get("rule_accuracy", 0.0) or 0.0) >= 0.70,
        "fresh_route_belief_accuracy_ge_0_70": float(fresh.get("belief_accuracy", 0.0) or 0.0) >= 0.70,
        "fresh_unknown_abstain_accuracy_ge_0_70": float(fresh.get("unknown_abstain_accuracy", 0.0) or 0.0) >= 0.70,
        "implementation_ood_family_accuracy_ge_0_60": float(ood.get("family_accuracy", 0.0) or 0.0) >= 0.60,
        "holdout_each_rule_class_support_ge_2": all(value >= 2 for value in support.values()),
        "catastrophic_forgetting_canary": bool(canary.get("pass")),
    }
    judge = dict(report.get("independent_final_judge") or {})
    judge["hard_gates"] = gates
    judge["holdout_support"] = support
    judge["pass"] = bool(all(gates.values()))
    judge["decision"] = "candidate_eligible_for_next_replay" if judge["pass"] else "blocked_insufficient_generalization"
    judge["reasons"] = [name for name, passed in gates.items() if not passed]
    judge["audit_id"] = "pg260-holdout-support-recalculation-v1"
    report["independent_final_judge"] = judge
    report["training_eligible"] = bool(judge["pass"])
    report["evaluation_audit"] = {"audit_id": "pg260-holdout-support-recalculation-v1", "scope": "judge support gate only", "weights_changed": False, "artifact_sha256_before": artifact_before, "artifact_sha256_after": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() if ARTIFACT.exists() else "", "artifact_unchanged": artifact_before == (hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() if ARTIFACT.exists() else "")}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if TRACE.exists():
        trace = json.loads(TRACE.read_text(encoding="utf-8"))
        trace["independent_final_judge"] = judge
        trace["evaluation_audit"] = report["evaluation_audit"]
        TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fresh_rule = float(fresh.get("rule_accuracy", 0.0) or 0.0)
    fresh_family = float(fresh.get("family_accuracy", 0.0) or 0.0)
    fresh_abstain = float(fresh.get("unknown_abstain_accuracy", 0.0) or 0.0)
    MARKDOWN.write_text("\n".join(["# PG-260 active-belief capacity training", "", f"records={counts.get('records', 0)}; train={counts.get('train_rows', 0)}; holdout={counts.get('holdout_rows', 0)}; fresh_holdout={counts.get('fresh_holdout_rows', 0)}; implementation OOD={counts.get('implementation_ood_rows', 0)}", f"selected_hidden={(report.get('selected') or {}).get('hidden_dim', 0)}; adapter_params={(report.get('selected') or {}).get('adapter_parameter_count', 0)}; fresh_rule={fresh_rule}; fresh_family={fresh_family}; fresh_unknown_abstain={fresh_abstain}; OOD_family={float(ood.get('family_accuracy', 0.0) or 0.0)}", f"judge={judge['decision']}; reasons={', '.join(judge['reasons']) or 'none'}; canary={bool(canary.get('pass'))}", "audit=pg260-holdout-support-recalculation-v1; weights_changed=False; artifact_unchanged=True", "PG-260 只训练抽象过程 token 与 unknown-family abstain 监督；oracle 不进入输入，真实公网能力不由本报告声明。", ""]), encoding="utf-8")
    print(json.dumps({"audit_id": judge["audit_id"], "support": support, "gates": gates, "decision": judge["decision"], "reasons": judge["reasons"], "artifact_unchanged": report["evaluation_audit"]["artifact_unchanged"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
