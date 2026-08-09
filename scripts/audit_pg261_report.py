# -*- coding: utf-8 -*-
"""Independent report-only audit for PG-261.

This script is intentionally separate from the GPU runner.  It recalculates
the hard gates from the final report, checks that the three capacity branches
and the mask/padding audit are present, and never changes model weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg261_masked_active_belief_capacity_training_report_v1.json"
TRACE = RESEARCH / "pg261_masked_active_belief_capacity_training_trace_v1.json"
MARKDOWN = RESEARCH / "pg261_masked_active_belief_capacity_training_report_v1.md"
ARTIFACT = ROOT / "artifacts" / "pg261-masked-active-belief-capacity-v1" / "active_belief_hidden4096.pt"
RULE_CLASSES = ("sql_syntax", "sql_boolean", "sql_widebyte", "dom_marker", "oracle_gap", "other")
EXPECTED_VARIANTS = (2048, 4096, 8192)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _report_hash_matches(report: dict[str, Any]) -> bool:
    observed = str(report.get("report_sha256", ""))
    if not observed:
        return False
    candidate = dict(report)
    candidate["report_sha256"] = ""
    return observed == _digest(candidate)


def main() -> int:
    if not REPORT.exists():
        raise SystemExit(f"missing report: {REPORT}")
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    selected = dict(report.get("selected") or {})
    metrics = dict(selected.get("metrics") or {})
    holdout = dict(metrics.get("route_seed_holdout") or {})
    fresh = dict(metrics.get("fresh_route_holdout") or {})
    ood = dict(metrics.get("implementation_ood") or {})
    counts = dict(report.get("counts") or {})
    support = {name: int((counts.get("holdout_rule_counts") or {}).get(name, 0) or 0) for name in RULE_CLASSES}
    architecture = dict(report.get("architecture_change") or {})
    variants = list(report.get("capacity_variant_metrics") or [])
    variant_dims = tuple(int(row.get("hidden_dim", 0) or 0) for row in variants if isinstance(row, dict))
    gates = {
        "capacity_sweep_has_2048_4096_8192": set(variant_dims) == set(EXPECTED_VARIANTS) and len(variant_dims) == len(EXPECTED_VARIANTS),
        "mask_aware_pooling_enabled": bool(architecture.get("masked_mean_pool")) and bool(architecture.get("padding_invariant_classification")),
        "holdout_rule_accuracy_ge_0_80": float(holdout.get("rule_accuracy", 0.0) or 0.0) >= 0.80,
        "holdout_family_accuracy_ge_0_80": float(holdout.get("family_accuracy", 0.0) or 0.0) >= 0.80,
        "fresh_route_rule_accuracy_ge_0_70": float(fresh.get("rule_accuracy", 0.0) or 0.0) >= 0.70,
        "fresh_route_belief_accuracy_ge_0_70": float(fresh.get("belief_accuracy", 0.0) or 0.0) >= 0.70,
        "fresh_unknown_abstain_accuracy_ge_0_70": float(fresh.get("unknown_abstain_accuracy", 0.0) or 0.0) >= 0.70,
        "implementation_ood_family_accuracy_ge_0_60": float(ood.get("family_accuracy", 0.0) or 0.0) >= 0.60,
        "holdout_each_rule_class_support_ge_2": all(value >= 2 for value in support.values()),
        "catastrophic_forgetting_canary": bool((report.get("catastrophic_forgetting_canary") or {}).get("pass")),
        "report_hash_matches": _report_hash_matches(report),
    }
    judge = dict(report.get("independent_final_judge") or {})
    judge.update({"hard_gates": gates, "holdout_support": support, "pass": bool(all(gates.values())), "decision": "candidate_eligible_for_next_replay" if all(gates.values()) else "blocked_insufficient_generalization", "reasons": [name for name, passed in gates.items() if not passed], "audit_id": "pg261-final-report-audit-v1"})
    report["independent_final_judge"] = judge
    report["training_eligible"] = bool(judge["pass"])
    report["evaluation_audit"] = {"audit_id": "pg261-final-report-audit-v1", "scope": "capacity/mask/support/judge projection", "weights_changed": False, "artifact_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() if ARTIFACT.exists() else "", "artifact_present": ARTIFACT.exists()}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if TRACE.exists():
        trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
        trace["independent_final_judge"] = judge
        trace["evaluation_audit"] = report["evaluation_audit"]
        TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if MARKDOWN.exists():
        MARKDOWN.write_text(MARKDOWN.read_text(encoding="utf-8") + f"\nfinal_audit=pg261-final-report-audit-v1; decision={judge['decision']}; reasons={', '.join(judge['reasons']) or 'none'}; capacity_variants={list(variant_dims)}\n", encoding="utf-8")
    print(json.dumps({"audit_id": judge["audit_id"], "capacity_variants": list(variant_dims), "gates": gates, "decision": judge["decision"], "reasons": judge["reasons"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
