# -*- coding: utf-8 -*-
"""Independent report-only audit for PG-263.

PG-263 is the first capacity run that consumes the fresh PG-262 Pikachu
records.  This verifier recomputes the promotion gates from the finished
report and the immutable PG-262 integrity sidecar.  It never loads or edits
model weights and it never promotes a payload or a vulnerability claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_report_v1.json"
TRACE = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_trace_v1.json"
MARKDOWN = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_report_v1.md"
PG262_REPORT = RESEARCH / "pg262_targeted_paired_trace_collection_report_v1.json"
PG262_AUDIT = RESEARCH / "pg262_targeted_paired_trace_collection_audit_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg263-pg262-augmented-masked-capacity-v1"
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


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    report = _load(REPORT)
    pg262_report = _load(PG262_REPORT)
    pg262_audit = _load(PG262_AUDIT)
    selected = dict(report.get("selected") or {})
    metrics = dict(selected.get("metrics") or {})
    route_seed = dict(metrics.get("route_seed_holdout") or {})
    fresh = dict(metrics.get("fresh_route_holdout") or {})
    ood = dict(metrics.get("implementation_ood") or {})
    counts = dict(report.get("counts") or {})
    support = {name: int((counts.get("holdout_rule_counts") or {}).get(name, 0) or 0) for name in RULE_CLASSES}
    architecture = dict(report.get("architecture_change") or {})
    variants = [row for row in list(report.get("capacity_variant_metrics") or []) if isinstance(row, dict)]
    variant_dims = tuple(int(row.get("hidden_dim", 0) or 0) for row in variants)
    pg262_counts = dict(pg262_report.get("counts") or {})
    pg262_audit_ok = bool(pg262_audit.get("all_required_fields_complete")) and int(pg262_audit.get("audited_record_count", 0) or 0) == 20 and str(pg262_audit.get("audit_id", "")) == "pg262-fresh-replay-integrity-v1"
    gates = {
        "capacity_sweep_has_2048_4096_8192": set(variant_dims) == set(EXPECTED_VARIANTS) and len(variant_dims) == len(EXPECTED_VARIANTS),
        "mask_aware_pooling_enabled": bool(architecture.get("masked_mean_pool")) and bool(architecture.get("padding_invariant_classification")),
        "oracle_target_excluded_from_model_input": bool(architecture.get("oracle_target_off_input")) and bool(report.get("model_input_excludes_oracle_target")),
        "pg262_audit_complete": pg262_audit_ok and int(counts.get("pg262_rows", 0) or 0) == int(pg262_counts.get("records", 0) or 0) == 20,
        "pg262_even_seed_holdout_present": int(counts.get("pg262_holdout_rows", 0) or 0) == 9,
        "holdout_rule_accuracy_ge_0_80": float(route_seed.get("rule_accuracy", 0.0) or 0.0) >= 0.80,
        "holdout_family_accuracy_ge_0_80": float(route_seed.get("family_accuracy", 0.0) or 0.0) >= 0.80,
        "fresh_route_rule_accuracy_ge_0_70": float(fresh.get("rule_accuracy", 0.0) or 0.0) >= 0.70,
        "fresh_route_belief_accuracy_ge_0_70": float(fresh.get("belief_accuracy", 0.0) or 0.0) >= 0.70,
        "fresh_unknown_abstain_accuracy_ge_0_70": float(fresh.get("unknown_abstain_accuracy", 0.0) or 0.0) >= 0.70,
        "implementation_ood_family_accuracy_ge_0_60": float(ood.get("family_accuracy", 0.0) or 0.0) >= 0.60,
        "holdout_each_rule_class_support_ge_2": all(value >= 2 for value in support.values()),
        "catastrophic_forgetting_canary": bool((report.get("catastrophic_forgetting_canary") or {}).get("pass")),
        "report_hash_matches": _report_hash_matches(report),
        "raw_payload_and_response_bodies_excluded": bool((report.get("honesty") or {}).get("raw_payload_strings_stored") is False) and bool((report.get("honesty") or {}).get("raw_response_bodies_stored") is False),
    }
    judge = dict(report.get("independent_final_judge") or {})
    judge.update({
        "authority": list(judge.get("authority") or []) + ["PG-262 fresh replay integrity audit", "PG-263 independent report audit"],
        "hard_gates": gates,
        "holdout_support": support,
        "pass": bool(all(gates.values())),
        "decision": "candidate_eligible_for_next_replay" if all(gates.values()) else "blocked_insufficient_generalization",
        "reasons": [name for name, passed in gates.items() if not passed],
        "audit_id": "pg263-final-report-audit-v1",
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
    })
    report["independent_final_judge"] = judge
    report["training_eligible"] = bool(judge["pass"])
    artifact_candidates = sorted(ARTIFACT_DIR.glob("*.pt"))
    report["evaluation_audit"] = {
        "audit_id": "pg263-final-report-audit-v1",
        "scope": "capacity/mask/PG-262-integrity/support/judge projection",
        "weights_changed": False,
        "artifact_present": bool(artifact_candidates),
        "artifact_sha256": hashlib.sha256(artifact_candidates[0].read_bytes()).hexdigest() if artifact_candidates else "",
        "pg262_audit_sha256": str(pg262_audit.get("audit_sha256", "")),
    }
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if TRACE.exists():
        trace = _load(TRACE)
        trace["independent_final_judge"] = judge
        trace["evaluation_audit"] = report["evaluation_audit"]
        TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if MARKDOWN.exists():
        MARKDOWN.write_text(MARKDOWN.read_text(encoding="utf-8") + f"\nfinal_audit=pg263-final-report-audit-v1; decision={judge['decision']}; reasons={', '.join(judge['reasons']) or 'none'}; capacity_variants={list(variant_dims)}\n", encoding="utf-8")
    print(json.dumps({"audit_id": judge["audit_id"], "capacity_variants": list(variant_dims), "gates": gates, "decision": judge["decision"], "reasons": judge["reasons"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
