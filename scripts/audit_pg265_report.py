# -*- coding: utf-8 -*-
"""Independent audit for the PG-265 large-capacity growth training report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg265_growth_augmented_large_capacity_training_report_v1.json"
PG264_AUDIT = RESEARCH / "pg264_pikachu_growth_collection_audit_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg265-growth-augmented-large-capacity-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    pg264 = json.loads(PG264_AUDIT.read_text(encoding="utf-8-sig"))
    variants = {int(item.get("hidden_dim", 0) or 0): item for item in list(report.get("capacity_variant_metrics") or []) if isinstance(item, dict)}
    expected = [4096, 8192, 12288]
    judge = dict(report.get("independent_final_judge") or {})
    gates = {
        "status_completed": report.get("status") == "completed_pg265_growth_augmented_large_capacity_training",
        "capacity_variants_4096_8192_12288": sorted(variants) == expected,
        "pg264_audit_complete": bool(pg264.get("all_required_fields_complete")) and int(pg264.get("audited_record_count", 0) or 0) == 32,
        "pg264_growth_count_32": int((report.get("growth_counts") or {}).get("pg264_records", 0) or 0) == 32,
        "oracle_target_off_input": bool(report.get("model_input_excludes_oracle_target")),
        "raw_payload_and_response_bodies_excluded": bool((report.get("honesty") or {}).get("raw_payload_strings_stored") is False and (report.get("honesty") or {}).get("raw_response_bodies_stored") is False),
        "artifact_present": any(ARTIFACT_DIR.glob("*.pt")),
        "judge_decision_is_bounded": judge.get("decision") in {"candidate_eligible_for_next_replay", "blocked_insufficient_generalization"},
        "report_hash_matches": str(report.get("report_sha256", "")) == _digest(dict(report, report_sha256="")),
    }
    audit = {"audit_id": "pg265-final-report-audit-v1", "schema_version": "pg265-large-capacity-training-audit-v1", "gates": gates, "pass": bool(all(gates.values())), "decision": "candidate_eligible_for_next_replay" if all(gates.values()) else "blocked_insufficient_generalization", "reasons": [key for key, value in gates.items() if not value], "selected_hidden_dim": int((report.get("selected") or {}).get("hidden_dim", 0) or 0), "selected_adapter_parameter_count": int((report.get("selected") or {}).get("adapter_parameter_count", 0) or 0), "pg264_audit_sha256": str(pg264.get("audit_sha256", "")), "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    report["evaluation_audit"] = dict(report.get("evaluation_audit") or {}, audit_id=audit["audit_id"], audit_pass=audit["pass"], audit_reasons=audit["reasons"], weights_changed=False)
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    (RESEARCH / "pg265_growth_augmented_large_capacity_training_audit_v1.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_id": audit["audit_id"], "pass": audit["pass"], "decision": audit["decision"], "reasons": audit["reasons"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

