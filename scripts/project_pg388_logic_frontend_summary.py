"""Build a bounded, read-only PG-388 frontend research projection.

The UI needs to show where the Rule-IR experiment stands without loading the
dataset rows or evaluator-side evidence into the browser.  This projection
therefore contains counts, slot names, audit statuses, hashes and fail-closed
gates only.  It deliberately omits context/target token sequences, row IDs,
payloads, wire data, response bodies and oracle answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg388_logic_rule_ir_composition_dataset_v1.json"
DEFAULT_AUDIT = ROOT / "research" / "pg388_logic_rule_ir_composition_audit_v1.json"
DEFAULT_PLAN = ROOT / "research" / "pg388_logic_composed_candidate_plan_v1.json"
DEFAULT_LIVE = ROOT / "research" / "pg388_logic_canary_live_v1.json"
DEFAULT_ROW_BOUND_REPORT = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_v1.json"
DEFAULT_ROW_BOUND_AUDIT = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "frontend" / "public" / "research" / "pg388_logic_rule_ir_frontend_summary_v1.json"

SCHEMA_VERSION = "pg388-logic-rule-ir-frontend-summary-v1"
FORBIDDEN_KEYS = {
    "rows",
    "context_tokens",
    "target_tokens",
    "record_ref_sha256",
    "row_sha256",
    "payload",
    "wire",
    "response_body",
    "oracle_answer",
    "evaluator_answer",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_counts(rows: list[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            split = str(row.get("split", "unknown"))
            result[split] = result.get(split, 0) + 1
    return dict(sorted(result.items()))


def _safe_promotion(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        }
    return {
        key: value.get(key) is True
        for key in (
            "training_allowed",
            "memory_promotion_allowed",
            "payload_catalog_promotion_allowed",
            "vulnerability_claim_allowed",
        )
    }


def build_summary(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    audit_path: Path = DEFAULT_AUDIT,
    plan_path: Path = DEFAULT_PLAN,
    live_path: Path = DEFAULT_LIVE,
    row_bound_report_path: Path = DEFAULT_ROW_BOUND_REPORT,
    row_bound_audit_path: Path = DEFAULT_ROW_BOUND_AUDIT,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    audit = _load(audit_path)
    plan = _load(plan_path)
    live = _load(live_path)
    row_bound = _load(row_bound_report_path) if row_bound_report_path.exists() else None
    row_bound_audit = _load(row_bound_audit_path) if row_bound_audit_path.exists() else None
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ValueError("PG-388 dataset rows must be a list")
    slot_order = dataset.get("slot_order")
    if not isinstance(slot_order, list) or not slot_order or not all(isinstance(item, str) for item in slot_order):
        raise ValueError("PG-388 slot_order is missing")
    case_refs = {str(row.get("case_ref")) for row in rows if isinstance(row, dict) and row.get("case_ref")}
    implementation_refs = {
        str(row.get("implementation_ref"))
        for row in rows
        if isinstance(row, dict) and row.get("implementation_ref")
    }
    live_coverage = dataset.get("live_coverage") if isinstance(dataset.get("live_coverage"), dict) else {}
    source_contract = dataset.get("source_contract") if isinstance(dataset.get("source_contract"), dict) else {}
    plan_gate = plan.get("gate") if isinstance(plan.get("gate"), dict) else {}
    plan_counts = plan.get("counts") if isinstance(plan.get("counts"), dict) else {}
    dataset_counts = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else {}
    unknown_holdout = plan.get("unknown_holdout_slot_values", {})
    if isinstance(unknown_holdout, dict):
        unknown_holdout_count = len(unknown_holdout)
    elif isinstance(unknown_holdout, (int, float)) and not isinstance(unknown_holdout, bool):
        unknown_holdout_count = int(unknown_holdout)
    else:
        unknown_holdout_count = 0
    row_bound_counts = row_bound.get("counts") if isinstance(row_bound, dict) and isinstance(row_bound.get("counts"), dict) else {}
    row_bound_contract = row_bound.get("source_contract") if isinstance(row_bound, dict) and isinstance(row_bound.get("source_contract"), dict) else {}
    if row_bound is not None:
        live_projection = {
            "status": str(row_bound.get("status", "unknown")),
            "fresh_resets": int(row_bound_counts.get("fresh_resets", 0)),
            "typed_observations": int(row_bound_counts.get("typed", 0)),
            "candidate_effects": int(row_bound_counts.get("failure_repair", 0)),
            "negative_control_clean": max(0, int(row_bound_counts.get("typed", 0)) - int(row_bound_counts.get("negative_violations", 0))),
            "unsafe_allow": 0,
            "row_bound": row_bound_contract.get("row_bound_typed_evidence") is True,
            "report_file": row_bound_report_path.name,
            "report_sha256": _sha256(row_bound_report_path),
            "audit_status": str((row_bound_audit or {}).get("status", "pending")),
            "audit_sha256": _sha256(row_bound_audit_path) if row_bound_audit_path.exists() else "",
        }
    else:
        live_projection = {
            "status": str(live_coverage.get("status", "unknown")),
            "fresh_resets": int(live_coverage.get("fresh_resets", 0)),
            "typed_observations": int(live_coverage.get("typed_observations", 0)),
            "candidate_effects": int(live_coverage.get("candidate_effects", 0)),
            "negative_control_clean": int(live_coverage.get("negative_control_clean", 0)),
            "unsafe_allow": int(live_coverage.get("unsafe_allow", 0)),
            "row_bound": live_coverage.get("row_bound") is True,
            "report_file": live_path.name,
            "report_sha256": _sha256(live_path),
            "audit_status": "aggregate_only",
            "audit_sha256": "",
        }
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_rule_ir_candidate",
        "dataset": {
            "file": dataset_path.name,
            "sha256": _sha256(dataset_path),
            "status": str(dataset.get("status", "unknown")),
            "records": len(rows),
            "split_counts": _split_counts(rows),
            "declared_counts": {
                "records": int(dataset_counts.get("records", len(rows))),
                "train": int(dataset_counts.get("train", 0)),
                "implementation_holdout": int(dataset_counts.get("implementation_holdout", 0)),
                "slot_count": len(slot_order),
            },
            "case_count": len(case_refs),
            "implementation_count": len(implementation_refs),
            "slot_order": slot_order,
        },
        "audit": {
            "file": audit_path.name,
            "sha256": _sha256(audit_path),
            "status": str(audit.get("status", "unknown")),
            "records": int(audit.get("records", 0)),
            "invalid_rows": int(audit.get("invalid_rows", 0)),
            "unique_row_hashes": int(audit.get("unique_row_hashes", 0)),
            "context_firewall_passed": audit.get("context_firewall_passed") is True,
            "training_eligible": int(audit.get("training_eligible", 0)),
            "failure_count": len(audit.get("failures", [])) if isinstance(audit.get("failures"), list) else 0,
        },
        "live_evidence": live_projection,
        "plan": {
            "file": plan_path.name,
            "sha256": _sha256(plan_path),
            "status": str(plan.get("status", "unknown")),
            "records": int(plan_counts.get("records", 0)),
            "required_context_window": int(plan.get("required_context_window", 0)),
                "unknown_holdout_slot_value_count": unknown_holdout_count,
            "optimizer_started": plan_gate.get("optimizer_started") is True,
            "failures": [str(item) for item in plan_gate.get("failures", []) if isinstance(item, str)],
        },
        "gates": {
            "abstract_rule_ir_audit": audit.get("status") == "passed_candidate_rule_ir_audit",
            "context_firewall": audit.get("context_firewall_passed") is True,
            "source_row_bound_typed_evidence": (row_bound_contract.get("row_bound_typed_evidence") if row_bound is not None else source_contract.get("row_bound_typed_evidence")) is True,
            "fresh_role_reset_attested": (row_bound_contract.get("fresh_role_reset_attested") if row_bound is not None else source_contract.get("fresh_role_reset_attested")) is True,
            "operator_reviewed": (row_bound_contract.get("operator_reviewed") if row_bound is not None else source_contract.get("operator_reviewed")) is True,
            "candidate_reference_negative_replay": (row_bound_contract.get("candidate_reference_negative_replay") if row_bound is not None else source_contract.get("candidate_reference_negative_replay")) is True,
            "capability_training_allowed": False,
            "training_eligible": False,
            "promotion": _safe_promotion(dataset.get("promotion")),
        },
        "ui_boundary": {
            "context_contains_raw_payload": False,
            "context_contains_wire": False,
            "context_contains_raw_markup": False,
            "answers_out_of_context": True,
            "claim": "abstract Rule-IR candidate only; not a generic vulnerability or payload capability result",
        },
    }
    _assert_safe_projection(output)
    return output


def _assert_safe_projection(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden frontend projection key: {key}")
            _assert_safe_projection(item)
    elif isinstance(value, list):
        for item in value:
            _assert_safe_projection(item)


def write_summary(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    summary = build_summary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": summary["status"], "records": summary["dataset"]["records"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
