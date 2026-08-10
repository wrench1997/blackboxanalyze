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
DEFAULT_HOLDOUT_B_REPORT = ROOT / "research" / "pg388_logic_holdout_b_source_rows_v1.json"
DEFAULT_HOLDOUT_B_AUDIT = ROOT / "research" / "pg388_logic_holdout_b_source_rows_audit_v1.json"
DEFAULT_HOLDOUT_B_DOCKER_SMOKE = ROOT / "research" / "pg388_logic_holdout_b_docker_smoke_v1.json"
DEFAULT_CROSS_IMPLEMENTATION_AUDIT = ROOT / "research" / "pg388_logic_cross_implementation_audit_v1.json"
DEFAULT_TAXONOMY_AUDIT = ROOT / "research" / "pg388_logic_taxonomy_audit_v1.json"
DEFAULT_CANDIDATE_REPORTS = (
    ("logic invariant", ROOT / "research" / "pg388_logic_token_cpu_smoke_v1.json"),
    ("supplemental logic", ROOT / "research" / "pg388_logic_supplement_token_cpu_smoke_v1.json"),
    ("trajectory canary", ROOT / "research" / "pg388_logic_canary_token_cpu_smoke_28case_v2.json"),
    ("11-slot composition", ROOT / "research" / "pg388_logic_composed_candidate_cpu_smoke_v1.json"),
    ("11-slot composition (full CPU)", ROOT / "research" / "pg388_logic_composed_candidate_cpu_full_v1.json"),
    ("11-slot composition (full CPU e8)", ROOT / "research" / "pg388_logic_composed_candidate_cpu_full_e8_v1.json"),
)
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


def _candidate_run_projection(label: str, path: Path) -> dict[str, Any]:
    """Project CPU candidate metrics without exposing sequences or evaluator data."""
    if not path.exists():
        return {
            "label": label,
            "status": "missing",
            "report_file": path.name,
            "report_sha256": "",
            "train_count": 0,
            "holdout_count": 0,
            "seed_count": 0,
            "vocabulary_scope": "missing",
            "vocabulary_size": 0,
            "weakest_head": {"name": "missing", "accuracy": 0.0},
            "holdout_composition_exact": 0.0,
            "holdout_slot_accuracy": 0.0,
            "holdout_repair_recall": 0.0,
            "holdout_composition_entropy": 0.0,
            "holdout_ask_recall": 0.0,
            "holdout_negative_false_allow": 0,
            "execution": {"optimizer_started": False, "device": "unknown", "gpu_touched": False},
            "training_eligible": False,
            "capability_training_allowed": False,
            "promotion": _safe_promotion(None),
        }
    report = _load(path)
    seeds = report.get("seeds") if isinstance(report.get("seeds"), list) else []
    holdouts = [seed.get("holdout", {}) for seed in seeds if isinstance(seed, dict) and isinstance(seed.get("holdout"), dict)]
    head_values: dict[str, list[float]] = {}
    for holdout in holdouts:
        head_accuracy = holdout.get("head_accuracy") if isinstance(holdout.get("head_accuracy"), dict) else {}
        if not head_accuracy and isinstance(holdout.get("per_slot"), dict):
            head_accuracy = {
                str(name): value.get("accuracy")
                for name, value in holdout["per_slot"].items()
                if isinstance(value, dict) and isinstance(value.get("accuracy"), (int, float))
            }
        for name, value in head_accuracy.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                head_values.setdefault(str(name), []).append(float(value))
    worst_heads = {name: min(values) for name, values in head_values.items() if values}
    weakest_name, weakest_accuracy = ("missing", 0.0)
    if worst_heads:
        weakest_name, weakest_accuracy = min(worst_heads.items(), key=lambda item: (item[1], item[0]))
    ask_values = [float(item.get("ask_recall", 0.0)) for item in holdouts if isinstance(item.get("ask_recall"), (int, float))]
    composition_values = [float(item.get("composition_exact", 0.0)) for item in holdouts if isinstance(item.get("composition_exact"), (int, float))]
    slot_values = [float(item.get("slot_accuracy", 0.0)) for item in holdouts if isinstance(item.get("slot_accuracy"), (int, float))]
    repair_values = [float(item.get("repair_recall", 0.0)) for item in holdouts if isinstance(item.get("repair_recall"), (int, float))]
    entropy_values = [float(item.get("composition_entropy", 0.0)) for item in holdouts if isinstance(item.get("composition_entropy"), (int, float))]
    false_allow_values = [int(item.get("negative_false_allow", 0) or 0) for item in holdouts]
    vocabulary = report.get("train_only_vocabulary") if isinstance(report.get("train_only_vocabulary"), dict) else report.get("train_context_vocabulary") if isinstance(report.get("train_context_vocabulary"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), dict) else {}
    return {
        "label": label,
        "status": str(report.get("status", "unknown")),
        "report_file": path.name,
        "report_sha256": _sha256(path),
        "train_count": int(report.get("train_rows", report.get("train_count", 0)) or 0),
        "holdout_count": int(report.get("holdout_rows", report.get("holdout_count", 0)) or 0),
        "seed_count": len(holdouts),
        "vocabulary_scope": str(vocabulary.get("scope", "unknown")),
        "vocabulary_size": int(vocabulary.get("size", 0) or 0),
        "weakest_head": {"name": weakest_name, "accuracy": round(float(weakest_accuracy), 6)},
        "holdout_composition_exact": round(min(composition_values), 6) if composition_values else 0.0,
        "holdout_slot_accuracy": round(min(slot_values), 6) if slot_values else 0.0,
        "holdout_repair_recall": round(min(repair_values), 6) if repair_values else 0.0,
        "holdout_composition_entropy": round(max(entropy_values), 6) if entropy_values else 0.0,
        "holdout_ask_recall": round(min(ask_values), 6) if ask_values else 0.0,
        "holdout_negative_false_allow": max(false_allow_values) if false_allow_values else 0,
        "execution": {
            "optimizer_started": execution.get("optimizer_started") is True,
            "device": str(execution.get("device", "unknown")),
            "gpu_touched": execution.get("gpu_touched") is True,
        },
        "training_eligible": int(report.get("training_eligible", 0) or 0) > 0,
        "capability_training_allowed": report.get("capability_training_allowed") is True,
        "promotion": _safe_promotion(report.get("promotion")),
    }


def _candidate_model_projection() -> dict[str, Any]:
    runs = [_candidate_run_projection(label, path) for label, path in DEFAULT_CANDIDATE_REPORTS]
    return {
        "status": "candidate_only_projection",
        "runs": runs,
        "latest_label": runs[-1]["label"] if runs else "missing",
        "training_allowed": False,
        "capability_claim_allowed": False,
        "note": "CPU optimizer smoke is wiring evidence only; it is not a vulnerability or payload result.",
    }


def _taxonomy_coverage_projection(path: Path) -> dict[str, Any]:
    """Expose only bounded category counts for the frontend coverage panel."""
    if not path.exists():
        return {
            "status": "missing",
            "file": path.name,
            "sha256": "",
            "case_count": 0,
            "core_case_count": 0,
            "supplemental_case_count": 0,
            "category_count": 0,
            "missing_anchor_count": 0,
            "diagnostic_gap_count": 0,
            "candidate_only_count": 0,
            "categories": [],
        }
    report = _load(path)
    raw_categories = report.get("categories") if isinstance(report.get("categories"), dict) else {}
    categories: list[dict[str, Any]] = []
    for name, raw in raw_categories.items():
        spec = raw if isinstance(raw, dict) else {}
        candidate = spec.get("candidate_only_case_refs")
        unresolved = spec.get("unresolved_next_cases")
        categories.append(
            {
                "name": str(name),
                "covered": int(spec.get("covered_count", 0) or 0),
                "candidate_only": len(candidate) if isinstance(candidate, list) else 0,
                "unresolved": len(unresolved) if isinstance(unresolved, list) else 0,
            }
        )
    categories.sort(key=lambda item: item["name"])
    return {
        "status": str(report.get("status", "unknown")),
        "file": path.name,
        "sha256": _sha256(path),
        "case_count": int(report.get("case_count", 0) or 0),
        "core_case_count": int(report.get("core_case_count", 0) or 0),
        "supplemental_case_count": int(report.get("supplemental_case_count", 0) or 0),
        "category_count": len(categories),
        "missing_anchor_count": int(report.get("missing_anchor_count", 0) or 0),
        "diagnostic_gap_count": int(report.get("diagnostic_gap_count", 0) or 0),
        "candidate_only_count": int(report.get("candidate_only_count", 0) or 0),
        "categories": categories,
    }


def build_summary(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    audit_path: Path = DEFAULT_AUDIT,
    plan_path: Path = DEFAULT_PLAN,
    live_path: Path = DEFAULT_LIVE,
    row_bound_report_path: Path = DEFAULT_ROW_BOUND_REPORT,
    row_bound_audit_path: Path = DEFAULT_ROW_BOUND_AUDIT,
    holdout_b_report_path: Path = DEFAULT_HOLDOUT_B_REPORT,
    holdout_b_audit_path: Path = DEFAULT_HOLDOUT_B_AUDIT,
    holdout_b_docker_smoke_path: Path = DEFAULT_HOLDOUT_B_DOCKER_SMOKE,
    cross_implementation_audit_path: Path = DEFAULT_CROSS_IMPLEMENTATION_AUDIT,
    taxonomy_audit_path: Path = DEFAULT_TAXONOMY_AUDIT,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    audit = _load(audit_path)
    plan = _load(plan_path)
    live = _load(live_path)
    row_bound = _load(row_bound_report_path) if row_bound_report_path.exists() else None
    row_bound_audit = _load(row_bound_audit_path) if row_bound_audit_path.exists() else None
    holdout_b = _load(holdout_b_report_path) if holdout_b_report_path.exists() else None
    holdout_b_audit = _load(holdout_b_audit_path) if holdout_b_audit_path.exists() else None
    holdout_b_docker = _load(holdout_b_docker_smoke_path) if holdout_b_docker_smoke_path.exists() else None
    cross_implementation = _load(cross_implementation_audit_path) if cross_implementation_audit_path.exists() else None
    taxonomy_coverage = _taxonomy_coverage_projection(taxonomy_audit_path)
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
    holdout_b_counts = holdout_b.get("counts") if isinstance(holdout_b, dict) and isinstance(holdout_b.get("counts"), dict) else {}
    holdout_b_contract = holdout_b.get("source_contract") if isinstance(holdout_b, dict) and isinstance(holdout_b.get("source_contract"), dict) else {}
    holdout_b_execution = holdout_b.get("execution") if isinstance(holdout_b, dict) and isinstance(holdout_b.get("execution"), dict) else {}
    docker_observed = holdout_b_docker.get("observed") if isinstance(holdout_b_docker, dict) and isinstance(holdout_b_docker.get("observed"), dict) else {}
    docker_execution = holdout_b_docker.get("execution") if isinstance(holdout_b_docker, dict) and isinstance(holdout_b_docker.get("execution"), dict) else {}
    independent_holdout = {
        "implementation": "pg388-logic-lab-backend-b",
        "source_rows": {
            "status": str((holdout_b or {}).get("status", "missing")),
            "source_rows": int(holdout_b_counts.get("source_rows", 0)),
            "strict_valid": int(holdout_b_counts.get("strict_valid", 0)),
            "typed": int(holdout_b_counts.get("typed", 0)),
            "fresh_resets": int(holdout_b_counts.get("fresh_resets", 0)),
            "failure_repair": int(holdout_b_counts.get("failure_repair", 0)),
            "negative_violations": int(holdout_b_counts.get("negative_violations", 0)),
            "audit_status": str((holdout_b_audit or {}).get("status", "missing")),
            "report_file": holdout_b_report_path.name,
            "report_sha256": _sha256(holdout_b_report_path) if holdout_b_report_path.exists() else "",
            "audit_file": holdout_b_audit_path.name,
            "audit_sha256": _sha256(holdout_b_audit_path) if holdout_b_audit_path.exists() else "",
        },
        "docker_smoke": {
            "status": str((holdout_b_docker or {}).get("status", "missing")),
            "health_http_status": int(docker_observed.get("health_http_status", 0)),
            "case_count": int(docker_observed.get("case_count", 0)),
            "role_count": len(docker_observed.get("roles", [])) if isinstance(docker_observed.get("roles"), list) else 0,
            "fresh_before": int(docker_observed.get("fresh_reset_before_count", 0)),
            "fresh_after": int(docker_observed.get("fresh_reset_after_count", 0)),
            "candidate_state_delta": str(docker_observed.get("candidate_state_delta", "missing")),
            "reference_state_delta": str(docker_observed.get("reference_state_delta", "missing")),
            "negative_state_delta": str(docker_observed.get("negative_state_delta", "missing")),
            "negative_control_clean": docker_observed.get("negative_control_clean") is True,
            "docker_started": docker_execution.get("docker_started") is True,
            "target_contacted": docker_execution.get("target_contacted") is True,
            "external_network": docker_execution.get("network_contacted") is True,
            "persistent_storage": docker_observed.get("persistent_storage") is True,
            "safe_to_send": docker_observed.get("safe_to_send") is True,
            "report_file": holdout_b_docker_smoke_path.name,
            "report_sha256": _sha256(holdout_b_docker_smoke_path) if holdout_b_docker_smoke_path.exists() else "",
        },
        "gates": {
            "source_row_contract": holdout_b_contract.get("row_bound_typed_evidence") is True,
            "fresh_role_reset": holdout_b_contract.get("fresh_role_reset_attested") is True,
            "candidate_reference_negative_replay": holdout_b_contract.get("candidate_reference_negative_replay") is True,
            "image_attested": holdout_b_contract.get("image_attested") is True,
            "operator_reviewed": holdout_b_contract.get("operator_reviewed") is True,
            "training_eligible": int((holdout_b or {}).get("training_eligible", 0) or 0) > 0,
            "docker_smoke_observed": docker_execution.get("docker_started") is True and int(docker_observed.get("health_http_status", 0)) == 200,
        },
    }
    cross_sources = cross_implementation.get("sources") if isinstance(cross_implementation, dict) and isinstance(cross_implementation.get("sources"), dict) else {}
    cross_coverage = cross_implementation.get("coverage") if isinstance(cross_implementation, dict) and isinstance(cross_implementation.get("coverage"), dict) else {}
    cross_gate = cross_implementation.get("hard_gate") if isinstance(cross_implementation, dict) and isinstance(cross_implementation.get("hard_gate"), dict) else {}
    cross_implementation_projection = {
        "status": str((cross_implementation or {}).get("status", "missing")),
        "implementation_count": int(cross_sources.get("implementation_count", 0)),
        "source_row_count": int(cross_coverage.get("source_row_count", 0)),
        "split_counts": cross_sources.get("split_counts", {}) if isinstance(cross_sources.get("split_counts", {}), dict) else {},
        "strict_valid": int(cross_coverage.get("strict_valid", 0)),
        "typed_evidence": int(cross_coverage.get("typed_evidence", 0)),
        "fresh_resets": int(cross_coverage.get("fresh_resets", 0)),
        "negative_violations": int(cross_coverage.get("negative_violations", 0)),
        "context_signature_overlap": int(cross_sources.get("cross_implementation_context_signature_overlap", 0)),
        "target_signature_overlap": int(cross_sources.get("cross_implementation_target_signature_overlap", 0)),
        "train_split_present": cross_gate.get("train_split_present") is True,
        "training_eligible": int(cross_gate.get("training_eligible", 0) or 0),
        "failures": [str(item) for item in ((cross_implementation or {}).get("failures", []) if isinstance((cross_implementation or {}).get("failures", []), list) else [])],
        "report_file": cross_implementation_audit_path.name,
        "report_sha256": _sha256(cross_implementation_audit_path) if cross_implementation_audit_path.exists() else "",
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
        "independent_holdout": independent_holdout,
        "cross_implementation_audit": cross_implementation_projection,
        "taxonomy_coverage": taxonomy_coverage,
        "candidate_model": _candidate_model_projection(),
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
            "independent_holdout_docker_smoke": independent_holdout["gates"]["docker_smoke_observed"],
            "cross_implementation_audit": cross_implementation_projection["status"] == "passed_candidate_cross_implementation_logic_audit",
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
            "independent_holdout_context": "bounded counts and status only; implementation-B source rows and Docker evaluator details remain out of browser context",
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
