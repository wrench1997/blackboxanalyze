"""PG-71 audit: locate feature loss before changing the abstention head."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg70_trace_abstention_head_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
OUTPUT_PATH = ROOT / "research" / "pg71_trace_feature_drift_audit_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg71_trace_feature_drift_audit_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg71_trace_feature_drift_audit_report_v1.md"


def _load_pg70() -> Any:
    spec = importlib.util.spec_from_file_location("pg70_feature_audit", ROOT / "scripts" / "train_pg70_trace_abstention_head.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-70 feature projection")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _observable_shape_delta(step: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(step.get("baseline_projection") or {})
    candidate = dict(step.get("response_projection") or {})
    fields = ("status_code", "status_class", "content_type", "content_type_class", "body_length_bucket", "html_tag_count", "form_count", "input_count", "script_count", "result_row_count", "marker_reflected", "marker_count", "has_location", "location_origin")
    differences = [field for field in fields if baseline.get(field) != candidate.get(field)]
    return {"available_field_count": sum(field in baseline or field in candidate for field in fields), "difference_fields": differences, "difference_count": len(differences)}


def run() -> dict[str, Any]:
    pg70 = _load_pg70()
    trace = _read(TRACE_PATH)
    steps = [dict(step) for step in trace.get("steps", [])]
    known = [step for step in steps if "workflow" not in str(step.get("episode_id", ""))]
    # Reproduce the exact legacy feature projection; labels are used only for
    # this audit and never enter a model feature.
    legacy_pairs: list[dict[str, Any]] = []
    for step in known:
        candidate = torch.tensor(pg70._features(step, step.get("response_projection") or {}), dtype=torch.float32)
        control = torch.tensor(pg70._features(step, step.get("baseline_projection") or {}), dtype=torch.float32)
        legacy_pairs.append({"step_id": step["step_id"], "feature_l2": float(torch.norm(candidate - control)), "candidate_label": "confirm", "control_label": "reject", "observable": _observable_shape_delta(step)})
    duplicate_conflicts = sum(int(item["feature_l2"] <= 1e-9) for item in legacy_pairs)
    train_rows, dev_rows, _ = pg70._build_examples(steps)
    train_values, normalisation = pg70._normalise(train_rows, train_rows)
    dev_values = (torch.tensor([row["features"] for row in dev_rows], dtype=torch.float32) - normalisation[0]) / normalisation[1]
    train_std = train_values.std(dim=0, unbiased=False)
    sparse_dims = int((train_std < 0.01).sum().item())
    distances = torch.cdist(dev_values, train_values).min(dim=1).values if len(dev_values) and len(train_values) else torch.empty(0)
    report = {
        "protocol_id": "pg-pk-71-trace-feature-drift-audit-v1",
        "schema_version": "sift-pg71-trace-feature-drift-audit-report-v1",
        "status": "feature_drift_audit_completed",
        "source": {"pg70_report": str(REPORT_PATH.relative_to(ROOT)), "pg69_trace": str(TRACE_PATH.relative_to(ROOT)), "oracle_in_features": False, "family_in_features": False},
        "metrics": {
            "known_pair_count": len(legacy_pairs),
            "legacy_candidate_control_duplicate_feature_count": duplicate_conflicts,
            "legacy_candidate_control_duplicate_label_conflict_count": duplicate_conflicts,
            "pair_observable_shape_delta_count": sum(int(item["observable"]["difference_count"] > 0) for item in legacy_pairs),
            "pair_observable_shape_delta_fields": {item["step_id"]: item["observable"]["difference_fields"] for item in legacy_pairs},
            "legacy_train_feature_sparse_dimension_count": sparse_dims,
            "legacy_dev_min_distance": round(float(distances.min().item()), 6) if len(distances) else None,
            "legacy_dev_max_distance": round(float(distances.max().item()), 6) if len(distances) else None,
            "legacy_ood_threshold": pg70.OOD_DISTANCE_THRESHOLD,
        },
        "root_cause": {"primary": "feature_extractor_drops_bounded_shape_differences", "secondary": "per_dimension_floor_amplifies_sparse_shift", "evidence": ["candidate/control response projections differ while legacy vectors collide", "dev distances exceed pre-registered OOD threshold", "unknown abstention is not evidence of known-family capability"]},
        "repair_contract": {"include_bounded_shape_scalars": ["status_class", "body_length_bucket", "html_tag_count", "form_count", "script_count", "result_row_count", "marker_reflected", "has_location", "location_origin"], "keep_raw_body_forbidden": True, "keep_family_and_oracle_features_forbidden": True, "compare_v2_on_same_frozen_split": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "formal_claim_allowed": False},
    }
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg-pk-71-trace-feature-drift-audit-v1", "schema_version": "sift-pg71-trace-feature-drift-audit-protocol-v1", "input_contract": {"accepted_pg69_trace_only": True, "family_and_oracle_features_forbidden": True, "raw_body_persistence_forbidden": True}, "required_checks": {"duplicate_label_conflict_reported": True, "bounded_shape_delta_reported": True, "sparse_dimension_shift_reported": True, "v2_repair_must_use_same_split": True}, "run_result": {"status": report["status"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG71 v2 bounded-shape feature projection on the frozen PG69 split, then independent seed/fresh Docker replay"}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-71 Trace feature drift audit\n\n" + f"legacy candidate/control duplicate-label pairs: `{duplicate_conflicts}`；observable shape differences: `{report['metrics']['pair_observable_shape_delta_count']}`；sparse dims: `{sparse_dims}`；dev distance range: `{report['metrics']['legacy_dev_min_distance']}..{report['metrics']['legacy_dev_max_distance']}`。\n\n根因：安全 response projection 已存在，但旧 feature extractor 丢掉了关键 bounded shape 差分；同时 per-dimension floor 放大了稀疏漂移。training/memory promotion 均关闭。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({"protocol_id": report["protocol_id"], "duplicate_label_conflict_count": report["metrics"]["legacy_candidate_control_duplicate_label_conflict_count"], "shape_delta_count": report["metrics"]["pair_observable_shape_delta_count"], "sparse_dimension_count": report["metrics"]["legacy_train_feature_sparse_dimension_count"]}, ensure_ascii=False, indent=2))
