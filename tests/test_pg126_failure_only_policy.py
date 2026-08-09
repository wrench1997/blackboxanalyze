from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg126_failure_only_policy_repairs_pg125_scope_generalization() -> None:
    report = _load("pg126_failure_only_policy_report_v1.json")
    assert report["status"] == "completed_pg126_failure_only_policy"
    assert report["scope"]["feature_dim"] == 17
    assert report["holdout"]["full_failure_only"]["metrics"]["accuracy"] == 1.0
    assert report["holdout"]["full_failure_only"]["safety_compliance_rate"] == 1.0
    assert report["holdout"]["full_failure_only"]["per_surface"]["scope"]["accuracy"] == 1.0
    assert report["holdout"]["full_model_failure_zeroed"]["metrics"]["accuracy"] == 0.1875
    assert report["holdout"]["fresh_zero_input_baseline"]["metrics"]["accuracy"] == 0.5
    assert all(report["checks"].values())
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg126_uses_pg125_only_as_holdout() -> None:
    trace = _load("pg126_failure_only_policy_trace_v1.json")
    assert trace["training_source"] == "pg122_failure_guided_train_dev"
    assert trace["holdout_source"] == "pg125_scope_logic_ood"
    assert trace["get_holdout_count"] == trace["post_holdout_count"] == 72
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["memory_promotion_allowed"] is False
