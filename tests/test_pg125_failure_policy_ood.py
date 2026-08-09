from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg125_scope_family_failure_is_preserved_as_evaluation_only() -> None:
    report = _load("pg125_failure_policy_ood_report_v1.json")
    assert report["status"] == "completed_pg125_frozen_failure_policy_ood"
    assert report["collection"]["targets"] == 9
    assert report["collection"]["steps"] == 144
    assert report["collection"]["get"] == report["collection"]["post"] == 72
    assert report["full_failure_input"]["accuracy"] == 0.9375
    assert report["full_failure_input"]["safety_compliance_rate"] == 0.9375
    assert report["full_failure_input"]["per_surface"]["scope"]["accuracy"] == 0.75
    assert report["full_model_failure_zeroed"]["accuracy"] == 0.472222
    assert report["checks"]["all_surface_accuracy_floor"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg125_ood_trace_has_no_training_or_raw_payloads() -> None:
    trace = _load("pg125_failure_policy_ood_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["failure_signatures_visible"] is True
    assert all(
        step.get("failure_signature")
        for target in trace["targets"]
        for episode in target["episodes"]
        for step in episode["steps"]
    )
