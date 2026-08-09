from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def _load_module(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pg325_sql_family_holdout_has_real_get_post_evidence_but_stays_blocked() -> None:
    report = _load("pg325_sql_family_holdout_report_v1.json")
    catalog = _load("pg325_sql_family_holdout_catalog_v1.json")
    trace = _load("pg325_sql_family_holdout_trace_v1.json")
    protocol = _load("pg325_sql_family_holdout_protocol_v1.json")
    assert report["status"] == "completed_real_local_docker_pg325_sql_family_holdout"
    assert report["counts"] == {
        "seed_count": 3,
        "route_count": 9,
        "get_count": 6,
        "post_count": 3,
        "positive_route_count": 9,
        "positive_typed_effect_count": 9,
        "variant_role_count": 27,
        "variant_exact_count": 27,
        "model_send_count": 27,
        "negative_lane_violation_count": 0,
        "failure_repair_correct_count": 9,
        "failure_repair_count": 9,
        "failure_transition_required_count": 9,
        "failure_action_changed_count": 9,
        "multi_missing_question_rows": 135,
        "multi_missing_unsafe_allow": 0,
        "belief_transition_count": 27,
        "belief_duplicate_evidence_count": 0,
    }
    worst = report["worst_seed_metrics"]
    assert worst["multi_missing_question_recall_min"] == 1.0
    assert worst["variant_exact_min"] == 1.0
    assert worst["failure_repair_rate_min"] == 1.0
    assert worst["failure_action_changed_rate_min"] == 1.0
    assert worst["positive_typed_effect_route_rate_min"] == 1.0
    assert worst["negative_lane_violation_max"] == 0
    assert report["checks"]["docker_network_none"] is True
    assert report["checks"]["database_health_per_route"] is True
    assert report["checks"]["source_attestation_per_route"] is True
    assert report["checks"]["typed_evidence_hash_per_route"] is True
    assert report["checks"]["model_context_firewall"] is True
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["hypothesis_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert len(catalog["entries"]) == 9
    assert all(row["oracle"]["typed_effect_confirmed"] for row in catalog["entries"])
    assert all(row["oracle"]["evidence_sha256"] for row in catalog["entries"])
    assert all(row["training_eligible"] is False for row in catalog["entries"])
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert protocol["scope"]["network"] == "none"
    assert protocol["required_gates"]["typed_sql_effect"] is True


def test_pg325_independent_artifact_audit_is_read_only_and_passes() -> None:
    module = _load_module("pg325_audit_test", "audit_pg325_sql_family_holdout.py")
    result = module.audit()
    assert result["status"] == "passed"
    assert result["promotion_allowed"] is False
    assert result["target_contacted"] is False
