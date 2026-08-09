from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def _load_audit_module():
    path = ROOT / "scripts" / "audit_pg326_cross_impl_forgetting_matrix.py"
    spec = importlib.util.spec_from_file_location("pg326_matrix_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pg326_matrix_aggregates_observed_behavior_and_records_paired_forgetting():
    report = _load_json("pg326_cross_impl_forgetting_matrix_v1.json")
    protocol = _load_json("pg326_cross_impl_forgetting_matrix_protocol_v1.json")
    assert report["status"] == "completed_read_only_matrix_blocked"
    assert len(report["implementation_digests"]) == 3
    assert report["families"] == ["authentication", "sql", "xss"]
    assert report["totals"] == {
        "seed_count": 9,
        "route_count": 45,
        "get_count": 27,
        "post_count": 18,
        "positive_typed_effect_count": 18,
        "positive_route_count": 18,
        "variant_exact_count": 135,
        "variant_role_count": 135,
        "multi_missing_question_rows": 675,
        "failure_repair_correct_count": 45,
        "failure_repair_count": 45,
        "negative_lane_violation_count": 0,
    }
    assert report["worst_seed_metrics"]["typed_effect_rate_min"] == 1.0
    assert report["worst_seed_metrics"]["variant_exact_rate_min"] == 1.0
    assert report["worst_seed_metrics"]["ask_recall_min"] == 1.0
    assert report["worst_seed_metrics"]["repair_rate_min"] == 1.0
    assert report["uniform_checks"]["context_firewall"] is True
    assert report["uniform_checks"]["raw_payload_excluded"] is True
    assert report["uniform_checks"]["raw_response_excluded"] is True
    assert report["uniform_checks"]["failure_action_changed"] is False
    assert report["uniform_checks"]["role_bound_belief_evidence"] is False
    assert report["forgetting"]["paired_replay_present"] is True
    assert report["forgetting"]["same_canary_route_set"] is True
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["hypothesis_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert protocol["required_gates"]["forgetting_pair"] is True


def test_pg326_read_only_audit_passes_without_target_contact():
    result = _load_audit_module().audit()
    assert result["status"] == "passed"
    assert result["promotion_allowed"] is False
    assert result["target_contacted"] is False
