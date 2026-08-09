import hashlib

import pytest

from app.bsp_capacity_controller import evaluate_ablation_gate, plan_capacity_action, validate_pressure_observation


def _unit(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observation(**overrides):
    value = {
        "target_id": "capacity-test-target",
        "unit_kind": "bsp_node",
        "capacity_units": 8,
        "typed_bottleneck": False,
        "fresh_holdout_gap": False,
        "cross_dataset_evidence": True,
        "cross_seed_evidence": True,
        "cross_implementation_evidence": True,
        "known_recall": 1.0,
        "false_accept_count": 0,
        "unknown_abstain_rate": 1.0,
        "all_abstain": False,
        "latency_ms": 100.0,
        "latency_budget_ms": 150.0,
        "memory_ratio": 1.0,
        "redundancy_evidence": False,
        "low_contribution_units": [],
    }
    value.update(overrides)
    return value


def test_typed_gap_wakes_one_target_unit_but_is_not_executable():
    plan = plan_capacity_action(_observation(typed_bottleneck=True, fresh_holdout_gap=True, known_recall=.55))
    assert plan["action"] == "wake_target_unit"
    assert plan["capacity_before"] == 8
    assert plan["capacity_after_proposed"] == 9
    assert plan["executable"] is False
    assert plan["promotion_eligible"] is False


def test_speed_pressure_requires_redundancy_before_ablation():
    no_proof = plan_capacity_action(_observation(latency_ms=250, latency_budget_ms=150))
    assert no_proof["action"] == "measure_speed_without_ablation"
    with_proof = plan_capacity_action(_observation(latency_ms=250, latency_budget_ms=150, redundancy_evidence=True, low_contribution_units=[_unit("u0")]))
    assert with_proof["action"] == "merge_then_ablate_low_contribution_units"
    assert with_proof["capacity_after_proposed"] == 7


def test_capacity_and_speed_conflict_is_measured_before_mutation():
    plan = plan_capacity_action(_observation(typed_bottleneck=True, fresh_holdout_gap=True, known_recall=.6, latency_ms=250, latency_budget_ms=150))
    assert plan["action"] == "hold_and_measure_tradeoff"
    assert plan["capacity_after_proposed"] == plan["capacity_before"]


def test_ablation_gate_passes_only_with_fresh_holdout_and_no_capability_regression():
    baseline = {"known_recall": 1.0, "false_accept_count": 0, "unknown_abstain_rate": 1.0, "latency_ms": 250, "memory_ratio": 1.0, "fresh_holdout_tested": True, "capacity_units": 8}
    candidate = {"known_recall": 1.0, "false_accept_count": 0, "unknown_abstain_rate": 1.0, "latency_ms": 150, "memory_ratio": .9, "fresh_holdout_tested": True, "capacity_units": 7}
    result = evaluate_ablation_gate(baseline, candidate, action="merge_then_ablate_low_contribution_units")
    assert result["status"] == "passed_ablation_gate"
    assert result["rollback"] is False
    bad = dict(candidate)
    bad["known_recall"] = .9
    result = evaluate_ablation_gate(baseline, bad, action="merge_then_ablate_low_contribution_units")
    assert result["status"] == "rollback"
    assert result["rollback"] is True
    assert "known_recall_regressed" in result["reasons"]


def test_capacity_controller_rejects_evaluator_fields():
    with pytest.raises(ValueError, match="evaluator or raw field"):
        validate_pressure_observation(_observation(family="xss"))
