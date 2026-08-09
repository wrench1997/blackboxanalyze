import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg110_capacity_pressure_gate_passes_without_weight_mutation():
    report = _load("pg110_capacity_pressure_cycle_report_v1.json")
    assert report["status"] == "passed_capacity_pressure_diagnostic"
    assert report["capability_gate"]["claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["controller"]["neural_weight_mutation_performed"] is False
    assert report["controller"]["full_training_started"] is False
    assert report["controller"]["architecture_transfer_mode"] == "bsp_v3_structure_contract_only"
    assert report["controller"]["previous_checkpoint_reuse_forbidden"] is True
    assert report["controller"]["fresh_checkpoint_required"] is True
    assert report["controller"]["mandarin_foundation_training_isolated"] is True
    metrics = report["metrics"]
    assert metrics["scenario_count"] == 8
    assert metrics["growth_action_count"] == 2
    assert metrics["merge_ablate_action_count"] == 1
    assert metrics["ablation_pass"] is True
    assert metrics["ablation_rollback"] is True


def test_pg110_trace_is_bounded_and_records_tradeoff_actions():
    dataset = _load("pg110_capacity_pressure_cycle_visible_dataset_v1.json")
    trace = _load("pg110_capacity_pressure_cycle_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["evaluator_labels_in_trace"] is False
    assert len(trace["steps"]) == 8
    actions = {step["scenario_id"]: step["action"] for step in trace["steps"]}
    assert actions["typed_gap_growth"] == "wake_target_unit"
    assert actions["all_abstain_typed_gap"] == "wake_target_unit"
    assert actions["speed_redundancy"] == "merge_then_ablate_low_contribution_units"
    assert actions["speed_without_redundancy"] == "measure_speed_without_ablation"
    assert actions["speed_and_capability_gap"] == "hold_and_measure_tradeoff"
    assert actions["false_accept_safety"] == "hold_and_repair_evidence"
    assert all(step["executable"] is False and step["promotion_eligible"] is False for step in trace["steps"])
    text = json.dumps({"dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("xss", "sql_injection", "workflow_invariant", "<script", "union select"):
        assert forbidden not in text


def test_pg110_source_hashes_match_current_implementation():
    report = _load("pg110_capacity_pressure_cycle_report_v1.json")
    paths = {
        "pg108_report": "research/pg108_belief_stress_report_v1.json",
        "pg109_report": "research/pg109_fragment_composition_report_v1.json",
        "controller": "app/bsp_capacity_controller.py",
        "runner": "scripts/run_pg110_capacity_pressure_cycle.py",
    }
    for name, relative_path in paths.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"]["source_hashes"][name]


def test_pg110_capacity_policy_is_explicit_about_speed_and_conflict():
    rules = _load("improvement_rules.json")
    policy = rules["bsp_capacity_pressure_policy"]
    assert policy["capacity_actions"]["speed_pressure"] == "measure_fixed_shape_latency_then_merge_and_ablate_low_contribution_units"
    assert policy["capacity_actions"]["capability_and_speed_conflict"] == "hold_and_measure_capacity_gain_vs_latency_tradeoff_before_mutation"
    assert policy["ablation_gate"]["pre_post_fixed_holdout_required"] is True
    assert policy["ablation_gate"]["rollback_on_regression"] is True
    assert policy["architecture_only_transfer"] is True
    assert policy["previous_weights_reuse_forbidden"] is True
    assert policy["fresh_bsp_checkpoint_required"] is True
    assert policy["mandarin_foundation_stage_isolated"] is True
