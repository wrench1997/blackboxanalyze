import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg108_order_seed_duplicate_and_budget_gates_pass_without_promotion():
    report = _load("pg108_belief_stress_report_v1.json")
    assert report["status"] == "passed_belief_order_seed_stress"
    assert report["capability_gate"]["claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    metrics = report["metrics"]
    assert metrics["episode_count"] == 289
    assert metrics["scenario_count"] == 7
    assert metrics["order_invariant"] is True
    assert metrics["seed_invariant"] is True
    assert metrics["duplicate_step_count"] == 289
    assert metrics["conflicting_duplicate_step_count"] == 289
    assert metrics["duplicate_posterior_unchanged_rate"] == 1.0
    assert metrics["conflicting_posterior_unchanged_rate"] == 1.0
    assert metrics["budget_one_fail_closed_rate"] == 1.0
    assert metrics["canonical_confirmed_positive_count"] == 0
    assert metrics["canonical_typed_oracle_called_count"] == 0
    assert metrics["posterior_states"] == ["effect", "input_only", "no_effect", "unknown"]
    assert report["controller"]["posterior_family_free"] is True
    assert report["controller"]["duplicate_evidence_guard"] is True


def test_pg108_trace_separates_real_duplicate_steps_and_keeps_bounded_evidence():
    dataset = _load("pg108_belief_stress_visible_dataset_v1.json")
    trace = _load("pg108_belief_stress_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["evaluator_labels_in_trace"] is False
    assert len(trace["steps"]) == 4335
    counts = Counter(step["scenario"] for step in trace["steps"])
    assert counts == Counter({"canonical": 1734, "reverse": 578, "seed_shuffle": 1156, "duplicate": 289, "conflicting": 289, "budget_one": 289})
    duplicate_steps = [step for step in trace["steps"] if step["scenario"] == "duplicate"]
    conflicting_steps = [step for step in trace["steps"] if step["scenario"] == "conflicting"]
    assert len(duplicate_steps) == len(conflicting_steps) == 289
    for step in duplicate_steps + conflicting_steps:
        belief = step["belief_step"]
        assert belief["duplicate_evidence"] is True
        assert belief["accepted"] is False
        assert belief["information_gain"] == 0.0
    assert all(step["typed_oracle_called"] is False and step["confirmed_positive"] is False for step in trace["steps"])
    assert all(step["negative_control_matched"] and step["fresh_reset"]["fresh_target"] for step in trace["steps"])
    assert all(len(step["evidence_sha256"]) == 64 and all(char in "0123456789abcdef" for char in step["evidence_sha256"]) for step in trace["steps"])
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["long_term_memory_write"] is False


def test_pg108_source_hashes_match_the_frozen_inputs_and_runner():
    report = _load("pg108_belief_stress_report_v1.json")
    paths = {
        "train": "research/pg101_active_probe_signature_visible_dataset_v1.json",
        "pg105_dataset": "research/pg105_observable_projection_visible_dataset_v1.json",
        "pg105_trace": "research/pg105_observable_projection_trace_v1.json",
        "pg106_dataset": "research/pg106_decoy_projection_holdout_visible_dataset_v1.json",
        "pg106_trace": "research/pg106_decoy_projection_holdout_trace_v1.json",
        "belief": "app/generic_belief_state.py",
        "inducer": "app/active_goal_label_inducer.py",
        "runner": "scripts/run_pg108_belief_stress.py",
    }
    for name, relative_path in paths.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"]["source_hashes"][name]
