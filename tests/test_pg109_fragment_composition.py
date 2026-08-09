import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg109_fragment_composition_gate_passes_without_promotion():
    report = _load("pg109_fragment_composition_report_v1.json")
    assert report["status"] == "passed_fragment_composition_diagnostic"
    assert report["capability_gate"]["claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    metrics = report["metrics"]
    assert metrics["training_row_count"] == 32
    assert metrics["evaluation_row_count"] == 578
    assert metrics["episode_count"] == 289
    assert metrics["known_effect_expected_pair_count"] == 216
    assert metrics["known_effect_assembled_pair_count"] == 216
    assert metrics["known_effect_assembly_recall"] == 1.0
    assert metrics["unknown_or_decoy_abstain_rate"] == 1.0
    assert metrics["cross_sample_recombination_count"] == 8
    assert metrics["cross_sample_valid_rate"] == 1.0
    assert metrics["negative_case_abstain_rate"] == 1.0
    assert metrics["typed_oracle_called_count"] == 0
    assert metrics["confirmed_positive_count"] == 0
    assert report["controller"]["family_labels_in_fragments"] is False
    assert report["controller"]["copy_paste_order_invariant"] is True


def test_pg109_visible_trace_contains_only_bounded_fragments_and_no_family_labels():
    dataset = _load("pg109_fragment_composition_visible_dataset_v1.json")
    trace = _load("pg109_fragment_composition_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["evaluator_labels_in_trace"] is False
    assert len(dataset["episodes"]) == 289
    assert len(trace["steps"]) == 578
    assert all(episode["typed_oracle_called"] is False and episode["confirmed_positive"] is False for episode in dataset["episodes"])
    assert all(step["typed_oracle_called"] is False and step["confirmed_positive"] is False for step in trace["steps"])
    assert all(step["fresh_reset"] is True and step["negative_control_matched"] is True for step in trace["steps"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", step["evidence_sha256"]) for step in trace["steps"])
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    text = json.dumps({"dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("xss", "sql_injection", "workflow_invariant", "union select", "<script", "onerror"):
        assert forbidden not in text
    for episode in dataset["episodes"]:
        for fragment in episode["fragments"]:
            assert set(fragment["atoms"]).issubset({"effect_present", "input_only_anomaly", "no_effect", "ambiguous_effect", "probe_binding_valid", "negative_control_clear"})
            assert re.fullmatch(r"[0-9a-f]{64}", fragment["fragment_id"])
            assert re.fullmatch(r"[0-9a-f]{64}", fragment["evidence_sha256"])
            assert re.fullmatch(r"[0-9a-f]{64}", fragment["reset_sha256"])


def test_pg109_report_source_hashes_match_current_implementation():
    report = _load("pg109_fragment_composition_report_v1.json")
    paths = {
        "train_dataset": "research/pg101_active_probe_signature_visible_dataset_v1.json",
        "pg105_dataset": "research/pg105_observable_projection_visible_dataset_v1.json",
        "pg105_trace": "research/pg105_observable_projection_trace_v1.json",
        "pg106_dataset": "research/pg106_decoy_projection_holdout_visible_dataset_v1.json",
        "pg106_trace": "research/pg106_decoy_projection_holdout_trace_v1.json",
        "assembler": "app/rule_fragment_assembler.py",
        "inducer": "app/active_goal_label_inducer.py",
        "runner": "scripts/run_pg109_fragment_composition.py",
    }
    for name, relative_path in paths.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"]["source_hashes"][name]


def test_bsp_capacity_policy_requires_typed_growth_and_measured_ablation():
    rules = _load("improvement_rules.json")
    policy = rules["bsp_capacity_pressure_policy"]
    assert policy["growth_gate"]["typed_bottleneck_required"] is True
    assert policy["growth_gate"]["fresh_holdout_gap_required"] is True
    assert policy["growth_gate"]["family_label_growth_forbidden"] is True
    assert policy["capacity_actions"]["speed_pressure"] == "measure_fixed_shape_latency_then_merge_and_ablate_low_contribution_units"
    assert policy["ablation_gate"]["pre_post_fixed_holdout_required"] is True
    assert policy["ablation_gate"]["known_recall_must_not_drop"] is True
    assert policy["ablation_gate"]["false_accept_must_not_increase"] is True
    assert policy["ablation_gate"]["rollback_on_regression"] is True
    assert policy["training_promotion_allowed"] is False
