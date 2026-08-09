import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg42_catalog_has_real_get_post_fresh_typed_replay():
    catalog = _load("pg42_independent_semantic_catalog_v1.json")
    assert catalog["runtime_replay"] is True
    assert catalog["independent_target_implementation"] is True
    assert catalog["methods"] == ["GET", "POST"]
    assert catalog["typed_positive_count"] == 324
    assert catalog["negative_control_count"] == 2556
    assert catalog["fresh_reset_count"] == 2880
    assert catalog["source_count"] == 2
    assert catalog["target_instance_count"] == 2880
    assert catalog["trace_episode_count"] == 60
    assert catalog["accepted_evaluation_episode_count"] == 54
    assert catalog["raw_probe_strings_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert {row["implementation"] for row in catalog["samples"]} == {"cobalt", "quartz"}
    assert {row["surface_variant"] for row in catalog["samples"]} == {"ledger", "envelope", "framed"}
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in catalog["samples"])
    assert len({row["evidence"]["evidence_hash"] for row in catalog["samples"]}) == len(catalog["samples"])
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized


def test_pg42_trace_contains_both_methods_and_replay_safety():
    trace = _load("pg42_independent_semantic_trace_v1.json")
    assert trace["methods"] == ["GET", "POST"]
    assert trace["episode_count"] == 60
    assert trace["accepted_evaluation_episode_count"] == 54
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert all(step["online_weight_update"] is False and step["long_term_memory_write"] is False for step in trace["steps"])
    assert all(step["fresh_reset"]["fresh_target"] for step in trace["steps"])


def test_pg42_independent_ood_fails_capability_gate_without_false_accepts():
    report = _load("pg42_independent_ood_evaluation_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["splits"]["train"]["effect_recall_any_family"] == 1.0
    assert report["splits"]["implementation_holdout"]["effect_recall_any_family"] == 0.666667
    assert report["splits"]["family_holdout"]["effect_recall_any_family"] == 0.666667
    assert report["routing"]["metrics"]["negative_effect_false_accept_count"] == 0
    assert report["routing"]["metrics"]["unknown_misname_count"] == 0
    assert report["routing"]["metrics"]["unknown_strict_abstain"] is True
    assert report["safe_routing_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg42_protocol_records_failure_as_experiment_evidence():
    protocol = _load("pg42_independent_semantic_ood_protocol_v1.json")
    assert protocol["sample_contract"]["fresh_reset_per_sample"] is True
    assert protocol["sample_contract"]["planned_candidate_control_pairs"] == 1440
    assert protocol["collection_result"]["get_post_covered"] is True
    assert protocol["run_result"]["negative_effect_false_accept_count"] == 0
    assert protocol["run_result"]["safe_routing_gate_status"] == "blocked"
    assert protocol["run_result"]["capability_claim_allowed"] is False
    assert protocol["run_result"]["failure_classification"].startswith("experiment_")
    assert protocol["status"] == "run_completed_independent_generalization_gate_failed_no_promotion"
