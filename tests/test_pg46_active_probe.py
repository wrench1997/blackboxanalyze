import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg46_active_policy_finds_effect_with_get_post_belief_loop():
    report = _load("pg46_active_probe_report_v1.json")
    metrics = report["metrics"]
    assert report["status"] == "diagnostic_only"
    assert metrics["episode_count"] == 180
    assert metrics["positive_episode_count"] == 162
    assert metrics["negative_episode_count"] == 18
    assert metrics["effect_success_rate"] == 1.0
    assert metrics["known_family_recall"] == 1.0
    assert metrics["unknown_positive_count"] == 72
    assert metrics["unknown_safe_abstain_count"] == 72
    assert metrics["unknown_strict_abstain"] is True
    assert metrics["negative_false_accept_count"] == 0
    assert metrics["mean_queries"] == 3.1
    assert metrics["median_queries"] == 3.0
    assert metrics["mean_query_reduction_rate"] == 0.225
    assert metrics["belief_update_count"] == 558
    assert metrics["get_post_covered"] is True
    assert metrics["accepted_trace_episode_count"] == 180
    assert report["safe_gate"]["claim_allowed"] is True
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg46_trace_is_projection_only_and_each_episode_is_validated():
    trace = _load("pg46_active_probe_trace_v1.json")
    assert trace["episode_count"] == 180
    assert trace["accepted_evaluation_episode_count"] == 180
    assert trace["methods"] == ["GET", "POST"]
    assert len(trace["steps"]) == 558
    assert all(step["action_manifest"]["method"] in {"GET", "POST"} for step in trace["steps"])
    assert all(step["fresh_reset"]["fresh_target"] and step["fresh_reset"]["completed"] for step in trace["steps"])
    assert all(step["belief_before"] and step["belief_after"] for step in trace["steps"])
    assert all(step["online_weight_update"] is False and step["long_term_memory_write"] is False for step in trace["steps"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False


def test_pg46_protocol_keeps_query_reduction_separate_from_promotion():
    protocol = _load("pg46_active_probe_protocol_v1.json")
    assert protocol["input_contract"]["oracle_is_evaluation_only"] is True
    assert protocol["policy_contract"]["both_methods_required"] is True
    assert protocol["evaluation"]["fixed_probe_baseline_queries"] == 4
    assert protocol["run_result"]["safe_gate_status"] == "passed"
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["status"] == "run_completed_active_safety_gate_passed_no_promotion"
