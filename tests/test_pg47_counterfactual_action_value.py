import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg47_action_value_head_generalizes_active_selection_to_pg42():
    report = _load("pg47_counterfactual_action_value_report_v1.json")
    holdout = report["pg42_holdout"]
    assert report["status"] == "diagnostic_only"
    assert report["training"]["oracle_is_target_only"] is True
    assert report["training"]["pg42_used_for_training"] is False
    assert report["train_metrics"]["effect_recall"] == 1.0
    assert report["seed_holdout_metrics"]["effect_recall"] == 1.0
    assert holdout["episode_count"] == 180
    assert holdout["effect_success_rate"] == 1.0
    assert holdout["known_family_recall"] == 1.0
    assert holdout["unknown_positive_count"] == 72
    assert holdout["unknown_safe_abstain_count"] == 72
    assert holdout["unknown_strict_abstain"] is True
    assert holdout["negative_false_accept_count"] == 0
    assert holdout["mean_queries"] == 3.1
    assert holdout["median_queries"] == 3.0
    assert holdout["get_post_covered"] is True
    assert holdout["accepted_trace_episode_count"] == 180
    assert report["safe_gate"]["claim_allowed"] is True
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg47_action_trace_is_fresh_projection_only():
    trace = _load("pg47_counterfactual_action_trace_v1.json")
    assert trace["episode_count"] == 180
    assert trace["accepted_evaluation_episode_count"] == 180
    assert trace["methods"] == ["GET", "POST"]
    assert len(trace["steps"]) == 558
    assert all(step["fresh_reset"]["fresh_target"] for step in trace["steps"])
    assert all(step["online_weight_update"] is False and step["long_term_memory_write"] is False for step in trace["steps"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False


def test_pg47_protocol_separates_action_gain_from_formal_promotion():
    protocol = _load("pg47_counterfactual_action_replay_protocol_v1.json")
    assert protocol["training_contract"]["oracle_is_target_only"] is True
    assert protocol["training_contract"]["pg42_used_for_training"] is False
    assert protocol["policy_contract"]["get_post_exploration_required"] is True
    assert protocol["run_result"]["safe_gate_status"] == "passed"
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["status"] == "run_completed_action_value_safety_gate_passed_no_promotion"
