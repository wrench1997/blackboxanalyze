import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg36_protocol_freezes_delayed_maze_and_unknown_abstain_contract():
    protocol = _load("pg36_delayed_maze_protocol_v1.json")
    assert protocol["scope"]["methods"] == ["GET", "POST"]
    assert protocol["scope"]["phases"] == ["screen", "confirm", "error", "timeout"]
    fixture = protocol["fixture"]
    assert fixture["sample_count"] == 960
    assert fixture["typed_positive_count"] == 96
    assert fixture["accepted_evaluation_episode_count"] == 48
    contract = protocol["maze_contract"]
    assert contract["screen_is_ambiguous"] is True
    assert contract["confirm_is_the_only_positive_phase"] is True
    assert contract["unknown_surface_must_abstain"] is True
    assert contract["step_identity_includes_pair_role"] is True
    assert protocol["promotion_gate"]["zero_negative_false_accept"] is True
    assert protocol["promotion_gate"]["training_and_memory_blocked_on_failure"] is True
    assert protocol["status"] == "capability_gate_failed_experiment_triage_complete_no_promotion"


def test_pg36_improvement_rule_keeps_error_timeout_and_memory_guards():
    rules = _load("improvement_rules.json")
    policy = rules["pg36_delayed_maze_policy"]
    assert policy["required_channels"] == ["GET", "POST"]
    assert policy["required_phases"] == ["screen", "confirm", "error", "timeout"]
    assert policy["typed_oracle_after_probe_only"] is True
    assert policy["unknown_surface_strict_abstain"] is True
    assert policy["no_real_sleep_or_state_mutation"] is True
    assert policy["memory_promotion_on_failure"] is False


def test_pg36_active_belief_diagnostic_is_replay_only_and_not_a_capability_claim():
    protocol = _load("pg36_delayed_maze_protocol_v1.json")
    diagnostic = protocol["active_belief_diagnostic"]
    assert diagnostic["mode"] == "offline_projection_replay"
    assert diagnostic["typed_oracle_used_after_probe_for_stop_only"] is True
    assert diagnostic["controller_is_upper_bound_diagnostic"] is True
    assert diagnostic["capability_claim_allowed"] is False
    assert diagnostic["training_allowed"] is False
    assert diagnostic["memory_promotion_allowed"] is False
    assert diagnostic["active_policy"]["mean_queries"] < diagnostic["fixed_policy"]["mean_queries"]

    rules = _load("improvement_rules.json")
    policy = rules["pg36_active_belief_policy"]
    assert policy["replay_only"] is True
    assert policy["belief_duplicate_evidence_guard"] is True
    assert policy["active_controller_is_not_capability_proof"] is True
    assert policy["training_allowed"] is False


def test_pg36_protocol_records_formal_candidate_failure_without_promotion():
    protocol = _load("pg36_delayed_maze_protocol_v1.json")
    candidate = protocol["formal_candidate"]
    assert candidate["training_rows"] == 128
    assert candidate["typed_oracle_consumed_by_model"] is False
    assert candidate["family_holdout"] == "north implementation; logic and url_redirect"
    assert candidate["source_holdout"] == "south implementation; train/dev roles"
    assert candidate["strict_results"]["family_holdout_typed_recall"] == 0.0
    assert candidate["strict_results"]["source_holdout_false_positive_rate"] == 0.0
    assert candidate["capability_gate_status"] == "blocked"
    assert candidate["capability_claim_allowed"] is False
    assert candidate["training_allowed"] is False
