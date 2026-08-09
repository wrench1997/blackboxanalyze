import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg61_learns_state_dependent_get_post_action_on_holdout():
    report = _read("pg61_target_zone_counterfactual_report_v1.json")
    holdout = report["holdout"]
    assert report["status"] == "diagnostic_only"
    assert report["training"]["oracle_consumed_by_model"] is False
    assert report["training"]["family_consumed_by_model"] is False
    assert holdout["task_count"] == 160
    assert holdout["positive_action_accuracy"] == 1.0
    assert holdout["target_success_rate"] == 1.0
    assert holdout["selected_action_entropy"] >= 0.5
    assert set(holdout["selected_action_counts"]) == {"GET", "POST"}
    assert holdout["negative_false_accept_count"] == 0
    assert holdout["unknown_strict_abstain"] is True
    assert report["hard_gate"]["status"] == "passed"
    assert report["hard_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg61_trace_is_post_action_typed_and_bounded():
    trace = _read("pg61_target_zone_counterfactual_trace_v1.json")
    assert trace["step_count"] == 160
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    for step in trace["steps"]:
        assert step["reset"]["fresh_target"] is True
        assert step["reset"]["completed"] is True
        assert step["online_weight_update"] is False
        assert step["long_term_memory_write"] is False
        assert step["raw_probe_stored"] is False
        assert step["raw_response_stored"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_hash"])
        assert step["oracle_after_action"]["evaluator_state_hidden"] is True
    text = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text


def test_pg61_protocol_forbids_oracle_leakage_and_fixture_promotion():
    protocol = _read("pg61_target_zone_counterfactual_protocol_v1.json")
    contract = protocol["input_contract"]
    assert protocol["authorized_scope"]["target_host"] == "127.0.0.1"
    assert protocol["authorized_scope"]["external_network"] is False
    assert protocol["counterfactual_contract"]["same_base_state_with_both_best_methods"] is True
    assert protocol["counterfactual_contract"]["randomized_candidate_order"] is True
    assert protocol["counterfactual_contract"]["typed_oracle_after_action_only"] is True
    assert "oracle_projection" in contract["model_must_not_read"]
    assert "expected_method" in contract["model_must_not_read"]
    assert protocol["run_result"]["selected_threshold"] >= 0.5
