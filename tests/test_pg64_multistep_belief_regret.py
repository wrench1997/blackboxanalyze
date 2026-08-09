import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg64_active_belief_beats_fixed_order_on_steps_and_regret():
    report = _read("pg64_multistep_belief_regret_report_v1.json")
    active = report["active_belief"]
    fixed = report["fixed_order"]
    comparison = report["comparison"]
    assert report["status"] == "diagnostic_only"
    assert report["controller"]["typed_oracle_before_action"] is False
    assert report["controller"]["family_oracle_used_before_action"] is False
    assert active["target_recall"] == 1.0
    assert active["negative_false_accept_count"] == 0
    assert active["unknown_strict_abstain"] is True
    assert active["multi_step_episode_rate"] >= 0.5
    assert active["mean_steps"] < fixed["mean_steps"]
    assert active["mean_counterfactual_regret"] < fixed["mean_counterfactual_regret"]
    assert comparison["mean_step_delta"] < 0
    assert comparison["mean_regret_delta"] < 0
    assert report["hard_gate"]["status"] == "passed"
    assert report["hard_gate"]["claim_allowed"] is False


def test_pg64_trace_records_belief_chain_after_action_only():
    trace = _read("pg64_multistep_belief_regret_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    for policy in ("active_belief", "fixed_order"):
        group = trace[policy]
        assert group["episode_count"] == 288
        for episode in group["episodes"]:
            for step in episode["steps"]:
                assert step["reset"]["fresh_target"] is True
                assert step["reset"]["completed"] is True
                assert step["reset"]["evaluator_state_hidden"] is True
                assert step["oracle_after_action"]["evaluator_state_hidden"] is True
                assert step["raw_probe_stored"] is False
                assert step["raw_response_stored"] is False
                assert step["online_weight_update"] is False
                assert step["long_term_memory_write"] is False
                assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_hash"])
                assert "belief_before" in step and "belief_after" in step
                assert "expected_information_gain" in step
                assert "counterfactual_regret" in step
    text = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "union select" not in text
    assert "onerror" not in text


def test_pg64_protocol_requires_no_oracle_leakage_and_baseline():
    protocol = _read("pg64_multistep_belief_regret_protocol_v1.json")
    contract = protocol["input_contract"]
    gates = protocol["required_gates"]
    assert protocol["authorized_scope"]["target_host"] == "127.0.0.1"
    assert protocol["authorized_scope"]["external_network"] is False
    assert contract["pre_oracle_surface_projection_only"] is True
    assert contract["typed_oracle_after_action_only"] is True
    assert contract["family_label_before_action_forbidden"] is True
    assert gates["fixed_order_baseline_required"] is True
    assert gates["counterfactual_regret_reported"] is True
    assert gates["negative_false_accept_zero"] is True
    assert protocol["run_result"]["status"] == "passed"
