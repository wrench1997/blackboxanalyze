import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg67_separates_effect_confirmation_from_rule_ir_binding():
    report = _read("pg67_independent_rule_ir_oracle_noise_report_v1.json")
    metrics = report["metrics"]
    assert report["status"] == "diagnostic_only"
    assert report["source"]["model_retrained_on_pg67"] is False
    assert report["source"]["family_in_pre_oracle_input"] is False
    assert metrics["effect_recall"] == 1.0
    assert metrics["known_family_recall"] >= 0.8
    assert metrics["unknown_misname_count"] == 0
    assert metrics["negative_false_accept_count"] == 0
    assert metrics["unknown_strict_abstain"] is True
    assert metrics["rule_ir_abstain_count"] > 0
    assert report["hard_gate"]["status"] == "passed"
    assert report["hard_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg67_trace_keeps_noisy_oracle_after_action_and_hashes():
    trace = _read("pg67_independent_rule_ir_oracle_noise_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["model_retrained_on_pg67"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["episode_count"] == 192
    for episode in trace["episodes"]:
        for step in episode["steps"]:
            assert step["reset"]["fresh_target"] is True
            assert step["reset"]["evaluator_state_hidden"] is True
            assert step["oracle_after_action"]["evaluator_state_hidden"] is True
            assert step["raw_probe_stored"] is False
            assert step["raw_response_stored"] is False
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
            assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_hash"])
            assert step["rule_ir_after_action"]["executable"] is False
    text = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "union select" not in text
    assert "onerror" not in text


def test_pg67_protocol_forbids_family_leakage_and_requires_abstention():
    protocol = _read("pg67_independent_rule_ir_oracle_noise_protocol_v1.json")
    contract = protocol["input_contract"]
    gates = protocol["required_gates"]
    assert protocol["authorized_scope"]["target_host"] == "127.0.0.1"
    assert protocol["authorized_scope"]["external_network"] is False
    assert contract["family_before_action_forbidden"] is True
    assert contract["typed_oracle_after_action_only"] is True
    assert contract["rule_ir_binding_after_typed_exit_only"] is True
    assert gates["unknown_misname_zero"] is True
    assert gates["contradictory_oracle_must_abstain"] is True
    assert protocol["run_result"]["status"] == "passed"
