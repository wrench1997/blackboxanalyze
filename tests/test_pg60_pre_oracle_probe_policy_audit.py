import json
from pathlib import Path


def _read(name):
    return json.loads(Path("research", name).read_text(encoding="utf-8"))


def test_pg60_audit_detects_fixed_confirmation_order():
    protocol = _read("pg60_pre_oracle_probe_policy_audit_protocol_v1.json")
    report = _read("pg60_pre_oracle_probe_policy_audit_report_v1.json")
    metrics = report["metrics"]
    gate = report["hard_gate"]
    assert metrics["episode_count"] == 180
    assert metrics["step_count"] == 558
    assert metrics["paired_get_post_group_count"] == 180
    assert metrics["first_action_counts"] == {"GET": 180}
    assert metrics["confirmation_action_counts"] == {"GET.confirm": 162}
    assert metrics["confirmation_action_entropy"] == 0.0
    assert gate["status"] == "blocked"
    assert "confirmation_action_is_fixed_order_confounded" in gate["reasons"]
    assert protocol["required_hard_gates"]["confirmation_action_entropy_min"] == 0.5


def test_pg60_keeps_pre_oracle_input_clean_and_never_promotes():
    report = _read("pg60_pre_oracle_probe_policy_audit_report_v1.json")
    contract = report["inputs"]["model_input_contract"]
    metrics = report["metrics"]
    assert contract["state_features_reads_evaluator_fields"] is False
    assert contract["invariant_reads_evaluator_fields"] is False
    assert contract["report_declares_typed_oracle_consumed_by_model"] is False
    assert contract["report_declares_family_consumed_by_model"] is False
    assert metrics["raw_probe_stored_count"] == 0
    assert metrics["raw_response_stored_count"] == 0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_capability_claim_allowed"] is False
