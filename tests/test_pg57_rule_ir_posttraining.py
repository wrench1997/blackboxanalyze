import json
from pathlib import Path


def _read(name):
    return json.loads(Path("research", name).read_text(encoding="utf-8"))


def test_pg57_oracle_is_target_only_and_prefix_is_bounded():
    protocol = _read("pg57_rule_ir_posttraining_protocol_v1.json")
    report = _read("pg57_rule_ir_posttraining_report_v1.json")
    contract = report["training_contract"]
    assert contract["oracle_is_target_only"] is True
    assert contract["family_name_in_input"] is False
    assert contract["unknown_family_class"] is False
    assert contract["prefix_ends_at_oracle_target"] is True
    assert report["device"] == "cuda"
    assert report["split_counts"] == {"train": 322, "dev": 188, "holdout": 120}
    assert protocol["gates"]["threshold_calibrated_on_dev_only"] is True


def test_pg57_raw_signal_is_not_promotion_and_calibrated_gate_abstains():
    report = _read("pg57_rule_ir_posttraining_report_v1.json")
    raw = report["metrics"]["holdout_raw"]
    calibrated = report["metrics"]["holdout_calibrated"]
    assert raw["confirmed_recall"] == 1.0
    assert raw["unknown_family_confirmed_attempts"] == 12
    assert calibrated["confirmed_recall"] == 0.0
    assert calibrated["unknown_family_confirmed_attempts"] == 0
    assert calibrated["negative_confirmed_false_accept_count"] == 0
    assert calibrated["abstain_or_reject_rate"] == 1.0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_capability_claim_allowed"] is False
