import json
from pathlib import Path


def _read(name):
    return json.loads(Path("research", name).read_text(encoding="utf-8"))


def test_pg58_keeps_effect_and_family_heads_separate():
    protocol = _read("pg58_effect_family_decoupling_protocol_v1.json")
    report = _read("pg58_effect_family_decoupling_report_v1.json")
    contract = report["training_contract"]
    assert contract["family_head_train_known_families_only"] is True
    assert contract["unknown_family_excluded_from_training"] == "template_injection"
    assert contract["family_name_in_input"] is False
    assert contract["effect_head_separate"] is True
    assert contract["effect_head_is_not_family_evidence"] is True
    assert protocol["gates"]["zero_known_wrong_family_required"] is True
    assert report["device"] == "cuda"


def test_pg58_family_gate_does_not_turn_abstention_into_capability():
    report = _read("pg58_effect_family_decoupling_report_v1.json")
    raw = report["metrics"]["holdout_raw"]
    calibrated = report["metrics"]["holdout_calibrated"]
    assert raw["unknown_misname_count"] == 0
    assert calibrated["known_family_recall"] == 0.0
    assert calibrated["known_wrong_family_count"] == 0
    assert calibrated["unknown_misname_count"] == 0
    assert calibrated["negative_false_accept_count"] == 0
    assert calibrated["abstain_rate"] == 1.0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_capability_claim_allowed"] is False
