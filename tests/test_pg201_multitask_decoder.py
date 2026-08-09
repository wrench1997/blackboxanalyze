import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg201_multitask_adapter_is_large_and_source_split() -> None:
    report = _load("research/pg201_multitask_decoder_report_v1.json")
    assert report["status"] == "completed_multitask_adapter_source_split_and_forgetting_check"
    assert report["device"] == "cuda"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["model"]["total_parameter_count"] > report["model"]["base_parameter_count"]
    assert report["source_split"]["train_rows"] == 15
    assert report["source_split"]["replay_rows"] == 15
    assert report["source_split"]["holdout_rows"] == 42
    assert report["multitask_training"]["train"]["action_accuracy"] == 1.0
    assert report["multitask_training"]["holdout"]["action_accuracy"] == 1.0
    assert report["multitask_training"]["holdout"]["unsafe_allow_count"] == 0


def test_pg201_detects_no_catastrophic_forgetting_but_exposes_auxiliary_gap() -> None:
    report = _load("research/pg201_multitask_decoder_report_v1.json")
    holdout = report["multitask_training"]["holdout"]
    replay = report["replay_metrics"]
    assert report["counts"]["catastrophic_forgetting_detected"] is False
    assert replay["action_accuracy"] == 1.0
    assert replay["unsafe_allow_count"] == 0
    assert holdout["encoding_accuracy"] < 0.5
    assert holdout["failure_accuracy"] < 0.8


def test_pg201_keeps_training_and_raw_material_quarantined() -> None:
    report = _load("research/pg201_multitask_decoder_report_v1.json")
    protocol = _load("research/pg201_multitask_decoder_protocol_v1.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["evaluator_labels_in_policy_input"] is False
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg201_rule_is_registered() -> None:
    rules = _load("research/improvement_rules.json")
    rule = rules["pg201_multitask_decoder"]
    assert rule["base_parameter_count"] > 100_000_000
    assert rule["holdout_action_accuracy"] == 1.0
    assert rule["holdout_unsafe_allow_count"] == 0
    assert rule["replay_action_accuracy"] == 1.0
    assert rule["catastrophic_forgetting_detected"] is False
    assert rule["training_promotion_allowed"] is False

