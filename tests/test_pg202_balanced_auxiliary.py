import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg202_balanced_training_preserves_action_gate() -> None:
    report = _load("research/pg202_balanced_auxiliary_report_v1.json")
    assert report["status"] == "completed_balanced_encoding_failure_auxiliary_training"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["data"]["augmentation_rows"] == 80
    assert report["counts"]["holdout_action_accuracy"] == 1.0
    assert report["counts"]["holdout_unsafe_allow_count"] == 0
    assert report["counts"]["replay_action_accuracy"] == 1.0
    assert report["counts"]["replay_unsafe_allow_count"] == 0


def test_pg202_exposes_encoding_information_gap_instead_of_hiding_it() -> None:
    report = _load("research/pg202_balanced_auxiliary_report_v1.json")
    assert report["counts"]["holdout_encoding_accuracy"] < 0.5
    assert report["counts"]["holdout_failure_accuracy"] >= 0.7
    assert report["counts"]["catastrophic_forgetting_detected"] is False


def test_pg202_is_quarantined() -> None:
    report = _load("research/pg202_balanced_auxiliary_report_v1.json")
    protocol = _load("research/pg202_balanced_auxiliary_protocol_v1.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["safety"]["abstract_augmentation_only"] is True
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg202_rule_is_registered() -> None:
    rules = _load("research/improvement_rules.json")
    rule = rules["pg202_balanced_auxiliary"]
    assert rule["base_parameter_count"] > 100_000_000
    assert rule["augmentation_rows"] == 80
    assert rule["holdout_encoding_accuracy"] == 0.2857143
    assert rule["holdout_unsafe_allow_count"] == 0
    assert rule["training_promotion_allowed"] is False

