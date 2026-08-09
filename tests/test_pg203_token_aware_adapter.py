import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg203_structural_tokens_fix_auxiliary_generalization() -> None:
    report = _load("research/pg203_token_aware_adapter_report_v1.json")
    assert report["status"] == "completed_explicit_encoding_failure_token_adapter"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["model"]["token_feature_dim"] == 10
    assert report["counts"]["holdout_action_accuracy"] == 1.0
    assert report["counts"]["holdout_encoding_accuracy"] == 1.0
    assert report["counts"]["holdout_failure_accuracy"] == 1.0
    assert report["counts"]["holdout_unsafe_allow_count"] == 0


def test_pg203_replay_has_no_forgetting_or_unsafe_allow() -> None:
    report = _load("research/pg203_token_aware_adapter_report_v1.json")
    assert report["counts"]["replay_action_accuracy"] == 1.0
    assert report["counts"]["replay_encoding_accuracy"] == 1.0
    assert report["counts"]["replay_failure_accuracy"] == 1.0
    assert report["counts"]["replay_unsafe_allow_count"] == 0
    assert report["counts"]["catastrophic_forgetting_detected"] is False


def test_pg203_keeps_structural_tokens_separate_from_evaluator_labels() -> None:
    report = _load("research/pg203_token_aware_adapter_report_v1.json")
    protocol = _load("research/pg203_token_aware_adapter_protocol_v1.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["data"]["evaluator_labels_in_tokens"] is False
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg203_rule_is_registered() -> None:
    rules = _load("research/improvement_rules.json")
    rule = rules["pg203_token_aware_adapter"]
    assert rule["base_parameter_count"] > 100_000_000
    assert rule["holdout_encoding_accuracy"] == 1.0
    assert rule["holdout_failure_accuracy"] == 1.0
    assert rule["holdout_unsafe_allow_count"] == 0
    assert rule["training_promotion_allowed"] is False

