import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg206_xxl_capacity_improves_field_and_failure_generalization() -> None:
    report = _load("research/pg206_body_capacity_holdout_report_v1.json")
    assert report["status"] == "completed_matched_large_xxl_field_token_holdout"
    large, xxl = report["variants"]
    assert large["body_parameter_count"] < 20_000_000
    assert xxl["body_parameter_count"] > 100_000_000
    assert large["training"]["holdout"]["action_accuracy"] == 1.0
    assert large["training"]["holdout"]["encoding_accuracy"] == 0.0
    assert large["training"]["holdout"]["failure_accuracy"] == 0.42857143
    assert large["fresh_route_replay"]["unsafe_allow_count"] == 2
    assert xxl["training"]["holdout"]["action_accuracy"] == 1.0
    assert xxl["training"]["holdout"]["encoding_accuracy"] == 1.0
    assert xxl["training"]["holdout"]["failure_accuracy"] == 1.0
    assert xxl["fresh_route_replay"]["action_accuracy"] == 1.0
    assert xxl["fresh_route_replay"]["unsafe_allow_count"] == 0
    assert report["capacity_101m_better"] is True
    assert report["promotion"]["selected_variant"] == "xxl"


def test_pg206_capacity_report_is_quarantined_and_raw_free() -> None:
    report = _load("research/pg206_body_capacity_holdout_report_v1.json")
    protocol = _load("research/pg206_body_capacity_holdout_protocol_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert protocol["raw_payload_and_response_excluded"] is True
    assert rules["pg206_body_capacity_holdout"]["capacity_101m_better"] is True
    assert rules["pg206_body_capacity_holdout"]["selected_variant"] == "xxl"
