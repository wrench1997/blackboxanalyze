import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg209_uses_route_seed_holdout_and_keeps_xxl_capacity_candidate() -> None:
    report = _load("research/pg209_parameterized_trace_training_report_v1.json")
    assert report["status"] == "completed_pg208_route_seed_holdout_capacity_sweep"
    assert report["device"] == "cuda"
    assert report["data"]["train_rows"] == 25
    assert report["data"]["holdout_rows"] == 58
    large, xxl = report["variants"]
    assert large["body_parameter_count"] < 20_000_000
    assert xxl["body_parameter_count"] > 100_000_000
    assert large["old_replay"]["unsafe_allow_count"] == 0
    assert xxl["old_replay"]["unsafe_allow_count"] == 0
    assert large["pg208_holdout"]["action_accuracy"] == 1.0
    assert xxl["pg208_holdout"]["action_accuracy"] == 1.0
    assert large["catastrophic_forgetting_detected"] is False
    assert xxl["catastrophic_forgetting_detected"] is False
    assert report["capacity_101m_better"] is True
    assert report["selected_variant"] == "xxl"


def test_pg209_training_artifacts_remain_quarantined_and_raw_free() -> None:
    report = _load("research/pg209_parameterized_trace_training_report_v1.json")
    protocol = _load("research/pg209_parameterized_trace_training_protocol_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert protocol["oracle_labels_as_model_inputs"] is False
    assert protocol["raw_payload_and_response_excluded"] is True
    assert rules["pg209_parameterized_trace_training"]["selected_variant"] == "xxl"
    assert rules["pg209_parameterized_trace_training"]["training_promotion_allowed"] is False

