import json
from pathlib import Path

from app.pg213_history_policy import HistoryProcessPolicy, process_feature_vector, predict_process_action


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg213_history_features_distinguish_environment_failure() -> None:
    row = {"typed_available": False, "backend_state": "database_unavailable", "method": "POST", "field_count": 2, "history_len": 1, "previous_feedback": "environment_failure", "surface_family": "injection", "binding_valid": True}
    values = process_feature_vector(row)
    assert len(values) == 24
    assert values[2] == 1.0
    assert values[4] == 1.0
    assert values[15] == 1.0


def test_pg213_policy_predicts_a_bounded_action() -> None:
    model = HistoryProcessPolicy().eval()
    result = predict_process_action(model, {"typed_available": False, "backend_state": "database_unavailable", "method": "GET", "field_count": 1, "history_len": 1, "previous_feedback": "environment_failure", "surface_family": "injection", "binding_valid": True})
    assert result["action"] in {"abstain", "safe_candidate", "retry_alternate"}
    assert result["feature_dim"] == 24


def test_pg213_report_has_history_and_counterfactual_gates() -> None:
    report = _load("research/pg213_history_policy_training_report_v1.json")
    assert report["status"] == "completed_history_policy_counterfactual_holdout"
    assert report["data"]["counterfactual_rows"] > 0
    assert report["training"]["holdout"]["unsafe_allow_count"] == 0
    assert report["promotion"]["artifact_promotion_allowed"] is False
