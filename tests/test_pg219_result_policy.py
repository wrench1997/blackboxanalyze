import json
from pathlib import Path

import torch

from app.pg219_result_policy import (
    RESULT_FEATURE_DIM,
    ResultAwareProcessPolicy,
    guarded_action,
    hard_gate,
    predict_result_policy,
    result_features_for_row,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg219_features_and_hard_gate_are_bounded() -> None:
    row = {
        "method": "POST",
        "field_count": 2,
        "fresh_reset_ok": True,
        "database_health_ok": True,
        "backend_observed": True,
        "negative_clean": True,
        "typed_available": True,
        "binding_valid": True,
        "previous_feedback": "candidate_error",
    }
    assert len(result_features_for_row(row)) == RESULT_FEATURE_DIM
    assert hard_gate(row) is True
    row["typed_available"] = False
    assert hard_gate(row) is False
    assert guarded_action("safe_candidate", row) == "abstain"


def test_pg219_policy_prediction_is_fail_closed() -> None:
    model = ResultAwareProcessPolicy(None, hidden_dim=32).eval()
    ids = torch.ones((1, 4), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    result = predict_result_policy(model, {"method": "GET", "typed_available": False, "fresh_reset_ok": False}, ids, mask)
    assert result["action"] != "safe_candidate"
    assert result["hard_gate"] is False
    assert result["result_feature_dim"] == RESULT_FEATURE_DIM


def test_pg219_report_has_real_split_and_explicit_failure_counterfactuals() -> None:
    report = _load("pg219_result_aware_policy_training_report_v1.json")
    data = report["data"]
    assert report["status"] == "completed_result_aware_process_policy_seed_route_holdout"
    assert data["typed_failure_counterfactual_rows"] > 0
    assert data["route_holdout"] == "/vul/sqli/sqli_x.php"
    assert data["counterfactual_holdout_rows"] > 0
    assert report["promotion"]["live_send_takeover_allowed"] is False
    assert report["safety"]["oracle_labels_as_features"] is False
    assert all(item["training"]["holdout"]["gated_unsafe_allow_count"] == 0 for item in report["variants"])
    assert all(row["raw_payload_strings_stored"] is False for row in _load("pg219_result_aware_policy_dataset_v1.json")["rows"])

