import json
from pathlib import Path

import pytest
import torch

from app.pg222_problem_diagnoser import (
    FEATURE_DIM,
    ProblemDiagnoser,
    diagnose_features,
    guarded_diagnosis,
    hard_diagnostic_gate,
    predict_diagnosis,
)


ROOT = Path(__file__).resolve().parents[1]


def _positive_row() -> dict:
    return {
        "method": "GET",
        "field_count": 2,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "database_health_ok": True,
        "backend_observed": True,
        "transport_error": False,
        "container_restart_used": False,
        "binding_valid": True,
        "candidate_sent": True,
        "reference_sent": True,
        "negative_sent": True,
        "oracle_available": True,
        "typed_effect_observed": True,
        "result_fixture_verified": True,
        "boolean_differential": False,
        "candidate_reference_agreement": True,
        "negative_clean": True,
        "result_mismatch_observed": False,
        "source_hash": "0" * 64,
        "evidence_hash": "1" * 64,
    }


def test_pg222_feature_vector_is_bounded_and_rejects_targets() -> None:
    row = _positive_row()
    assert len(diagnose_features(row)) == FEATURE_DIM
    with pytest.raises(ValueError):
        diagnose_features({**row, "diagnosis": "confirmed_local_effect"})


def test_pg222_guard_catches_model_positive_claim_without_gate() -> None:
    row = {**_positive_row(), "oracle_available": False, "typed_effect_observed": False, "result_fixture_verified": False, "model_claimed_positive": True}
    assert hard_diagnostic_gate(row) is False
    assert guarded_diagnosis("confirmed_local_effect", row) == "model_decision_error"


def test_pg222_positive_prediction_is_allowed_only_after_gate() -> None:
    model = ProblemDiagnoser(hidden_dim=16).eval()
    result = predict_diagnosis(model, {**_positive_row(), "model_claimed_positive": False}, device=torch.device("cpu"))
    assert result["feature_dim"] == FEATURE_DIM
    assert result["hard_diagnostic_gate"] is True
    assert result["raw_payload_strings_stored"] is False
    assert result["raw_response_bodies_stored"] is False


def test_pg222_report_has_seed_route_holdout_and_zero_guarded_positive_false_accepts() -> None:
    report = json.loads((ROOT / "research" / "pg222_problem_diagnoser_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg222_problem_diagnoser_dataset_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_structured_problem_diagnosis_training"
    assert report["row_counts"]["counterfactual"] > 0
    assert report["selected"]["holdout"]["guarded_positive_false_accept_count"] == 0
    assert report["selected"]["holdout"]["guarded_confirmed_local_effect_recall"] == 1.0
    assert dataset["split"]["holdout_rows"] > 0
    assert dataset["contract"]["evaluator_targets_not_features"] is True
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["rows"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["rows"])
