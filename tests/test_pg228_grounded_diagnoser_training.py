import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg228_separates_typed_rows_from_abstention_and_self_error_rows() -> None:
    report = json.loads((ROOT / "research" / "pg228_grounded_diagnoser_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg228_grounded_diagnoser_dataset_v1.json").read_text(encoding="utf-8-sig"))
    counts = report["row_counts"]
    assert report["status"] == "completed_typed_untyped_self_error_diagnoser_training"
    assert counts["total"] == 560
    assert counts["train"] == 164
    assert counts["holdout"] == 396
    assert counts["pg226_typed_sql_result_rows"] == 8
    assert counts["pg227_dom_redirect_rows"] == 14
    assert counts["self_error_counterfactual_rows"] == 110
    assert report["payload_grounded_eligible_count"] == 8
    assert report["selected"]["holdout"]["guarded_diagnosis_accuracy"] == 1.0
    assert report["selected"]["holdout"]["guarded_positive_false_accept_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert dataset["contract"]["typed_sql_result_rows_only_payload_grounded"] is True
    assert dataset["contract"]["dom_effect_is_not_xss"] is True
    assert dataset["contract"]["self_error_counterfactuals_marked"] is True
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["rows"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["rows"])
    grounded = [row for row in dataset["rows"] if row.get("payload_grounded_eligible")]
    assert len(grounded) == 8
    assert all(row["grounding_status"] == "typed_sql_result" for row in grounded)
    assert all(row["diagnosis"] == "confirmed_local_effect" for row in grounded)
    self_errors = [row for row in dataset["rows"] if row.get("grounding_status") == "self_error_counterfactual"]
    assert len(self_errors) == 110
    assert all(row["diagnosis"] == "model_decision_error" for row in self_errors)
    assert all(row["model_claimed_positive"] is True for row in self_errors)
