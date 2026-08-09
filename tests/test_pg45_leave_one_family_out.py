import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg45_removes_injection_from_training_and_abstains_on_it():
    report = _load("pg45_leave_one_family_out_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["held_out"]["family"] == "injection"
    assert report["held_out"]["semantic_reference"] == "operator-context"
    assert report["held_out"]["removed_from_training"] is True
    assert report["training"]["pg42_used_for_training"] is False
    assert report["training"]["typed_oracle_consumed_by_model"] is False
    assert "operator-context" not in report["training"]["remaining_semantic_index"]
    assert report["pg42_splits"]["retained_known"]["known_family_recall"] == 1.0
    held_out = report["pg42_splits"]["held_out_family"]
    assert held_out["unknown_positive_count"] == 36
    assert held_out["unknown_effect_recall"] == 1.0
    assert held_out["unknown_misname_count"] == 0
    assert held_out["unknown_not_abstain_count"] == 0
    assert held_out["unknown_strict_abstain"] is True
    assert report["pg42_splits"]["negative_control"]["false_positive_rate"] == 0.0
    assert report["safe_leave_one_out_gate"]["claim_allowed"] is True
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg45_protocol_requires_zero_misname_without_claiming_new_family_learning():
    protocol = _load("pg45_leave_one_family_out_protocol_v1.json")
    assert protocol["held_out_family"]["training_rows_allowed"] is False
    assert protocol["held_out_family"]["holdout_labels_allowed_as_features"] is False
    assert protocol["promotion_gate"]["held_out_family_zero_misname"] is True
    assert protocol["run_result"]["held_out_family_named_recall"] == 0.0
    assert protocol["run_result"]["held_out_family_unknown_misname_count"] == 0
    assert protocol["run_result"]["safe_leave_one_out_gate_status"] == "passed"
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["status"] == "run_completed_leave_one_out_safe_abstain_no_promotion"
