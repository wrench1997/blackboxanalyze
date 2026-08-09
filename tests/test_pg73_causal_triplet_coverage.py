import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg73_audit_blocks_training_until_true_negative_triplets_exist():
    report = _read("pg73_causal_triplet_coverage_audit_report_v1.json")
    assert report["status"] == "coverage_audit_completed"
    assert report["metrics"]["combined_known"]["step_count"] == 25
    assert report["metrics"]["neutral_projection_count"] == 0
    assert report["metrics"]["negative_control_projection_count"] == 0
    assert report["metrics"]["typed_negative_probe_count"] == 0
    assert report["metrics"]["all_current_reject_rows_are_synthetic_zero"] is True
    assert report["metrics"]["pg74_triplet"]["triplet_count"] == 21
    assert report["metrics"]["pg74_triplet_typed_positive_count"] == 21
    assert report["metrics"]["pg74_triplet_typed_negative_count"] == 42
    assert report["root_cause"]["repair_status"] == "pg74_triplet_collector_passed_collection_gate"
    assert report["root_cause"]["primary"] == "missing_causal_triplet_negative_probe_in_pg69_pg72"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
