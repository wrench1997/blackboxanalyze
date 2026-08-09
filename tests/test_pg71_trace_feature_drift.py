import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg71_audit_records_legacy_collision_and_blocks_promotion():
    report = _read("pg71_trace_feature_drift_audit_report_v1.json")
    assert report["status"] == "feature_drift_audit_completed"
    assert report["metrics"]["known_pair_count"] == 4
    assert report["metrics"]["legacy_candidate_control_duplicate_label_conflict_count"] == 3
    assert report["metrics"]["pair_observable_shape_delta_count"] == 4
    assert report["root_cause"]["primary"] == "feature_extractor_drops_bounded_shape_differences"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_claim_allowed"] is False

