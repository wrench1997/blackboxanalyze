import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg232_strict_source_holdout_blocks_promotion() -> None:
    report = json.loads((ROOT / "research" / "pg232_source_holdout_audit_report_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_strict_source_holdout_audit"
    assert report["usable_records"] == 60
    assert report["heldout_source_count"] == 5
    assert report["strict_source_holdout_pass"] is False
    folds = {row["heldout_source"]: row for row in report["folds"]}
    assert folds["pg222_observed_process"]["holdout"]["repair_accuracy"] == 0.47826087
    assert folds["pg229_juice_shop_fresh_typed_replay"]["holdout"]["repair_accuracy"] == 0.0
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["frozen_body_changed"] is False
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False

