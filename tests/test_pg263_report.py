import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg263_pg262_augmented_masked_capacity_training_report_v1.json"


def test_pg263_completed_audited_candidate_report_is_integrity_bounded():
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "completed_pg263_pg262_augmented_masked_capacity_training"
    assert report["capacity_variants"] == [2048, 4096, 8192]
    assert report["counts"]["pg262_rows"] == 20
    assert report["counts"]["pg262_holdout_rows"] == 9
    assert report["evaluation_audit"]["audit_id"] == "pg263-final-report-audit-v1"
    assert report["evaluation_audit"]["artifact_present"] is True
    assert report["evaluation_audit"]["weights_changed"] is False
    assert report["independent_final_judge"]["pass"] is True
    assert report["independent_final_judge"]["decision"] == "candidate_eligible_for_next_replay"
    assert report["model_input_excludes_oracle_target"] is True
    assert report["honesty"]["raw_payload_strings_stored"] is False
    assert report["honesty"]["raw_response_bodies_stored"] is False
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["resource_profile"]["gradient_accumulation_enabled"] is True

