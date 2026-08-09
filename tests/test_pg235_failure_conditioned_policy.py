import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg235_unseen_family_false_send_blocks_promotion() -> None:
    report = json.loads((ROOT / "research" / "pg235_failure_conditioned_policy_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg235_failure_conditioned_policy_dataset_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_failure_conditioned_abstention_training"
    assert report["counts"]["train_rows"] == 63
    assert report["counts"]["unseen_family_holdout_rows"] == 19
    assert report["selected"]["metrics"]["unseen_family_holdout"]["false_send_count"] == 4
    assert report["selected"]["metrics"]["unseen_family_holdout"]["abstain_recall"] == 0.78947368
    assert report["strict_unseen_family_abstain_pass"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["frozen_body_changed"] is False
    assert dataset["contract"]["typed_result_required_for_send_candidate"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False

