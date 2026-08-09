import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg236_seed_holdout_separates_safety_from_capability() -> None:
    report = json.loads((ROOT / "research" / "pg236_failure_conditioned_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg236_failure_conditioned_training_dataset_v1.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))["pg236_failure_conditioned_training"]

    assert report["status"] == "completed_independent_seed_holdout_failure_conditioned_training"
    assert report["counts"]["independent_replay_records"] == 28
    assert report["counts"]["seed23632_holdout_rows"] == 14
    assert report["honesty"]["seed23632_is_never_in_training"] is True
    assert report["strict_seed_holdout_abstain_pass"] is True
    assert report["seed_holdout_capability_gate_pass"] is False
    assert report["holdout_action_counts"] == {"abstain": 14}
    assert report["selected"]["metrics"]["seed23632_holdout"]["false_send_count"] == 0
    assert report["selected"]["metrics"]["seed23632_holdout"]["abstain_recall"] == 1.0
    assert report["frozen_body_changed"] is False
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert dataset["contract"]["typed_oracle_unavailable_records_abstain_only"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert rule["next_token_loss_is_not_quality_gate"] is True
    assert rule["seed_holdout_capability_gate_pass"] is False
    assert "failure_signature" in rule["self_check_dataset_contract"]["required_before_gold"]
    assert "replay_evidence_hash" in rule["self_check_dataset_contract"]["required_before_gold"]
    assert rule["memory_promotion_allowed"] is False

