import json
from pathlib import Path

from app.pg231_feedback_trajectory import feedback_tokens, prepare_feedback_record


ROOT = Path(__file__).resolve().parents[1]


def test_pg231_trajectory_keeps_observation_and_target_suffix_separate() -> None:
    row = {
        "source": "local-lab",
        "method": "POST",
        "seed": 23101,
        "surface_role": "sql_surface",
        "status_class": "2xx",
        "field_count": 2,
        "history_len": 1,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "candidate_sent": True,
        "oracle_available": False,
        "candidate_reference_agreement": True,
        "negative_clean": True,
        "binding_valid": True,
        "evidence_hash": "b" * 64,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "next_step": "recheck_oracle",
        "previous_feedback": "none",
    }
    record = prepare_feedback_record(row)
    tokens = feedback_tokens(row, record["lane"])
    assert record["lane"] == "silver"
    assert record["classification_position"] == tokens.index("failure=oracle_unavailable")
    assert tokens[record["classification_position"] + 1] == "phase=repair"
    assert "lane=silver" in tokens
    assert "repair=recheck_oracle" in tokens


def test_pg231_report_is_expanded_but_not_promoted() -> None:
    report = json.loads((ROOT / "research" / "pg231_feedback_trajectory_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg231_feedback_trajectory_dataset_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_feedback_trajectory_frozen_xxl_training"
    assert report["funnel"]["raw_records"] == 478
    assert report["funnel"]["unique_records"] == 102
    assert report["funnel"]["duplicate_records"] == 376
    assert report["funnel"]["lane_counts"] == {"gold": 30, "silver": 21, "hard_negative": 9, "quarantine": 42}
    assert report["selected"]["holdout"]["next_token_accuracy"] > 0.7
    assert report["selected"]["holdout"]["lane_accuracy"] == 1.0
    assert report["selected"]["holdout"]["self_error_recall"] == 1.0
    assert report["split"]["holdout"] == 13
    assert report["honesty"]["data_expanded_but_local"] is True
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert dataset["contract"]["diagnosis_targets_not_features"] is True
    assert dataset["contract"]["classification_context_excludes_lane_and_repair_targets"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    for record in dataset["records"]:
        assert "route" not in record
        assert "payload" not in record
        assert record["classification_position"] < len(record["tokens"])
