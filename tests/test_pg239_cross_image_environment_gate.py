import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg239_cross_image_failure_is_quarantined_not_labeled_negative() -> None:
    report = json.loads((ROOT / "research" / "pg239_alt_pikachu_get_post_replay_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg239_alt_pikachu_get_post_replay_dataset_v1.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((ROOT / "research" / "pg239_alt_pikachu_get_post_replay_protocol_v1.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))["pg239_cross_image_environment_gate"]

    assert report["status"] == "completed_cross_image_environment_gated_replay"
    assert report["counts"]["fresh_container_count"] == 14
    assert report["counts"]["get_count"] == 10
    assert report["counts"]["post_count"] == 4
    assert report["counts"]["reference_send_count"] == 14
    assert report["counts"]["negative_send_count"] == 14
    assert report["counts"]["typed_oracle_available_count"] == 0
    assert report["counts"]["confirmed_positive_count"] == 0
    assert report["counts"]["abstain_count"] == 14
    assert report["counts"]["environment_failure_count"] == 14
    assert report["counts"]["training_eligible_count"] == 0
    assert report["honesty"]["alternate_php_oracle_unavailable"] is True
    assert report["honesty"]["all_rows_abstain_only"] is True
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert dataset["contract"]["environment_failure_is_not_model_label"] is True
    assert dataset["contract"]["typed_oracle_required_for_positive"] is True
    assert dataset["contract"]["training_eligible"] is False
    assert protocol["environment_failure_quarantined"] is True
    assert rule["cross_implementation_capability_established"] is False
    assert rule["environment_failure_is_not_model_label"] is True

