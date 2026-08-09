import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg205_trains_field_aware_xxl_and_keeps_wider_negative_diagnostic() -> None:
    report = _load("research/pg205_field_token_training_and_replay_report_v1.json")
    assert report["status"] == "completed_field_token_capacity_and_fresh_pikachu_replay"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["model"]["field_token_dim"] == 31
    assert report["training"]["train_rows"] == 75
    assert report["training"]["augmentation_rows"] == 60
    selected = next(row for row in report["training"]["variants"] if row["variant"] == "standard")
    wide = next(row for row in report["training"]["variants"] if row["variant"] == "wide")
    assert selected["training"]["holdout"]["action_accuracy"] == 1.0
    assert selected["training"]["holdout"]["encoding_accuracy"] == 1.0
    assert selected["training"]["holdout"]["failure_accuracy"] == 1.0
    assert selected["training"]["holdout"]["unsafe_allow_count"] == 0
    assert wide["training"]["holdout"]["encoding_accuracy"] < 1.0
    assert report["training"]["selected_variant"] == "standard"


def test_pg205_fresh_pikachu_replays_get_post_redirect_and_multi_parameter_surfaces() -> None:
    report = _load("research/pg205_field_token_training_and_replay_report_v1.json")
    counts = report["counts"]
    assert counts["fresh_container_count"] == 2
    assert counts["route_replay_count"] == 10
    assert counts["get_candidate_send_count"] == 2
    assert counts["post_candidate_send_count"] == 0
    assert counts["unknown_oracle_abstain_count"] == 6
    assert counts["multi_parameter_route_count"] == 8
    assert counts["redirect_chain_observed_count"] == 2
    assert counts["false_positive_count"] == 0
    assert all(row["token_validation"]["valid"] for row in report["route_runs"])


def test_pg205_field_faults_fail_closed_and_artifacts_are_quarantined() -> None:
    report = _load("research/pg205_field_token_training_and_replay_report_v1.json")
    protocol = _load("research/pg205_field_token_training_and_replay_protocol_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["counts"]["field_token_fault_count"] == 40
    assert report["counts"]["network_allowed_on_fault_count"] == 0
    assert all(not row["network_allowed"] for row in report["fault_runs"])
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert protocol["raw_payload_and_response_excluded"] is True
    assert rules["pg205_field_token_training_and_replay"]["selected_variant"] == "standard"
    assert rules["pg205_field_token_training_and_replay"]["network_allowed_on_fault_count"] == 0
