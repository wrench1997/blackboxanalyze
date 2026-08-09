import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg244_real_multistep_sql_xss_repair_has_fresh_replay_and_lineage() -> None:
    report = _read("pg244_failure_repair_trajectory_report_v1.json")
    dataset = _read("pg244_failure_repair_trajectory_dataset_v1.json")
    assert report["status"] == "completed_local_sql_xss_multistep_failure_repair_replay"
    assert report["device"] == "cuda"
    assert report["counts"] == {
        "episode_count": 8,
        "fresh_container_count": 16,
        "record_count": 24,
        "sql_episode_count": 4,
        "xss_episode_count": 4,
        "get_episode_count": 8,
        "post_episode_count": 8,
        "gold_count": 16,
        "hard_negative_count": 8,
        "model_self_error_count": 0,
        "replay_count": 8,
        "wire_display_count": 64,
        "external_network_count": 0,
    }
    assert dataset["counts"] == {"records": 24, "gold": 16, "hard_negative": 8, "silver": 0, "quarantine": 0}
    assert dataset["contract"]["counterfactual_failure_has_repair_target"] is True
    assert dataset["contract"]["fresh_replay_required"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert {row["surface_class"] for row in dataset["records"]} == {"sql_surface", "dom_surface"}
    assert all(row["raw_payload_strings_stored"] is False and row["raw_response_bodies_stored"] is False for row in dataset["records"])
    assert all(row["parent_record_id"] for row in dataset["records"])
    assert sum(row["step"] == "counterfactual" for row in dataset["records"]) == 8
    assert sum(row["step"] == "repair" for row in dataset["records"]) == 8
    assert sum(row["step"] == "replay" for row in dataset["records"]) == 8
    assert all(row["failure_kind"] == "candidate_no_effect" for row in dataset["records"] if row["step"] == "counterfactual")
    assert all(row["repair_delta"] == {"from": "counterfactual_no_effect", "to": "typed_effect"} for row in dataset["records"] if row["step"] in {"repair", "replay"})


def test_pg245_2048_adapter_holdout_measures_repair_not_only_next_token() -> None:
    report = _read("pg244_failure_repair_capacity_training_report_v1.json")
    holdout = report["selected"]["metrics"]["seed_holdout"]
    assert report["status"] == "completed_multistep_failure_repair_capacity_training"
    assert report["device"] == "cuda"
    assert report["holdout_contract"]["pg244_seed_24402_never_in_training"] is True
    assert report["holdout_contract"]["pg242_xss_seed_24202_never_in_training"] is True
    assert report["counts"]["holdout_rows"] == 14
    assert report["counts"]["holdout_family_counts"] == {"dom_surface": 8, "sql_surface": 6}
    assert report["selected"]["hidden_dim"] == 2048
    assert holdout["next_token_accuracy"] == 0.89264706
    assert holdout["repair_accuracy"] == 0.92857146
    assert holdout["positive_send_recall"] == 1.0
    assert holdout["abstain_recall"] == 1.0
    assert holdout["false_send_count"] == 0
    assert holdout["missed_send_count"] == 0
    assert report["safety_abstain_gate_pass"] is True
    assert report["capability_gate_pass"] is True
    assert report["promotion"]["training_promotion_allowed"] is False


def test_pg244_registry_and_rules_keep_capacity_diagnostic_only() -> None:
    registry = _read("pg_pk_24_cross_lab_registry_v1.json")
    rules = _read("improvement_rules.json")
    trajectory = next(item for item in registry["targets"] if item["target_id"] == "pg244_failure_repair_trajectory")
    capacity = next(item for item in registry["targets"] if item["target_id"] == "pg245_failure_repair_capacity_training")
    assert trajectory["training_eligible"] is True
    assert trajectory["gold_count"] == 16
    assert trajectory["hard_negative_count"] == 8
    assert capacity["training_completed"] is True
    assert capacity["training_eligible"] is False
    assert capacity["holdout_repair_accuracy"] == 0.92857146
    assert capacity["selected_hidden_dim"] == 2048
    assert registry["training_eligible_target_count"] == 40
    assert rules["pg244_failure_repair_trajectory"]["fresh_container_count"] == 16
    assert rules["pg245_failure_repair_capacity_training"]["holdout_repair_accuracy"] == 0.92857146
