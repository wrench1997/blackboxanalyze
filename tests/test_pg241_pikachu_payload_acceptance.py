import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg241_real_get_post_ai_reference_acceptance_is_not_shape_only() -> None:
    report = _read("pg241_pikachu_payload_acceptance_report_v1.json")
    counts = report["counts"]
    assert report["status"] == "completed_local_get_post_ai_reference_payload_acceptance"
    assert report["device"] == "cuda"
    assert counts["fresh_container_count"] == 14
    assert counts["get_episode_count"] == 10
    assert counts["post_episode_count"] == 4
    assert counts["database_health_gate_count"] == 14
    assert counts["ai_send_count"] == 12
    assert counts["reference_send_count"] == 12
    assert counts["confirmed_positive_count"] == 12
    assert counts["reference_positive_count"] == 12
    assert counts["negative_clean_count"] == 14
    assert counts["forbidden_timing_abstain_count"] == 2
    assert counts["false_positive_count"] == 0
    assert all(row["typed_oracle"]["confirmed_positive"] for row in report["results"] if row["route"] != "/vul/sqli/sqli_blind_t.php")
    assert all(row["ai"]["sent"] and row["reference"]["sent"] for row in report["results"] if row["route"] != "/vul/sqli/sqli_blind_t.php")
    assert all(not row["ai"]["sent"] and not row["reference"]["sent"] for row in report["results"] if row["route"] == "/vul/sqli/sqli_blind_t.php")


def test_pg241_dataset_is_redacted_and_gold_is_process_grounded() -> None:
    report_text = (ROOT / "research" / "pg241_pikachu_payload_acceptance_report_v1.json").read_text(encoding="utf-8-sig")
    dataset_text = (ROOT / "research" / "pg241_pikachu_payload_acceptance_dataset_v1.json").read_text(encoding="utf-8-sig")
    dataset = json.loads(dataset_text)
    assert dataset["counts"] == {"records": 14, "gold": 12, "hard_negative": 2, "silver": 0, "quarantine": 0}
    assert dataset["contract"]["ai_participates_in_send"] is True
    assert dataset["contract"]["typed_result_shape_required"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert "1 OR 1=1" not in report_text
    assert "%bf%27" not in report_text
    assert "1 OR 1=1" not in dataset_text
    assert "%bf%27" not in dataset_text


def test_pg241_registry_and_rule_point_to_same_artifacts() -> None:
    registry = _read("pg_pk_24_cross_lab_registry_v1.json")
    rule = _read("improvement_rules.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg241_pikachu_payload_acceptance")
    assert target["confirmed_positive_count"] == 12
    assert target["training_eligible"] is True
    assert target["memory_promotion_allowed"] is False
    assert registry["training_eligible_target_count"] == 40
    policy = rule["pg241_pikachu_payload_acceptance"]
    assert policy["confirmed_positive_count"] == 12
    assert policy["timing_abstain_count"] == 2
    assert policy["raw_payload_strings_stored"] is False


def test_pg241_capacity_training_keeps_false_send_gate_closed() -> None:
    report = _read("pg241_payload_capacity_training_report_v1.json")
    holdout = report["selected"]["metrics"]["seed_holdout"]
    assert report["status"] == "completed_grounded_payload_trace_capacity_training"
    assert report["device"] == "cuda"
    assert report["selected"]["hidden_dim"] == 1024
    assert holdout["next_token_accuracy"] == 0.81302521
    assert holdout["positive_send_recall"] == 1.0
    assert holdout["abstain_recall"] == 1.0
    assert holdout["false_send_count"] == 0
    assert holdout["missed_send_count"] == 0
    assert report["safety_abstain_gate_pass"] is True
    assert report["capability_gate_pass"] is True
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
