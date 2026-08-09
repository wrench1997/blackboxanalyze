import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg242_browser_dom_acceptance_is_dual_channel_and_loopback_guarded() -> None:
    report = _read("pg242_pikachu_xss_dom_acceptance_report_v1.json")
    counts = report["counts"]
    assert report["status"] == "completed_local_xss_dom_browser_dual_channel"
    assert report["device"] == "cuda"
    assert counts["fresh_container_count"] == 16
    assert counts["preflight_count"] == 4
    assert counts["get_episode_count"] == 14
    assert counts["post_episode_count"] == 6
    assert counts["ai_send_count"] == 16
    assert counts["reference_send_count"] == 16
    assert counts["confirmed_positive_count"] == 12
    assert counts["negative_control_confirmed_count"] == 2
    assert counts["oracle_gap_count"] == 2
    assert counts["false_positive_count"] == 0
    assert counts["model_missed_positive_count"] == 0
    assert counts["external_network_count"] == 0
    assert counts["external_request_blocked_count"] > 0

    active = [row for row in report["results"] if row["fresh_reset"]]
    positives = [row for row in active if row["typed_oracle"]["confirmed_positive"]]
    negatives = [row for row in active if row["typed_oracle"].get("negative_control_confirmed")]
    oracle_gaps = [row for row in active if row["failure_kind"] == "oracle_unavailable"]
    assert len(positives) == 12
    assert len(negatives) == 2
    assert len(oracle_gaps) == 2
    assert all(row["ai"]["sent"] and row["reference"]["sent"] for row in active)
    assert all(row["typed_oracle"]["evidence_hash"] for row in active)
    assert all(not (row["ai"].get("response") or {}).get("external_network", False) for row in active)
    assert all(not (row["reference"].get("response") or {}).get("external_network", False) for row in active)
    assert all(row["typed_oracle"]["oracle_available"] is False for row in oracle_gaps)


def test_pg242_dataset_keeps_success_failure_oracle_gap_without_raw_payloads() -> None:
    report_text = (ROOT / "research" / "pg242_pikachu_xss_dom_acceptance_report_v1.json").read_text(encoding="utf-8-sig")
    dataset_text = (ROOT / "research" / "pg242_pikachu_xss_dom_acceptance_dataset_v1.json").read_text(encoding="utf-8-sig")
    dataset = json.loads(dataset_text)
    assert dataset["counts"] == {"records": 16, "gold": 12, "hard_negative": 2, "silver": 2, "quarantine": 0}
    assert dataset["contract"]["controlled_browser_dom_oracle"] is True
    assert dataset["contract"]["ai_participates_in_send"] is True
    assert dataset["contract"]["independent_reference_required"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert "xss-get-ai" not in report_text
    assert "<svg" not in report_text
    assert "xss-get-ai" not in dataset_text
    assert "<svg" not in dataset_text
    assert all(row["parent_record_id"].startswith("pg242:") for row in dataset["records"])
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["records"])


def test_pg242_reports_href_protocol_oracle_gap_instead_of_negative_label() -> None:
    report = _read("pg242_pikachu_xss_dom_acceptance_report_v1.json")
    rows = [row for row in report["results"] if row["route"] == "/vul/xss/xss_03.php" and row["fresh_reset"]]
    assert len(rows) == 2
    for row in rows:
        assert row["failure_kind"] == "oracle_unavailable"
        assert row["typed_oracle"]["confirmed_positive"] is False
        assert row["typed_oracle"]["oracle_mode"] == "href_protocol_execution_unobserved"
        assert "browser_javascript_href_execution_not_observed" in row["typed_oracle"]["reasons"]
        assert row["training_eligible"] is False


def test_pg243_family_holdout_exposes_repair_gap_without_opening_send_gate() -> None:
    report = _read("pg242_sql_xss_family_capacity_training_report_v1.json")
    holdout = report["selected"]["metrics"]["seed_holdout"]
    assert report["status"] == "completed_sql_xss_family_holdout_capacity_training"
    assert report["device"] == "cuda"
    assert report["holdout_contract"]["pg242_all_xss_seeds_never_in_training"] is True
    assert report["holdout_contract"]["family_holdout_is_disjoint"] is True
    assert report["selected"]["hidden_dim"] == 512
    assert report["counts"]["holdout_family_counts"] == {"sql_surface": 7, "dom_surface": 16}
    assert holdout["positive_send_recall"] == 1.0
    assert holdout["abstain_recall"] == 1.0
    assert holdout["false_send_count"] == 0
    assert holdout["missed_send_count"] == 0
    assert holdout["next_token_accuracy"] == 0.69181586
    assert holdout["repair_accuracy"] == 0.39130434
    assert report["safety_abstain_gate_pass"] is True
    assert report["capability_gate_pass"] is True
    assert report["promotion"]["training_promotion_allowed"] is False


def test_pg242_rule_and_registry_record_family_holdout_and_repair_gap() -> None:
    registry = _read("pg_pk_24_cross_lab_registry_v1.json")
    rules = _read("improvement_rules.json")
    acceptance = next(item for item in registry["targets"] if item["target_id"] == "pg242_pikachu_xss_dom_acceptance")
    capacity = next(item for item in registry["targets"] if item["target_id"] == "pg242_sql_xss_family_capacity_holdout")
    assert acceptance["confirmed_positive_count"] == 12
    assert acceptance["external_network_count"] == 0
    assert acceptance["training_eligible"] is True
    assert capacity["training_completed"] is True
    assert capacity["holdout_repair_accuracy"] == 0.39130434
    assert capacity["training_artifact_promotion_allowed"] is False
    assert registry["training_eligible_target_count"] == 40
    assert rules["pg242_pikachu_xss_dom_acceptance"]["oracle_gap_count"] == 2
    assert rules["pg243_sql_xss_family_capacity_training"]["holdout_next_token_accuracy"] == 0.69181586
    assert rules["pg243_sql_xss_family_capacity_training"]["holdout_repair_accuracy"] == 0.39130434
