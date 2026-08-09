import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg177_expands_data_and_keeps_two_seed_continued_gate():
    report = _load("pg177_data_capacity_report_v1.json")
    dataset = _load("pg177_data_capacity_dataset_v1.json")
    assert report["status"] == "completed_pg177_data_capacity_sweep"
    assert report["dataset"]["train_row_count"] == 7200
    assert report["dataset"]["projection_overlap_prior"] == 0
    assert len(report["gates"]) == 2
    assert all(gate["pass"] for gate in report["gates"])
    assert report["selection"]["selected_variant"] == "160m_continued"
    assert report["selection"]["promotion_allowed"] is True
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert dataset["counts"]["new_row_count"] == 7200
    assert len(dataset["rows"]) == 7200
    assert dataset["projection_overlap_prior"] == 0
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False
    assert dataset["training_contract"]["oracle_labels_stored"] is False
    assert dataset["training_contract"]["memory_promotion_allowed"] is False


def test_pg177_reports_capacity_result_without_overclaiming():
    report = _load("pg177_data_capacity_report_v1.json")
    assert {variant["variant"] for variant in report["variants"]} == {"160m_continued", "160m_scratch", "200m_scratch"}
    assert {variant["parameter_count"] for variant in report["variants"] if variant["variant"] == "200m_scratch"} == {197537513}
    assert all(item["200m_better"] is False for item in report["capacity_comparison"])
    assert report["safety"]["raw_payloads_in_model"] is False
    assert report["safety"]["raw_responses_in_model"] is False
    assert report["safety"]["vulnerability_labels_in_model"] is False
    assert report["safety"]["memory_promotion_allowed"] is False


def test_pg177_registry_records_data_and_capacity_candidates():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg177_data_capacity_sweep")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_artifact_promotion_allowed"] is True
    assert entry["memory_promotion_allowed"] is False
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["capacity_200m_better_on_new_ood"] is False
