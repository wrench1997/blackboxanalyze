import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg173_matches_epoch_plans_and_shows_budget_dependent_capacity_gain():
    report = _load("pg173_matched_budget_capacity_report_v1.json")
    dataset = _load("pg173_matched_budget_dataset_v1.json")
    assert report["status"] == "completed_pg173_matched_budget_capacity"
    assert report["comparison_contract"]["same_epoch_plans"] is True
    assert report["comparison_contract"]["same_token_budget_per_epoch"] is True
    assert report["dataset"]["rows_per_source_per_epoch"] == 250
    assert report["dataset"]["evaluation_rows_per_split"] == 128
    assert report["capacity_variants"]["101m_epoch4"]["typed_holdout"]["perplexity"] == 2.52596231
    assert report["capacity_variants"]["160m_epoch4"]["typed_holdout"]["perplexity"] == 2.45548802
    assert report["capacity_variants"]["160m_epoch4"]["pg172_ood"]["perplexity"] == 2.44197393
    assert report["capacity_variants"]["160m_epoch4"]["base_holdout"]["perplexity"] > report["capacity_variants"]["101m_epoch4"]["base_holdout"]["perplexity"]
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False
    assert dataset["training_contract"]["oracle_labels_stored"] is False
    assert dataset["training_contract"]["family_labels_stored"] is False


def test_pg173_registry_blocks_promotion_until_full_holdout_recheck():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg173_matched_budget_capacity")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "matched_budget_capacity_curve_diagnostic"
    assert entry["capacity_gain_requires_old_holdout"] is True
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
