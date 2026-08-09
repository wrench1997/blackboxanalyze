import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg170_cross_generator_ood_has_zero_overlap_and_learns_new_syntax_axis():
    report = _load("pg170_cross_generator_report_v1.json")
    dataset = _load("pg170_cross_generator_dataset_v1.json")
    assert report["status"] == "completed_pg170_cross_generator_ood"
    assert report["dataset"]["train_projection_overlap_prior"] == 0
    assert report["dataset"]["ood_projection_overlap_prior"] == 0
    assert report["dataset"]["dev_ood_projection_overlap"] == 0
    assert report["cross_generator"]["generator_ood"]["perplexity"] == 2.48539371
    assert report["cross_generator"]["generator_ood"]["perplexity"] < report["baseline"]["generator_ood"]["perplexity"]
    assert report["interpretation"]["cross_generator_ood_isolated"] is True
    assert report["interpretation"]["vulnerability_claim_allowed"] is False
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False
    assert dataset["training_contract"]["oracle_labels_stored"] is False
    assert dataset["training_contract"]["family_labels_stored"] is False


def test_pg170_registry_is_cross_generator_diagnostic_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg170_cross_generator_ood")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "cross_generator_ood_diagnostic"
    assert entry["ood_projection_overlap_prior"] == 0
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
