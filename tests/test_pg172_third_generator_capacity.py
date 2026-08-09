import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg172_third_generator_isolated_and_capacity_gain_not_assumed():
    report = _load("pg172_third_generator_capacity_report_v1.json")
    dataset = _load("pg172_third_generator_dataset_v1.json")
    assert report["status"] == "completed_pg172_third_generator_capacity"
    assert report["dataset"]["projection_overlap_third_prior"] == 0
    assert report["dataset"]["projection_overlap_third_dev_ood"] == 0
    small = report["variants"]["capacity_101m_scratch"]
    large = report["variants"]["capacity_160m_scratch"]
    assert small["parameter_count"] == 101380329
    assert large["parameter_count"] == 160089065
    assert small["third_generator_ood"]["perplexity"] == 2.3325
    assert large["third_generator_ood"]["perplexity"] == 2.37691395
    assert report["capacity_comparison"]["same_training_rows"] is True
    assert report["capacity_comparison"]["initialization"] == "from_scratch_for_both"
    assert report["interpretation"]["vulnerability_claim_allowed"] is False
    assert report["interpretation"]["promotion_allowed"] is False
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False


def test_pg172_registry_records_capacity_negative_result_without_promotion():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg172_third_generator_capacity")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "third_generator_capacity_causal_diagnostic"
    assert entry["capacity_gain_observed"] is False
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
