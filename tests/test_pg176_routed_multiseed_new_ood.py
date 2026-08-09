import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg176_dataset_and_three_seed_gates_are_complete():
    report = _load("pg176_routed_multiseed_new_ood_report_v1.json")
    dataset = _load("pg176_fourth_generator_ood_dataset_v1.json")
    assert report["status"] == "completed_pg176_routed_multiseed_new_ood"
    assert report["dataset"]["train_row_count"] == 1000
    assert report["dataset"]["full_eval_counts"] == {"base_holdout": 1210, "typed_holdout": 203, "pg168_ood": 1000, "pg170_ood": 1000, "pg172_ood": 1000, "fourth_generator_ood": 1000}
    assert report["dataset"]["fourth_generator_projection_overlap_prior"] == 0
    assert report["selection"]["all_seeds_pass"] is True
    assert report["selection"]["promotion_allowed"] is True
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert len(report["seed_results"]) == 3
    assert all(result["aggregate_existing"] < report["baseline_existing_aggregate_ppl"] for result in report["seed_results"])
    assert all(result["fourth_generator_ood"]["perplexity"] < report["baseline"]["fourth_generator_ood"]["perplexity"] for result in report["seed_results"])
    assert dataset["row_count"] == 1000
    assert dataset["projection_overlap_prior"] == 0
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False
    assert dataset["training_contract"]["memory_promotion_allowed"] is False


def test_pg176_registry_marks_checkpoint_candidate_but_blocks_memory_and_vulnerability_claims():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg176_routed_multiseed_new_ood")
    assert registry["training_eligible_target_count"] == 40
    assert entry["training_eligible"] is True
    assert entry["all_seed_gates_pass"] is True
    assert entry["training_artifact_promotion_allowed"] is True
    assert entry["memory_promotion_allowed"] is False
    assert entry["vulnerability_claim_allowed"] is False
