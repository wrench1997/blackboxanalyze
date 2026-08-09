import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg168_adds_collision_free_abstract_slots_and_learns_slot_ood():
    report = _load("pg168_discriminative_slot_report_v1.json")
    dataset = _load("pg168_discriminative_slot_dataset_v1.json")
    assert report["status"] == "completed_pg168_discriminative_slot_augmentation"
    assert report["dataset"]["slot_train_count"] == 8000
    assert report["dataset"]["slot_dev_count"] == 1000
    assert report["dataset"]["slot_ood_count"] == 1000
    assert report["dataset"]["projection_overlap_dev_ood"] == 0
    assert report["variants"]["replay_only"]["slot_ood"]["perplexity"] == 471.30196493
    assert report["variants"]["slot_augmented"]["slot_ood"]["perplexity"] == 2.38018788
    assert report["variants"]["slot_augmented"]["slot_ood"]["perplexity"] < report["variants"]["replay_only"]["slot_ood"]["perplexity"]
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False
    assert dataset["training_contract"]["oracle_labels_stored"] is False
    assert dataset["training_contract"]["family_labels_stored"] is False
    token_text = json.dumps(dataset["rows"], ensure_ascii=False).lower()
    assert "payload" not in token_text
    assert "vulnerability" not in token_text


def test_pg168_registry_blocks_capability_promotion_until_replay_tuning():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg168_discriminative_slot_augmentation")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "abstract_slot_information_gain_diagnostic"
    assert entry["projection_overlap_dev_ood"] == 0
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
