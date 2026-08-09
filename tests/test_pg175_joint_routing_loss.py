import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg175_routed_low_lr_beats_full_baseline_with_split_gate():
    report = _load("pg175_joint_routing_loss_report_v1.json")
    dataset = _load("pg175_joint_routing_loss_dataset_v1.json")
    assert report["status"] == "completed_pg175_joint_routing_loss_search"
    assert report["dataset"]["full_eval_counts"] == {"base_holdout": 1210, "typed_holdout": 203, "pg168_ood": 1000, "pg170_ood": 1000, "pg172_ood": 1000}
    assert report["baseline_aggregate_ppl"] == 2.51077335
    assert report["selection"]["selected_variant"] == "routed_low_lr"
    assert report["strict_gate"]["routed_low_lr"] is True
    assert report["aggregate_ppl"]["routed_low_lr"] == 2.43481149
    assert report["aggregate_ppl"]["routed_low_lr"] < report["baseline_aggregate_ppl"]
    assert report["selection"]["promotion_allowed"] is True
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert dataset["training_contract"]["raw_payloads_stored"] is False
    assert dataset["training_contract"]["raw_responses_stored"] is False
    assert dataset["training_contract"]["vulnerability_labels_stored"] is False
    assert dataset["training_contract"]["oracle_labels_stored"] is False
    assert dataset["training_contract"]["family_labels_stored"] is False


def test_pg175_registry_allows_only_selected_training_candidate():
    registry = _load("research/pg_pk_24_cross_lab_registry_v1.json") if Path("research/research/pg_pk_24_cross_lab_registry_v1.json").exists() else _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg175_joint_routing_loss_search")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "full_holdout_positive_routing_loss_diagnostic"
    assert entry["selected_variant"] == "routed_low_lr"
    assert entry["training_artifact_promotion_allowed"] is True
    assert entry["memory_promotion_allowed"] is False
    assert entry["vulnerability_claim_allowed"] is False
