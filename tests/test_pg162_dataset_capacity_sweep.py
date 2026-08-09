import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg162_dataset_is_fresh_balanced_and_bounded():
    manifest = _load("pg162_fresh_typed_trace_manifest_v1.json")
    dataset = _load("pg162_fresh_typed_training_dataset_v1.json")
    assert manifest["row_count"] == 384
    assert manifest["train_row_count"] == 192
    assert manifest["dev_row_count"] == 192
    assert manifest["get_count"] == manifest["post_count"] == 192
    assert manifest["evidence_hash_valid"] is True
    assert manifest["fresh_reset_per_step"] is True
    assert manifest["raw_probe_strings_stored"] is False
    assert manifest["raw_response_bodies_stored"] is False
    assert manifest["family_in_model_input"] is False
    assert manifest["oracle_labels_in_model_input"] is False
    assert dataset["training_eligible"] is True
    assert dataset["memory_promotion_allowed"] is False
    rows = dataset["train_rows"] + dataset["dev_rows"]
    assert len(rows) == 384
    assert all(row["training_eligible"] is True for row in rows)
    assert all(row["memory_promotion_allowed"] is False for row in rows)
    assert all(len(row["evidence_sha256"]) == 64 for row in rows)
    assert all("probe" not in json.dumps(row["model_input"], ensure_ascii=False).lower() for row in rows)


def test_pg162_capacity_sweep_has_source_holdout_and_no_false_accept():
    report = _load("pg162_dataset_capacity_sweep_report_v1.json")
    assert report["status"] == "completed_pg162_dataset_capacity_sweep"
    assert report["dataset"]["row_count"] == 384
    assert report["dataset"]["get_count"] == report["dataset"]["post_count"] == 192
    assert report["dataset"]["evidence_hash_valid"] is True
    assert report["dataset"]["fresh_reset_per_step"] is True
    assert report["selection"]["best_source_heldout_candidate"] == "base"
    for name, result in report["model_variants"].items():
        assert result["dev"]["macro_f1"] == 1.0
        assert result["source_holdout"]["pg118_delta"]["episodes"]["decoy_false_accept_count"] == 0
        assert result["source_holdout"]["pg118_delta"]["episodes"]["positive_final_confirm_recall"] == 1.0
        assert result["source_holdout"]["pg118_delta"]["episodes"]["unknown_final_abstain_rate"] == 1.0
    assert report["pg146_real_container_evaluation_only"]["typed_oracle_count"] == 0
    assert report["pg146_real_container_evaluation_only"]["training_eligible"] is False
    assert report["pg146_real_container_evaluation_only"]["ready_count"] == 4
    assert report["pg146_real_container_evaluation_only"]["hard_gates_passed"] is False


def test_pg162_registry_entry_is_training_data_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg162_dataset_capacity_sweep")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
    assert entry["get_step_count"] == entry["post_step_count"] == 192
    assert entry["base_source_heldout_decoy_false_accept_count"] == 0
