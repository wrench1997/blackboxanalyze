from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg144_expands_representation_data_without_claiming_capability():
    report = _load("pg144_surface_counterfactual_report_v1.json")
    assert report["status"] == "completed_pg144_surface_counterfactual_representation_only"
    assert report["hard_gates_passed"] is False
    assert report["representation_pretrain_allowed"] is False
    assert report["representation_diagnostic_only"] is True
    assert report["training_eligible"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["summary"]["base_row_count"] == 840
    assert report["summary"]["augmented_row_count"] == 13440
    assert report["summary"]["variant_count"] == 16
    assert report["summary"]["changed_surface_pair_count"] == 13440
    assert report["summary"]["unique_surface_delta_count"] == 144
    assert report["summary"]["variant_identity_in_model_input_count"] == 0
    assert report["summary"]["unique_sequence_count"] == 992
    assert report["summary"]["unique_sequence_density"] == 992 / 13440
    assert report["summary"]["duplicate_sequence_count"] == 12448
    assert report["summary"]["surface_diversity_gate"] is False
    assert report["summary"]["oracle_availability_counts"] == {"typed": 10080, "unknown": 3360}
    assert report["surface_diversity_gate"] is False
    assert report["unique_sequence_density"] == 992 / 13440
    assert report["duplicate_sequence_count"] == 12448
    assert report["gates"]["oracle_availability_preserved"] is True
    assert report["gates"]["same_split_parent_binding"] is True
    assert report["gates"]["counterfactual_not_authority"] is True
    assert report["gates"]["variant_identity_not_in_model_input"] is True
    assert report["gates"]["surface_diversity_gate"] is False
    assert report["promotion"]["representation_pretrain_allowed"] is False
    assert report["promotion"]["representation_diagnostic_only"] is True
    assert report["promotion"]["capability_train_allowed"] is False


def test_pg144_dataset_and_trace_hashes_are_stable_and_raw_free():
    dataset = _load("pg144_surface_counterfactual_model_dataset_v1.json")
    declared = dataset.pop("dataset_sha256")
    actual = hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert dataset["raw_source_retained"] is False
    assert dataset["raw_probe_response_retained"] is False
    assert dataset["evaluator_labels_retained"] is False
    assert dataset["variant_identity_in_model_input"] is False
    assert dataset["representation_pretrain_allowed"] is False
    assert dataset["representation_diagnostic_only"] is True
    assert dataset["summary"]["unique_sequence_count"] == 992
    assert dataset["summary"]["unique_sequence_density"] == 992 / 13440
    assert dataset["summary"]["duplicate_sequence_count"] == 12448
    assert dataset["summary"]["surface_diversity_gate"] is False
    assert all(row["action_supervision_allowed"] is False for row in dataset["rows"])
    assert all(row["safety_supervision_allowed"] is False for row in dataset["rows"])
    assert all(row["memory_promotion_allowed"] is False for row in dataset["rows"])
    text = json.dumps(dataset, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text
    trace = _load("pg144_surface_counterfactual_trace_v1.json")
    declared = trace.pop("trace_sha256")
    actual = hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert trace["raw_content_absent"] is True
    assert trace["variant_identity_not_in_model_input"] is True
    assert trace["action_supervision_allowed"] is False
    assert trace["representation_pretrain_allowed"] is False
    assert trace["representation_diagnostic_only"] is True
    assert trace["unique_sequence_density"] == 992 / 13440
    assert trace["duplicate_sequence_count"] == 12448
    assert trace["surface_diversity_gate"] is False


def test_pg144_registry_and_rule_remain_representation_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg144_surface_counterfactual_catalog")
    assert registry["evaluation_only_target_count"] == 116
    assert target["training_eligible"] is False
    assert target["representation_pretrain_allowed"] is False
    assert target["variant_identity_not_in_model_input"] is True
    assert target["capability_train_allowed"] is False
    assert target["memory_promotion_allowed"] is False
    rules = _load("improvement_rules.json")
    policy = rules["pg144_surface_counterfactual_catalog"]
    assert policy["training_eligible"] is False
    assert policy["representation_pretrain_allowed"] is False
    assert policy["variant_identity_not_in_model_input"] is True
    assert policy["capability_train_allowed"] is False
    assert policy["memory_promotion_allowed"] is False
