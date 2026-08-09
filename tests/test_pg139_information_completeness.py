import hashlib
import json
from pathlib import Path

from app.information_completeness import (
    SCHEMA_VERSION,
    learning_stage,
    sha256_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg139_information_audit_is_fail_closed_but_does_not_freeze_representation_learning():
    audit = _load("pg139_information_completeness_audit_v1.json")
    assert audit["schema_version"] == SCHEMA_VERSION
    assert audit["hard_gate_passed"] is False
    assert audit["training_eligible"] is False
    assert audit["memory_promotion_allowed"] is False
    assert audit["sensitive_content_stored"] is False
    assert "dataset_companion_catalog_manifest_missing" in audit["blocking_reasons"]
    assert "report_row_level_evidence_manifest_missing" in audit["blocking_reasons"]
    assert audit["internal_trace"]["step_count"] == 696
    assert audit["internal_trace"]["critical_missing_field_count"] == 648
    assert audit["internal_trace"]["implicit_unknown_projection_fields"] == {
        "baseline_projection.shape_class": 408,
        "response_projection.scope_changed": 48,
        "response_projection.visibility_changed": 192,
    }
    assert audit["public_dataset"]["pretrain_row_count"] == 624
    assert audit["public_dataset"]["action_row_count"] == 624
    assert audit["public_dataset"]["dataset_manifest_hash_valid"] is True
    assert audit["public_dataset"]["visible_manifest_hash_valid"] is True
    assert audit["public_dataset"]["companion_catalog_complete"] is False
    assert audit["public_dataset"]["row_level_replay_auditable"] is False
    assert audit["public_dataset"]["trace_manifest_indexed"] is False
    assert audit["staged_learning_policy"]["incomplete_rows"]["stage"] == "representation_pretrain"
    assert audit["staged_learning_policy"]["incomplete_rows"]["action_supervision_allowed"] is False
    assert audit["staged_learning_policy"]["complete_replay_rows"]["stage"] == "capability_train"
    assert audit["staged_learning_policy"]["complete_replay_rows"]["memory_promotion_allowed"] is False


def test_information_gate_hash_and_stage_contracts_are_deterministic():
    audit = _load("pg139_information_completeness_audit_v1.json")
    stored = audit["audit_sha256"]
    without = dict(audit)
    without.pop("audit_sha256")
    assert stored == sha256_json(without)
    assert learning_stage(
        complete_trace=False,
        replayable=False,
        labels_verified=False,
        cross_split_clean=False,
    ) == {
        "stage": "representation_pretrain",
        "capability_training_allowed": False,
        "action_supervision_allowed": False,
        "memory_promotion_allowed": False,
    }
    assert learning_stage(
        complete_trace=True,
        replayable=True,
        labels_verified=True,
        cross_split_clean=False,
    )["stage"] == "capability_train"
    assert learning_stage(
        complete_trace=True,
        replayable=True,
        labels_verified=True,
        cross_split_clean=True,
    )["memory_promotion_allowed"] is True


def test_pg139_registry_registers_the_information_audit():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg139_value_head_loio")
    assert registry["evaluation_only_target_count"] == 116
    assert "pg139_information_completeness_audit_v1.json" in target["artifacts"]
    assert target["information_completeness_audit"]["hard_gate_passed"] is False
    assert target["information_completeness_audit"]["representation_pretrain_allowed"] is True
    assert target["information_completeness_audit"]["capability_training_allowed"] is False

    rules = _load("improvement_rules.json")
    policy = rules["information_completeness_gate"]
    assert policy["schema_version"] == "information-completeness-gate-v1"
    assert policy["staged_learning_policy"]["representation_pretrain"]["action_supervision_allowed"] is False
    assert policy["staged_learning_policy"]["capability_train"]["memory_promotion_allowed"] is False
    assert policy["current_pg139_audit"]["critical_missing_field_count"] == 648
