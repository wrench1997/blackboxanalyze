from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_pg333_capability_sft_compat import (
    REQUIRED_TARGET_SLOTS,
    audit_pg333_capability_sft_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pg333_is_fail_closed_for_capability_sft() -> None:
    before = _file_sha(DATASET)
    report = audit_pg333_capability_sft_compatibility()
    after = _file_sha(DATASET)

    assert before == after
    assert report["status"] == "blocked_not_capability_sft_compatible"
    assert report["scope"] == {
        "dataset_schema": "pg331-cross-implementation-diagnostic-merge-v1",
        "diagnostic_only": True,
        "raw_material_reported": False,
        "split_relabelled": False,
        "fields_fabricated": False,
    }
    assert report["candidate_runner"]["compatible"] is False
    assert report["candidate_runner"]["remote_candidate_allowed"] is False
    assert report["candidate_runner"]["remote_command"] is None
    assert all(value is False for value in report["promotion"].values())


def test_pg333_target_schema_and_train_holdout_firewall_are_explicit() -> None:
    report = audit_pg333_capability_sft_compatibility()
    target = report["target_schema"]
    split = report["train_only_holdout"]

    assert target["required_slot_count"] == 13
    assert tuple(target["required_slots"]) == REQUIRED_TARGET_SLOTS
    assert target["expected_target_token_length"] == 15
    assert target["target_token_length_counts"] == {"10": 45}
    assert target["missing_required_slots"] == [
        "ask_reason",
        "syntax_category_ref",
        "payload_shape_ref",
        "oracle_ref",
        "negative_control_presence_ref",
    ]
    assert split["split_counts"] == {"implementation_holdout": 36, "train": 9}
    assert split["train_rows"] == 9
    assert split["holdout_rows"] == 36
    assert split["implementation_group_counts"] == {"holdout": 2, "train": 1}
    assert split["family_group_counts"] == {"holdout": 3, "train": 1}
    assert split["source_audit_group_disjoint"] is True
    assert split["train_only_vocab_closed"] is False
    assert split["holdout_unknown_context_count"] == 159
    assert split["accepted_training_eligible_rows"] == 0


def test_pg333_capacity_and_existing_runner_compatibility_are_blocked() -> None:
    report = audit_pg333_capability_sft_compatibility()
    compatibility = report["compatibility"]

    assert compatibility["candidate_runner_compatible"] is False
    assert compatibility["pg370"]["max_length"] == 768
    assert compatibility["pg370"]["required_window"] == 4145
    assert compatibility["pg370"]["capacity_pass"] is False
    assert compatibility["pg375"]["context_smoke_window"] == 606
    assert compatibility["pg375"]["capacity_pass"] is False
    assert "pg375_plan_blocked_data_contract" in compatibility["blocked_reasons"]
    assert "information_audit_not_passed" in compatibility["blocked_reasons"]
    assert "target_schema_missing_required_slots" in report["blocked_reasons"]


def test_projection_contains_no_raw_wire_or_payload_material() -> None:
    report = audit_pg333_capability_sft_compatibility()
    encoded = json.dumps(report, ensure_ascii=False).lower()

    # Slot names are abstract schema labels; raw request/response material is
    # not allowed in this projection.
    for fragment in ("http://", "https://", "/webgoat", "response_body=", "raw_payload="):
        assert fragment not in encoded
    assert "wire" not in encoded
    assert "canary" not in encoded
    assert "payload_shape_ref" in encoded


def test_missing_artifact_is_pending_without_remote_side_effects(tmp_path: Path) -> None:
    report = audit_pg333_capability_sft_compatibility(dataset_path=tmp_path / "missing.json")

    assert report["status"] == "blocked_missing_artifact"
    assert report["missing_artifacts"] == ["dataset"]
    assert report["candidate_runner"] == {
        "compatible": False,
        "remote_candidate_allowed": False,
        "remote_command": None,
    }
    assert all(value is False for value in report["promotion"].values())

