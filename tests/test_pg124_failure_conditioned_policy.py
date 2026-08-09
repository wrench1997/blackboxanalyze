from __future__ import annotations

import json
from pathlib import Path

from app.pg124_failure_conditioned_policy import FAILURE_FEATURE_DIM, FEATURE_DIM, failure_feedback_vector


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg124_failure_token_policy_ablation_passes_without_memory_promotion() -> None:
    report = _load("pg124_failure_conditioned_policy_report_v1.json")
    assert report["status"] == "completed_pg124_failure_token_policy_ablation"
    assert report["scope"]["feature_dim"] == FEATURE_DIM == 69
    assert report["scope"]["failure_feature_dim"] == FAILURE_FEATURE_DIM == 17
    assert report["holdout"]["full_failure_input"]["metrics"]["accuracy"] == 1.0
    assert report["holdout"]["full_failure_input"]["safety_compliance_rate"] == 1.0
    assert report["holdout"]["full_model_failure_zeroed"]["metrics"]["accuracy"] == 0.527778
    assert report["holdout"]["fresh_no_failure_baseline"]["metrics"]["accuracy"] == 0.854167
    assert report["holdout"]["failure_slot_behavior_changed"] is True
    assert all(report["checks"].values())
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg124_failure_vector_masks_oracle_authority_fields() -> None:
    signature = {
        "schema_version": "sift-failure-signature-v1",
        "kind": "oracle_unavailable",
        "failed_gate": "typed_effect",
        "observed_method": "GET",
        "methods_seen": ["GET"],
        "candidate_signal": True,
        "typed_available": False,
        "positive_authority": False,
        "next_action": "replay_other_method",
    }
    changed = dict(signature, typed_available=True, positive_authority=True)
    assert failure_feedback_vector(signature) == failure_feedback_vector(changed)


def test_pg124_protocol_keeps_get_post_and_unknown_holdout() -> None:
    protocol = _load("pg124_failure_conditioned_policy_protocol_v1.json")
    assert protocol["model_input"]["failure_slots"] == 17
    assert set(protocol["model_input"]["masked_fields"]) >= {"positive_authority", "typed_available", "evidence_hash", "raw_probe", "raw_response"}
    assert protocol["holdout"]["get"] == protocol["holdout"]["post"] == 72
    assert protocol["promotion"]["memory_promotion_allowed"] is False
