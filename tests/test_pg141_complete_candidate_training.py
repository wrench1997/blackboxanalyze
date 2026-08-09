from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg141_causal_pretraining_gains_action_but_fails_raw_safety_gate():
    report = _load("pg141_complete_candidate_training_report_v1.json")
    assert report["status"] == "completed_pg141_complete_candidate_training"
    assert report["scope"]["device"] == "cuda"
    assert report["training_eligible"] is False
    assert report["checks"]["candidate_only_training"] is True
    assert report["checks"]["labels_separate_from_model_rows"] is True
    assert report["checks"]["action_gain_both_loio_folds"] is False
    assert report["checks"]["unknown_abstain_floor"] is False
    assert report["checks"]["raw_safety_floor"] is False
    assert report["checks"]["guarded_safety_floor"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["folds"]["holdout_pg127"]["selected_variant"] == "causal_pretrained_action"
    assert report["folds"]["holdout_pg125"]["selected_variant"] == "scratch_action"
    pg127 = report["folds"]["holdout_pg127"]["variants"]["causal_pretrained_action"]["holdouts"]
    pg125 = report["folds"]["holdout_pg125"]["variants"]["scratch_action"]["holdouts"]
    assert pg127["pg127"]["raw"]["accuracy"] == 0.708333
    assert pg127["pg127"]["raw"]["safety_compliance_rate"] == 0.708333
    assert pg127["pg127"]["raw"]["unknown_abstain_rate"] == 0.666667
    assert pg127["pg122"]["raw"]["safety_compliance_rate"] == 0.625
    assert pg125["pg125"]["raw"]["safety_compliance_rate"] == 0.625
    assert pg125["pg122"]["raw"]["safety_compliance_rate"] == 0.625
    assert pg127["pg127"]["guarded"]["safety_compliance_rate"] == 0.916667
    assert pg125["pg125"]["guarded"]["safety_compliance_rate"] == 0.9375
    assert report["candidate_source"]["unknown_rows_excluded_from_action_supervision"] is True


def test_pg141_report_trace_protocol_and_checkpoints_are_hashed_and_local_only():
    report = _load("pg141_complete_candidate_training_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    trace = _load("pg141_complete_candidate_training_trace_v1.json")
    declared = trace.pop("trace_sha256")
    actual = hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    protocol = _load("pg141_complete_candidate_training_protocol_v1.json")
    assert protocol["required_gates"]["candidate_only_training"] is True
    assert protocol["required_gates"]["raw_safety_floor"] is False
    assert protocol["promotion"]["memory_promotion_allowed"] is False
    for checkpoint in (
        "holdout_pg127_scratch_action.pt",
        "holdout_pg127_causal_pretrained_action.pt",
        "holdout_pg127_joint_lm_action.pt",
        "holdout_pg125_scratch_action.pt",
        "holdout_pg125_causal_pretrained_action.pt",
        "holdout_pg125_joint_lm_action.pt",
    ):
        assert (ROOT / "artifacts" / "pg141-complete-candidate-v1" / checkpoint).exists()

    text = json.dumps(_load("pg141_complete_candidate_training_report_v1.json"), ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text


def test_pg141_registry_and_rule_are_evaluation_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg141_complete_candidate_training")
    assert registry["evaluation_only_target_count"] == 116
    assert target["training_eligible"] is False
    assert target["action_gain_both_loio_folds"] is False
    assert target["raw_safety_floor"] is False
    assert target["memory_promotion_allowed"] is False
    rules = _load("improvement_rules.json")
    policy = rules["pg141_complete_candidate_training"]
    assert policy["action_gain_both_loio_folds"] is False
    assert policy["raw_safety_floor"] is False
    assert policy["training_eligible"] is False
    assert policy["memory_promotion_allowed"] is False
