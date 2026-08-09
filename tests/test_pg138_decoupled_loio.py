from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.pg138_safety_head import DecoupledCausalSafetyPolicy, DecoupledSafetyHead, SCHEMA_VERSION


def test_pg138_head_is_explicitly_decoupled() -> None:
    assert SCHEMA_VERSION == "pg138-decoupled-safety-head-v1"
    assert DecoupledSafetyHead(64).action_classifier.out_features == 7
    assert DecoupledCausalSafetyPolicy


def test_pg138_loio_keeps_raw_masked_and_calibrated_separate() -> None:
    report = json.loads(Path("research/pg138_decoupled_loio_report_v1.json").read_text(encoding="utf-8"))
    assert report["hard_gates_passed"] is False
    assert report["training_eligible"] is False
    assert report["input_contract"]["leave_one_implementation_out"] is True
    assert report["input_contract"]["typed_contract_mask_separate"] is True
    assert report["checks"]["loio_unknown_abstain"] is True
    assert report["checks"]["loio_nontrivial_action_rate"] is True
    fold = report["folds"]["holdout_pg127"]
    selected = fold["selection"]["selected_variant"]
    metrics = fold["variants"][selected]["holdouts"]["pg127"]
    assert metrics["raw"]["safety_compliance_rate"] == 0.75
    assert metrics["contract_masked"]["safety_compliance_rate"] == 1.0
    assert metrics["contract_override_rate"] == 0.25
    assert metrics["calibrated"]["safety_compliance_rate"] == 0.916667
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg138_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg138_decoupled_loio_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
