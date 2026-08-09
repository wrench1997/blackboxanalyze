from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg142_independent_safety_head_fails_closed_as_a_negative_result():
    report = _load("pg142_safety_aware_candidate_report_v1.json")
    assert report["status"] == "completed_pg142_safety_aware_candidate"
    assert report["training_eligible"] is False
    assert report["checks"]["candidate_only_training"] is True
    assert report["checks"]["raw_safety_floor"] is False
    assert report["checks"]["value_safety_floor"] is False
    assert report["checks"]["guarded_safety_floor"] is False
    assert report["checks"]["parser_ood_safety_floor"] is False
    assert report["checks"]["unknown_abstain_floor"] is True
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["summary_metrics"]["raw_safety_min"] == 0.625
    assert report["summary_metrics"]["value_safety_min"] == 0.625
    assert report["summary_metrics"]["guarded_safety_min"] == 0.875
    assert report["summary_metrics"]["parser_value_safety_min"] == 0.25
    assert report["summary_metrics"]["unknown_abstain_min"] == 1.0
    assert report["summary_metrics"]["value_override_total"] == 0
    assert report["input_contract"]["typed_contract_used_as_safety_target_only"] is True
    assert report["input_contract"]["typed_contract_not_in_model_input"] is True


def test_pg142_artifacts_are_hashed_and_raw_content_is_not_saved():
    report = _load("pg142_safety_aware_candidate_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    trace = _load("pg142_safety_aware_candidate_trace_v1.json")
    declared = trace.pop("trace_sha256")
    actual = hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    protocol = _load("pg142_safety_aware_candidate_protocol_v1.json")
    assert protocol["required_gates"]["raw_safety_floor"] is False
    assert protocol["promotion"]["memory_promotion_allowed"] is False
    text = json.dumps(_load("pg142_safety_aware_candidate_report_v1.json"), ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text
    for variant in ("scratch_safety", "pretrained_safety", "frozen_safety", "joint_safety"):
        for fold in ("holdout_pg127", "holdout_pg125"):
            assert (ROOT / "artifacts" / "pg142-safety-aware-candidate-v1" / f"{fold}_{variant}.pt").exists()


def test_pg142_registry_and_rule_preserve_the_negative_result():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg142_safety_aware_candidate")
    assert registry["evaluation_only_target_count"] == 116
    assert target["training_eligible"] is False
    assert target["raw_safety_floor"] is False
    assert target["unknown_abstain_floor"] is True
    assert target["memory_promotion_allowed"] is False
    rules = _load("improvement_rules.json")
    policy = rules["pg142_safety_aware_candidate"]
    assert policy["training_eligible"] is False
    assert policy["raw_safety_floor"] is False
    assert policy["memory_promotion_allowed"] is False
