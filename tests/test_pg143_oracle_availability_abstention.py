from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg143_availability_head_passes_coverage_but_not_safety_gate():
    report = _load("pg143_oracle_availability_abstention_report_v1.json")
    assert report["status"] == "completed_pg143_oracle_availability_abstention"
    assert report["scope"]["device"] == "cuda"
    assert report["training_eligible"] is False
    assert report["checks"]["all_train_representation_rows_allowed"] is True
    assert report["checks"]["action_supervision_complete_candidates_only"] is True
    assert report["checks"]["availability_supervision_no_authority"] is True
    assert report["checks"]["availability_unknown_recall_floor"] is True
    assert report["checks"]["availability_known_false_abstain_floor"] is True
    assert report["checks"]["value_safety_floor"] is False
    assert report["checks"]["guarded_safety_floor"] is False
    assert report["checks"]["parser_ood_value_safety_floor"] is False
    assert report["summary_metrics"] == {
        "availability_unknown_recall_min": 1.0,
        "availability_known_false_abstain_max": 0.0,
        "value_safety_min": 0.625,
        "guarded_safety_min": 0.916667,
        "parser_value_safety_min": 0.625,
    }
    assert [item["variant"] for item in report["selected_summary"]] == [
        "scratch_availability",
        "scratch_availability",
    ]
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg143_artifacts_are_hashed_and_keep_authority_out_of_inputs():
    report = _load("pg143_oracle_availability_abstention_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    trace = _load("pg143_oracle_availability_abstention_trace_v1.json")
    declared = trace.pop("trace_sha256")
    actual = hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    protocol = _load("pg143_oracle_availability_abstention_protocol_v1.json")
    assert protocol["required_gates"]["availability_supervision_no_authority"] is True
    assert protocol["promotion"]["memory_promotion_allowed"] is False
    report = _load("pg143_oracle_availability_abstention_report_v1.json")
    assert report["input_contract"]["positive_authority_in_model_input"] is False
    assert report["input_contract"]["raw_html_javascript_retained"] is False
    assert report["input_contract"]["raw_probe_response_retained"] is False
    text = json.dumps(report, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text
    for variant in ("scratch_availability", "pretrained_availability", "frozen_availability", "joint_availability"):
        for fold in ("holdout_pg127", "holdout_pg125"):
            assert (ROOT / "artifacts" / "pg143-oracle-availability-v1" / f"{fold}_{variant}.pt").exists()


def test_pg143_registry_and_rule_preserve_evaluation_only_status():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg143_oracle_availability_abstention")
    assert registry["evaluation_only_target_count"] == 116
    assert target["training_eligible"] is False
    assert target["availability_unknown_recall_floor"] is True
    assert target["availability_known_false_abstain_floor"] is True
    assert target["value_safety_floor"] is False
    assert target["parser_ood_value_safety_floor"] is False
    assert target["memory_promotion_allowed"] is False
    rules = _load("improvement_rules.json")
    policy = rules["pg143_oracle_availability_abstention"]
    assert policy["training_eligible"] is False
    assert policy["availability_unknown_recall_floor"] is True
    assert policy["availability_known_false_abstain_floor"] is True
    assert policy["value_safety_floor"] is False
    assert policy["memory_promotion_allowed"] is False
