import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg54_replays_independent_get_post_and_keeps_typed_oracle_out_of_model():
    report = _load("pg54_pg42_rule_ir_ood_report_v1.json")
    summary = report["summary"]
    assert report["status"] == "diagnostic_only"
    assert summary["case_count"] == 360
    assert summary["confirmed_positive_count"] == 324
    assert summary["confirmed_negative_count"] == 36
    assert summary["get_post_covered"] == {"GET": 180, "POST": 180}
    assert summary["negative_control_pass_count"] == 360
    assert summary["fresh_reset_count"] == 360
    assert report["model"]["oracle_visible_before_probe"] is False
    assert report["model"]["typed_oracle_in_features"] is False
    assert report["model"]["family_label_in_features"] is False
    assert report["model"]["raw_request_response_in_features"] is False


def test_pg54_feature_transfer_and_unknown_family_gates_are_explicitly_blocked():
    report = _load("pg54_pg42_rule_ir_ood_report_v1.json")
    review = report["feature_review"]
    assert review["review"]["decision"] == "approved_for_downstream_ood_experiment"
    assert review["selected_features_revalidated_on_pg54"] is False
    assert review["feature_transfer_gate"] == "blocked"
    unknown = report["unknown_family_policy"]
    assert unknown["family"] == "template_injection"
    assert unknown["model_class_present"] is False
    assert unknown["must_abstain"] is True
    assert unknown["unknown_misname_count"] == 0
    assert unknown["strict_abstain"] is True
    assert report["model"]["density_gate"]["gate_enabled"] is True
    assert report["model"]["density_gate"]["calibration_source"] == "pg53-pg35-dev-only"
    assert report["splits"]["all"]["negative_effect_false_accept_count"] == 0
    assert report["splits"]["all"]["abstain_rate"] == 1.0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_claim_allowed"] is False


def test_pg54_persists_only_bounded_evidence():
    report = _load("pg54_pg42_rule_ir_ood_report_v1.json")
    trace = _load("pg54_pg42_rule_ir_ood_trace_v1.json")
    assert report["trace"]["raw_probe_strings_stored"] is False
    assert report["trace"]["raw_response_bodies_stored"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert len(trace["rows"]) == 360
    for row in trace["rows"]:
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"])
        assert row["negative_control"]["matched"] is True
        assert row["fresh_reset"]["fresh_target"] is True
        assert row["fresh_reset"]["completed"] is True
    text = json.dumps(report, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "union select" not in text
    assert "onerror" not in text


def test_pg54_registry_keeps_holdout_out_of_training():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    target = next(item for item in registry["targets"] if item["target_id"] == "pg54_pg42_rule_ir_ood")
    assert target["training_eligible"] is False
    assert target["feature_transfer_gate"] == "blocked"
    assert registry["evaluation_only_target_count"] == 116
    assert registry["training_eligible_target_count"] == 40
