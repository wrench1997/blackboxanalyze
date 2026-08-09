from __future__ import annotations

from scripts.audit_pg368_slot_coverage import (
    REQUIRED_SLOTS,
    _contains_forbidden_keys,
    build_pg368_slot_coverage_audit,
)


def _dataset(report, name):
    return next(item for item in report["datasets"] if item["dataset"] == name)


def test_reference_exposes_exact_syntax_and_shape_slots_without_reclassifying_old_rows():
    report = build_pg368_slot_coverage_audit()
    reference = report["reference"]
    assert reference["reference_status"] == "passed"
    assert reference["target_slot_schema"]["syntax_category_ref"]["reference_exact_count"] > 0
    assert reference["target_slot_schema"]["payload_shape_ref"]["reference_exact_count"] > 0
    assert reference["target_slot_schema"]["syntax_category_ref"]["reference_distinct_value_count"] > 1
    assert reference["target_slot_schema"]["payload_shape_ref"]["reference_distinct_value_count"] > 1
    assert report["safe_abstract_projection"]["reclassification_performed"] is False
    assert report["safe_abstract_projection"]["new_training_rows_generated"] is False


def test_old_rows_keep_missing_exact_slots_and_sidecar_oracle_is_not_model_input():
    report = build_pg368_slot_coverage_audit()
    for name in ("pg333_webgoat", "pg337_dvwa", "pg342_webgoat"):
        dataset = _dataset(report, name)
        assert dataset["target_slot_coverage"]["syntax_category_ref"]["exact_target_count"] == 0
        assert dataset["target_slot_coverage"]["payload_shape_ref"]["exact_target_count"] == 0
        assert dataset["target_slot_coverage"]["oracle_ref"]["exact_target_count"] == 0
        assert dataset["evaluator_sidecar"]["model_context_allowed"] is False
        assert "typed_oracle_sidecar_not_model_slot" in dataset["blocked_reasons"]
        assert dataset["source_training_eligible_count"] in {0, 12}


def test_encoding_and_parameter_aliases_are_counted_but_not_promoted_to_missing_slots():
    report = build_pg368_slot_coverage_audit()
    for name in ("pg333_webgoat", "pg337_dvwa", "pg342_webgoat"):
        dataset = _dataset(report, name)
        assert dataset["target_slot_coverage"]["encoding_ref"]["exact_target_count"] == dataset["unit_count"]
        assert dataset["target_slot_coverage"]["field_role_ref"]["exact_target_count"] == dataset["unit_count"]
        assert dataset["target_slot_coverage"]["encoding_ref"]["distinct_value_count"] == 1
        assert dataset["target_slot_coverage"]["field_role_ref"]["distinct_value_count"] == 1
        assert dataset["target_slot_coverage"]["syntax_category_ref"]["exact_target_count"] == 0


def test_failure_repair_and_get_post_coverage_are_reported_separately():
    report = build_pg368_slot_coverage_audit()
    pg333 = _dataset(report, "pg333_webgoat")
    pg337 = _dataset(report, "pg337_dvwa")
    pg342 = _dataset(report, "pg342_webgoat")
    assert pg333["failure_repair"]["observed_repair_count"] == 0
    assert pg337["failure_repair"]["observed_repair_count"] == 6
    assert pg342["failure_repair"]["observed_repair_count"] == 4
    assert pg333["methods"] == {"get": 9, "post": 9}
    assert pg337["methods"] == {"get": 9}
    assert pg342["methods"] == {"get": 3, "post": 3}


def test_pg368_plan_is_planned_unobserved_and_global_gate_stays_blocked():
    report = build_pg368_slot_coverage_audit()
    plan = _dataset(report, "pg368_plan")
    assert report["status"] == "blocked"
    assert "missing_exact_target_slot:syntax_category_ref" in plan["blocked_reasons"]
    assert "missing_exact_target_slot:payload_shape_ref" in plan["blocked_reasons"]
    assert plan["evaluator_sidecar"]["typed_available_count"] == 0
    assert plan["evaluator_sidecar"]["planned_unobserved_count"] == plan["unit_count"]
    assert plan["evaluator_sidecar"]["model_context_allowed"] is False
    assert plan["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    assert all(value is False for value in report["promotion"].values())


def test_report_has_all_required_slot_names_and_forbidden_wire_keys_are_absent():
    report = build_pg368_slot_coverage_audit()
    assert tuple(report["required_slots"]) == REQUIRED_SLOTS
    assert _contains_forbidden_keys(report) == []
    assert report["safe_abstract_projection"]["unsafe_literal_presence"] is False
    assert report["safe_abstract_projection"]["transport_literal_presence"] is False
