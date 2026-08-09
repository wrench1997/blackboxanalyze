from __future__ import annotations

import json
from pathlib import Path

from app.pg371_model_slot_binder_contract import (
    parse_target_slots,
    select_and_bind_model_slots,
)
from scripts.plan_pg371_model_slot_binder_contract import (
    build_pg371_model_slot_binder_plan,
    validate_pg371_plan,
)


def _candidate_rule(*, method: str = "GET", safe: bool = True) -> dict[str, object]:
    return {
        "question": "none",
        "ask_reason": "none",
        "next_action": "select_probe_variant",
        "repair_action": "none",
        "transport_ref": "get_query" if method == "GET" else "post_form",
        "field_role_ref": "query_term" if method == "GET" else "form_field",
        "encoding_ref": "identity" if method == "GET" else "form_urlencoded",
        "syntax_category_ref": "marker",
        "payload_shape_ref": "query_marker" if method == "GET" else "html_form_marker",
        "oracle_ref": "response_shape",
        "probe_variant_ref": "source_attested_candidate",
        "safe_to_send": safe,
        "negative_control_presence_ref": "matched_triplet",
    }


def test_model_selected_is_distinct_from_typed_effect_and_wire_creation():
    result = select_and_bind_model_slots(_candidate_rule(), expected_method="GET", role="candidate")
    assert result["model_selected"] is True
    assert result["status"] == "blocked_missing_typed_evidence"
    assert result["safe_to_send"] is False
    assert result["typed_effect_confirmed"] is False
    assert result["wire_created"] is False


def test_model_selected_safe_false_is_explicit_abstain_not_allowlist_error():
    result = select_and_bind_model_slots(_candidate_rule(safe=False), expected_method="GET", role="negative")
    assert result["status"] == "abstain_safe_to_send_false"
    assert result["model_selected"] is True
    assert result["safe_to_send"] is False
    assert result["wire_created"] is False


def test_complete_evidence_still_stops_at_contract_boundary():
    evidence = {
        "typed_available": True,
        "fresh_reset": True,
        "candidate_reference_negative_replay": True,
        "network_none": True,
        "loopback_only": True,
        "evidence_sha256": "a" * 64,
        "context_firewall_closed": True,
    }
    result = select_and_bind_model_slots(_candidate_rule(method="POST"), expected_method="POST", role="reference", evidence=evidence)
    assert result["model_selected"] is True
    assert result["status"] == "template_binding_required"
    assert result["typed_effect_confirmed"] is False
    assert result["wire_created"] is False
    assert result["safe_to_send"] is False


def test_invalid_or_raw_model_slots_fail_closed_without_wire():
    invalid = {**_candidate_rule(), "payload": "literal"}
    result = select_and_bind_model_slots(invalid, expected_method="GET", role="negative")
    assert result["status"] == "rejected_raw_or_evaluator_slot"
    assert result["model_selected"] is True
    assert result["wire_created"] is False
    assert parse_target_slots(["[TARGET_BOS]", "payload=literal", "[TARGET_EOS]"]) is None


def test_pg367_full_target_sequence_maps_to_all_thirteen_abstract_slots():
    dataset = json.loads((Path("research") / "pg367_waf_staircase_dataset_v2.json").read_text(encoding="utf-8-sig"))
    parsed = parse_target_slots(dataset["records"][0]["target_tokens"])
    assert parsed is not None
    assert parsed["syntax_category_ref"] == "marker"
    assert parsed["payload_shape_ref"] in {"query_marker", "html_form_marker"}
    assert parsed["oracle_ref"] in {"typed_effect", "unknown", "negative_no_effect"}
    assert len(parsed) == 13


def test_pg371_plan_preserves_get_post_and_four_roles_with_ask_state():
    report = build_pg371_model_slot_binder_plan()
    validation = validate_pg371_plan(report)
    assert validation["status"] == "passed"
    assert report["counts"] == {
        "seeds": 3,
        "routes": 2,
        "episodes": 6,
        "roles": 24,
        "get_rows": 12,
        "post_rows": 12,
        "candidate_rows": 6,
        "reference_rows": 6,
        "negative_rows": 6,
        "replay_rows": 6,
        "model_selected": 0,
        "typed_effect_confirmed": 0,
        "wire_created": 0,
        "target_contacted": 0,
    }
    assert all(row["model_selected"] is False for row in report["rows"])
    assert all(row["typed_evidence_sha256_required"] is True for row in report["rows"])
    assert all(value is False for key, value in report["promotion"].items() if key.endswith("_allowed"))


def test_pg371_plan_has_no_raw_wire_keys_or_literals():
    report = build_pg371_model_slot_binder_plan()
    text = str(report).casefold()
    assert "http://" not in text
    assert "https://" not in text
    assert "response_body" not in text
    assert "route_literal" not in text
