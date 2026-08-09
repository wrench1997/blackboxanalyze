from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.pg331_source_row import collect_pg331_source_row, validate_pg331_source_row


def _field_capture_manifest() -> dict[str, dict[str, str]]:
    ontology = json.loads((Path(__file__).parents[1] / "research" / "pg331_web_token_ontology_v1.json").read_text(encoding="utf-8"))
    observation = _observation()
    manifest: dict[str, dict[str, str]] = {}
    for axis, spec in dict(ontology["axes"]).items():
        section = observation[str(axis)]
        assert isinstance(section, dict)
        manifest[str(axis)] = {}
        for field in list(spec.get("fields") or []):
            raw = section.get(str(field))
            manifest[str(axis)][str(field)] = "unknown" if raw is None or str(raw).casefold() == "unknown" else "absent" if str(raw).casefold() == "absent" else "observed"
    return manifest


def _observation() -> dict[str, object]:
    observation: dict[str, object] = {
        "document_structure": {"doctype": "html", "html_lang": "zh", "title_shape": "alpha", "elements": []},
        "navigation": {"links": [], "path_segment_count": 2, "query_key_count": 1, "form_action_shape": "path_like"},
        "request_transport": {"method": "GET", "placement": "query", "content_type_class": "none", "encoding_chain": "url_percent", "charset_class": "utf8", "body_shape": "empty", "query_count": 1, "form_count": 0, "json_field_count": 0, "multipart_part_count": 0, "header_presence_class": "basic", "cookie_presence_class": "absent", "csrf_presence_class": "absent", "content_length": 0, "parameters": []},
        "response_transport": {"status_class": "2xx", "content_type_class": "html", "connection_outcome": "complete", "body_length": 128, "redirect_hop_count": 0, "body_shape": "html", "charset_class": "utf8", "header_presence_class": "basic", "cache_shape": "none", "redirect_location_class": "none", "redirect_chain_shape": "empty"},
        "javascript_surface": {"script_count": 0, "event_handler_count": 0, "fetch_count": 0, "xhr_count": 0, "ast_node_count": 0, "script_kind": "none", "module_presence": "absent", "inline_external_class": "none", "source_category": "none", "sink_category": "none", "syntax_shape": "empty", "dynamic_code_presence": "absent", "storage_api_presence": "absent", "fetch_method": "ABSENT", "xhr_method": "ABSENT", "fetch_target_shape": "empty", "xhr_target_shape": "empty", "event_handler_kinds": []},
        "failure_feedback": {"failure_class": "none", "failure_stage": "none", "error_shape": "empty", "parse_error_class": "none", "encoding_error_class": "none", "redirect_error_class": "none", "blocked_reason_class": "none", "environment_failure_class": "none", "previous_action": "none", "next_action": "ask", "repair_delta_axis": "none", "repair_outcome": "not_applicable", "timeout_ms": 0},
        "belief_and_replay": {"observation_presence": "present", "observation_delta_axis": "response_shape", "belief_prior_bucket": "mid", "belief_posterior_bucket": "mid", "belief_delta_axis": "response_shape", "history_action": "candidate_request", "typed_available": "present", "evidence_present": "present", "negative_control": "present", "fresh_reset": "present", "replay_ready": "present", "reference_present": "present", "candidate_present": "present", "step_budget": "present", "evidence_hash_present": "present", "history_length": 1, "probe_count": 1},
    }
    ontology = json.loads((Path(__file__).parents[1] / "research" / "pg331_web_token_ontology_v1.json").read_text(encoding="utf-8"))
    for axis, spec in dict(ontology["axes"]).items():
        section = observation[str(axis)]
        assert isinstance(section, dict)
        for field in list(spec.get("fields") or []):
            name = str(field)
            if name not in section:
                section[name] = 0 if any(marker in name for marker in ("count", "length", "bucket", "order", "hop_count")) else "absent"
    return observation


def _source_meta() -> dict[str, str]:
    return {
        "source_id": "fixture-page-01",
        "implementation": "fixture-impl-a",
        "family_id": "family-heldout-a",
        "surface_id": "surface-01",
        "collector_id": "collector-test",
        "authorization_id": "auth-local-test",
        "image_digest": "a" * 64,
        "source_digest": "b" * 64,
    }


def _reset() -> dict[str, object]:
    return {"fresh_reset": True, "reset_id": "reset-01", "target_instance_digest": "c" * 64, "network_mode": "none", "external_network": False, "loopback_only": True, "state_clean": True}


def _evaluator() -> dict[str, object]:
    return {"typed_available": True, "negative_control": True, "reference_present": True, "candidate_present": True, "fresh_reset": True, "evidence_hash": "d" * 64, "confirmed_positive": False, "effect_class": "no_effect", "evaluator_version": "fixture-oracle-v1"}


def _target() -> dict[str, object]:
    return {"question": "none", "next_action": "assemble_rule_ir", "repair_action": "none", "transport_ref": "request_method", "field_role_ref": "parameter_role", "encoding_ref": "encoding_chain", "probe_variant_ref": "negative_control", "safe_to_send": True}


def test_complete_row_is_abstract_and_training_eligible_only_after_review() -> None:
    row = collect_pg331_source_row(record_id="pg331a:complete", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), split="train", operator_reviewed=True)
    assert row["training_eligible"] is True
    assert row["context_firewall"]["sidecars_off_context"] is True
    assert all(value == "observed" for value in row["axis_presence"].values())
    assert all("payload" not in token.casefold() for token in row["context_tokens"])
    assert validate_pg331_source_row(row, require_training_eligible=True)["valid"] is True


def test_missing_axis_is_explicit_and_blocks_training() -> None:
    observation = _observation()
    observation.pop("javascript_surface")
    row = collect_pg331_source_row(record_id="pg331a:missing-js", observation=observation, source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), operator_reviewed=True)
    assert row["training_eligible"] is False
    assert "javascript_presence" in row["axis_presence"]
    assert row["axis_presence"]["javascript_presence"] == "not_observed"
    assert any(item.startswith("axis_not_observed:javascript_presence") for item in row["failures"])
    assert "question=ask_typed" in row["target_tokens"]
    assert "next_action=ask_typed" in row["target_tokens"]
    assert "safe_to_send=0" in row["target_tokens"]


def test_raw_side_channel_field_never_becomes_training_data() -> None:
    observation = deepcopy(_observation())
    observation["request_transport"]["family_label"] = "sql"
    row = collect_pg331_source_row(record_id="pg331a:raw-sidecar", observation=observation, source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), operator_reviewed=True)
    assert row["training_eligible"] is False
    assert "raw_fields_in_observation" in row["failures"]
    assert all("sql" not in token.casefold() for token in row["context_tokens"])


def test_failure_without_action_change_cannot_enter_training() -> None:
    observation = deepcopy(_observation())
    observation["failure_feedback"] = {"failure_class": "parse_error", "failure_stage": "request", "previous_action": "candidate_probe", "next_action": "candidate_probe"}
    row = collect_pg331_source_row(record_id="pg331a:stuck", observation=observation, source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), operator_reviewed=True)
    assert row["training_eligible"] is False
    assert "failure_action_not_changed" in row["failures"]


def test_field_not_observed_forces_ask_even_when_all_axes_exist() -> None:
    manifest = _field_capture_manifest()
    manifest["javascript_surface"]["script_count"] = "not_observed"
    row = collect_pg331_source_row(record_id="pg331a:missing-field", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=manifest, target_projection=_target(), operator_reviewed=True)
    assert row["training_eligible"] is False
    assert any(item.startswith("field_not_observed:javascript_surface.script_count") for item in row["failures"])
    assert "question=ask_typed" in row["target_tokens"]


def test_missing_evaluator_forces_safe_ask_target() -> None:
    evaluator = _evaluator()
    evaluator["typed_available"] = False
    row = collect_pg331_source_row(record_id="pg331a:missing-evaluator", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=evaluator, field_capture_manifest=_field_capture_manifest(), target_projection=_target(), operator_reviewed=True)
    assert row["training_eligible"] is False
    assert "evaluator_missing:typed_available" in row["failures"]
    assert row["target_projection"]["question"] == "ask_typed"
    assert row["target_projection"]["next_action"] == "ask_typed"
    assert row["target_projection"]["safe_to_send"] is False


def test_abstract_oracle_and_negative_control_slots_are_allowed_without_raw_value() -> None:
    target = _target()
    target.update({"payload_shape_ref": "sql_string_marker", "oracle_ref": "typed_effect", "negative_control_presence_ref": "matched_triplet"})
    row = collect_pg331_source_row(record_id="pg331a:abstract-slots", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=target, split="train")
    assert row["target_projection"]["oracle_ref"] == "typed_effect"
    assert "oracle_ref=typed_effect" in row["target_tokens"]
    assert "negative_control_presence_ref=matched_triplet" in row["target_tokens"]
    assert all("raw" not in token.casefold() for token in row["target_tokens"])


def test_missing_evaluator_downgrades_optional_oracle_slots_to_unknown() -> None:
    evaluator = _evaluator()
    evaluator["typed_available"] = False
    target = _target()
    target.update({"payload_shape_ref": "sql_string_marker", "oracle_ref": "typed_effect", "negative_control_presence_ref": "matched_triplet"})
    row = collect_pg331_source_row(record_id="pg331a:abstract-slots-ask", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=evaluator, field_capture_manifest=_field_capture_manifest(), target_projection=target, operator_reviewed=True)
    assert row["target_projection"]["oracle_ref"] == "unknown"
    assert row["target_projection"]["negative_control_presence_ref"] == "unknown"
    assert row["target_projection"]["safe_to_send"] is False


def test_stuck_failure_forces_repair_and_safe_stop() -> None:
    observation = deepcopy(_observation())
    observation["failure_feedback"] = {"failure_class": "parse_error", "failure_stage": "request", "previous_action": "candidate_probe", "next_action": "candidate_probe"}
    row = collect_pg331_source_row(record_id="pg331a:stuck-target", observation=observation, source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), operator_reviewed=True)
    assert row["target_projection"]["question"] == "ask_failure"
    assert row["target_projection"]["next_action"] == "repair"
    assert row["target_projection"]["repair_action"] == "observe"
    assert row["target_projection"]["safe_to_send"] is False


def test_serialized_row_rejects_unsafe_target_for_missing_evaluator() -> None:
    evaluator = _evaluator()
    evaluator["typed_available"] = False
    row = collect_pg331_source_row(record_id="pg331a:tampered-target", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=evaluator, field_capture_manifest=_field_capture_manifest(), target_projection=_target(), operator_reviewed=True)
    row["target_projection"]["safe_to_send"] = True
    row["target_tokens"] = ["[TARGET_BOS]", "safe_to_send=1", "[TARGET_EOS]"]
    row["record_sha256"] = "0" * 64
    result = validate_pg331_source_row(row)
    assert "unsafe_target_on_incomplete_row" in result["failures"] or "target_token_projection_mismatch" in result["failures"]


def test_row_hash_is_required_for_serialized_replay() -> None:
    row = collect_pg331_source_row(record_id="pg331a:hash", observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target())
    assert validate_pg331_source_row(row)["valid"] is True
    row["context_tokens"] = list(row["context_tokens"]) + ["tampered=one"]
    assert "record_hash_mismatch" in validate_pg331_source_row(row)["failures"]


def test_database_health_gate_is_allowlisted_when_present() -> None:
    reset = _reset()
    reset["database_health_gate"] = "mysqli_root_pikachu_ok"
    row = collect_pg331_source_row(record_id="pg331a:db-health", observation=_observation(), source_meta=_source_meta(), reset=reset, evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target())
    assert validate_pg331_source_row(row)["valid"] is True
    reset["database_health_gate"] = "unverified"
    with pytest.raises(ValueError, match="database_health_gate"):
        collect_pg331_source_row(record_id="pg331a:db-health-bad", observation=_observation(), source_meta=_source_meta(), reset=reset, evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target())


def test_independent_juice_shop_health_gate_is_allowlisted() -> None:
    reset = _reset()
    reset["database_health_gate"] = "juice_shop_http_health_ok"
    row = collect_pg331_source_row(record_id="pg331a:juice-db-health", observation=_observation(), source_meta=_source_meta(), reset=reset, evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target())
    assert row["reset"]["database_health_gate"] == "juice_shop_http_health_ok"
