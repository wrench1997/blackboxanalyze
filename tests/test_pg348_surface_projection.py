from __future__ import annotations

import copy

import pytest

from app.pg348_surface_projection import AXES, SCHEMA_VERSION, project_fixture_metadata, project_surface


def _record(**overrides):
    value = {
        "challenge_id": "pg348-a-0001",
        "local_path": "pages_a/page_0001.html",
        "mechanism_id": "mechanism_alpha",
        "surface_template_id": "template_01",
        "implementation_group": "impl_a",
        "transport_method": "GET",
        "parameter_role": "search_term",
        "encoding_chain": ["url_percent", "utf8"],
        "response_shape": "row_shape",
        "redirect_shape": "none",
        "script_surface": "none",
        "synthetic_oracle_kind": "shape_only",
        "source_hash": "a" * 64,
        "raw_source_for_evaluator_only": True,
        "training_context_raw": False,
    }
    value.update(overrides)
    return value


def test_pg348_projects_seven_axes_and_keeps_provenance_off_context() -> None:
    result = project_surface(_record())
    assert result["schema_version"] == SCHEMA_VERSION
    assert set(result["axis_presence"]) == {
        "document_presence",
        "navigation_presence",
        "request_transport_presence",
        "response_transport_presence",
        "javascript_presence",
        "failure_feedback_presence",
        "belief_replay_presence",
    }
    assert len(result["context_tokens"]) > 0
    assert "request_method=get" in result["context_tokens"]
    assert "param_role=search_term" in result["context_tokens"]
    assert any(token.startswith("request_encoding_chain=") and "url_percent" in token for token in result["context_tokens"])
    assert any(token.startswith("response_body_shape=") for token in result["context_tokens"])
    assert any(token.startswith("response_redirect_chain_shape=") for token in result["context_tokens"])
    assert "js_script_kind=none" in result["context_tokens"]
    assert result["context_firewall"]["sidecars_off_context"] is True
    assert result["sidecar"]["challenge_id"] == "pg348-a-0001"
    assert result["sidecar"]["local_path"] == "pages_a/page_0001.html"
    assert not any("pg348-a-0001" in token or "pages_a" in token for token in result["context_tokens"])
    assert result["raw_payload_stored"] is False
    assert result["raw_response_body_stored"] is False


def test_pg348_supports_post_and_nested_abstract_surface_metadata() -> None:
    result = project_surface(
        _record(
            transport_method="POST",
            parameter_role=[{"role": "anti_csrf", "value_type": "opaque", "presence": "present", "order": 1}],
            encoding_chain=["form_urlencoded", "utf8"],
            response_shape="json_rows",
            redirect_shape="single_hop",
            script_surface={
                "script_count": 1,
                "script_kind": "inline",
                "module_presence": "absent",
                "inline_external_class": "inline",
                "source_category": "handler",
                "sink_category": "dom_text",
                "syntax_shape": "statement",
                "dynamic_code_presence": "absent",
                "storage_api_presence": "absent",
                "event_handler_kinds": ["submit"],
            },
        )
    )
    assert "request_method=post" in result["context_tokens"]
    assert "request_placement=form" in result["context_tokens"]
    assert "param_role=anti_csrf" in result["context_tokens"]
    assert any(token.startswith("request_encoding_chain=") and "form_urlencoded" in token for token in result["context_tokens"])
    assert "js_script_kind=inline" in result["context_tokens"]
    assert "js_event=submit" in result["context_tokens"]
    assert result["field_capture_manifest"]["request_transport"]["parameter_role"] == "observed"


def test_pg348_missing_fields_are_not_observed_and_force_safe_ask() -> None:
    result = project_surface({"challenge_id": "pg348-incomplete", "transport_method": "GET"})
    assert all(axis in result["field_capture_manifest"] for axis in AXES)
    assert result["field_capture_manifest"]["response_transport"]["body_shape"] == "not_observed"
    assert result["axis_presence"]["response_transport_presence"] == "not_observed"
    assert result["target"]["safe_to_send"] is False
    assert result["target"]["next_action"] in {"ask", "ask_typed"}
    assert result["target"]["question"] in {"ask_parameter_role", "ask_response", "ask_typed", "ask_failure", "ask_belief"}
    assert result["loss_report"]["training_eligible"] is False
    assert result["promotion"]["training_allowed"] is False


def test_pg348_explicit_unknown_status_remains_unknown_and_drives_ask() -> None:
    result = project_surface(
        _record(
            observed_fields={"request_transport": {"encoding_chain": "unknown"}},
        )
    )
    assert result["field_capture_manifest"]["request_transport"]["encoding_chain"] == "unknown"
    assert result["target"]["safe_to_send"] is False
    assert result["target"]["question"] in {"ask_encoding", "ask_response", "ask_typed"}


@pytest.mark.parametrize(
    "field",
    ["payload", "raw_payload", "response_body", "oracle_answer", "url", "route_literal", "family_label"],
)
def test_pg348_rejects_raw_or_literal_side_channels(field: str) -> None:
    value = _record()
    value[field] = "literal"
    with pytest.raises(ValueError, match="rejects"):
        project_surface(value)


def test_pg348_does_not_accept_raw_source_as_training_context() -> None:
    value = _record(training_context_raw=True)
    with pytest.raises(ValueError, match="training_context_raw"):
        project_surface(value)


def test_pg348_missing_provenance_hash_keeps_projection_incomplete() -> None:
    value = _record()
    value.pop("source_hash")
    result = project_surface(value)
    assert result["status"] == "incomplete"
    assert result["sidecar"]["source_hash_valid"] is False
    assert result["target"]["safe_to_send"] is False


def test_pg348_projection_is_pure_and_alias_is_same_contract() -> None:
    value = _record()
    before = copy.deepcopy(value)
    first = project_surface(value)
    second = project_fixture_metadata(value)
    assert value == before
    assert first == second
