from __future__ import annotations

import pytest

from app.pg331_web_tokenizer import tokenize_web_observation


def _complete_observation() -> dict[str, object]:
    return {
        "document_structure": {
            "doctype": "html",
            "html_lang": "zh",
            "title_shape": "alpha",
            "head_count": 1,
            "meta_count": 2,
            "style_count": 1,
            "script_count": 2,
            "section_count": 3,
            "elements": [{"tag": "form", "depth": 2, "sibling_count": 1, "role": "form", "id_shape": "alpha", "class_shape": "word_mixed", "text_shape": "alpha", "text_length": 12, "attribute_presence": ["method", "name"]}],
        },
        "navigation": {"links": [{"method": "GET", "target_shape": "path_like", "same_origin": "yes", "query_present": "absent", "fragment_present": "absent"}], "path_segment_count": 2, "query_key_count": 1, "form_action_shape": "path_like"},
        "request_transport": {"method": "GET", "placement": "query", "content_type_class": "none", "encoding_chain": "url_percent", "charset_class": "utf8", "body_shape": "empty", "query_count": 1, "form_count": 0, "json_field_count": 0, "multipart_part_count": 0, "header_presence_class": "basic", "cookie_presence_class": "absent", "csrf_presence_class": "unknown", "content_length": 0, "parameters": [{"role": "query", "name_shape": "alpha", "value_type": "text", "presence": "present", "order": 1}]},
        "response_transport": {"status_class": "2xx", "content_type_class": "html", "connection_outcome": "complete", "body_length": 256, "redirect_hop_count": 0, "body_shape": "html", "charset_class": "utf8", "header_presence_class": "basic", "cache_shape": "none", "redirect_location_class": "none", "redirect_chain_shape": "empty"},
        "javascript_surface": {"script_count": 2, "event_handler_count": 1, "fetch_count": 1, "xhr_count": 0, "ast_node_count": 12, "script_kind": "module", "module_presence": "present", "inline_external_class": "external", "source_category": "dom_input", "sink_category": "dom_update", "syntax_shape": "call_member", "dynamic_code_presence": "absent", "storage_api_presence": "absent", "fetch_method": "GET", "xhr_method": "UNKNOWN", "fetch_target_shape": "path_like", "xhr_target_shape": "empty", "event_handler_kinds": ["submit"]},
        "failure_feedback": {"failure_class": "none", "failure_stage": "none", "error_shape": "empty", "parse_error_class": "none", "encoding_error_class": "none", "redirect_error_class": "none", "blocked_reason_class": "none", "environment_failure_class": "none", "previous_action": "none", "next_action": "ask", "repair_delta_axis": "none", "repair_outcome": "not_applicable", "timeout_ms": 0},
        "belief_and_replay": {"observation_presence": "present", "observation_delta_axis": "response_shape", "belief_prior_bucket": "mid", "belief_posterior_bucket": "mid", "belief_delta_axis": "response_shape", "history_action": "candidate_request", "typed_available": "present", "evidence_present": "present", "negative_control": "present", "fresh_reset": "present", "replay_ready": "present", "reference_present": "present", "candidate_present": "present", "step_budget": "present", "evidence_hash_present": "present", "history_length": 1, "probe_count": 1},
    }


def test_complete_observation_emits_every_ontology_presence_axis() -> None:
    result = tokenize_web_observation(_complete_observation())
    tokens = set(result["context_tokens"])
    assert result["loss_report"]["training_eligible"] is True
    assert result["loss_report"]["chunk_count"] == 1
    assert result["chunks"]
    assert "chunk_boundary=begin" in tokens and "chunk_boundary=end" in tokens
    assert any(token.startswith("chunk_digest=b") for token in tokens)
    assert {f"{key}=observed" for key in ("document_presence", "navigation_presence", "request_transport_presence", "response_transport_presence", "javascript_presence", "failure_feedback_presence", "belief_replay_presence")} <= tokens
    assert "document_structure_field_doctype=html" in tokens
    assert "request_transport_field_parameter_role=one" in tokens
    assert "response_transport_field_body_length_bucket=medium" in tokens
    assert "javascript_surface_field_event_handler_kind=one" in tokens
    assert result["loss_report"]["raw_fields_omitted"] == []


def test_missing_axes_are_explicit_not_observed() -> None:
    result = tokenize_web_observation({})
    tokens = set(result["context_tokens"])
    assert result["loss_report"]["training_eligible"] is False
    assert result["loss_report"]["chunk_count"] >= 1
    assert "document_presence=not_observed" in tokens
    assert "response_transport_presence=not_observed" in tokens
    assert len(result["loss_report"]["missing_axes"]) == 7


def test_unknown_shape_states_are_not_disguised_as_alpha() -> None:
    observation = _complete_observation()
    document = dict(observation["document_structure"])
    document["doctype"] = "unknown"
    document["html_lang"] = "unknown"
    observation["document_structure"] = document
    tokens = set(tokenize_web_observation(observation)["context_tokens"])
    assert "doc_doctype=unknown" in tokens
    assert "doc_html_lang=unknown" in tokens
    assert "doc_doctype=alpha" not in tokens
    assert "doc_html_lang=alpha" not in tokens


def test_raw_fields_are_not_emitted_and_literal_executable_input_is_rejected() -> None:
    result = tokenize_web_observation({"request_transport": {"method": "GET", "debug": {"raw_payload": "<ignored>"}}})
    assert "request_transport.debug.raw_payload" in result["loss_report"]["raw_fields_omitted"]
    assert all("<ignored>" not in token for token in result["context_tokens"])
    with pytest.raises(ValueError):
        tokenize_web_observation({"request_transport": {"method": "GET", "body_shape": "<script>alert"}})
