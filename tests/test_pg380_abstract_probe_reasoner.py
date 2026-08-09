from __future__ import annotations

from app.pg380_abstract_probe_reasoner import derive_abstract_probe_plan


def _observation(**feedback: object) -> dict[str, object]:
    return {
        "method": "POST",
        "surface_context": "html_form",
        "parameter_role": "form_field",
        "response_shape": "html_text",
        "negative_control": True,
        "filter_feedback": {
            "state": "filtered",
            "filter_class": "encoding_normalized",
            "encoding_observed": "identity",
            **feedback,
        },
    }


def test_filtered_encoding_feedback_selects_one_variable_abstract_repair() -> None:
    result = derive_abstract_probe_plan(_observation())
    assert result["status"] == "abstract_variant_selected"
    assert result["wire_binding_requested"] is True
    assert result["safe_to_send"] is False
    assert result["rule_ir"]["repair_action"] == "encoding"
    assert result["rule_ir"]["encoding_ref"] == "double_layer_order_sensitive"
    assert result["rule_ir"]["payload_shape_ref"] == "html_form_marker"
    assert result["raw_payload"] is None


def test_delimiter_failure_changes_syntax_not_everything_at_once() -> None:
    result = derive_abstract_probe_plan(_observation(filter_class="delimiter_rejected"))
    assert result["rule_ir"]["repair_action"] == "syntax"
    assert result["rule_ir"]["syntax_category_ref"] == "structured_value"
    assert result["rule_ir"]["encoding_ref"] == "identity"


def test_missing_filter_class_asks_before_binding() -> None:
    result = derive_abstract_probe_plan(_observation(filter_class="unknown"))
    assert result["status"] == "ask_filter_observation"
    assert result["next_action"] == "ask"
    assert result["wire_binding_requested"] is False
    assert result["safe_to_send"] is False


def test_typed_effect_requests_fresh_replay() -> None:
    result = derive_abstract_probe_plan(_observation(state="typed_effect", filter_class="none"))
    assert result["status"] == "ready_for_fresh_replay"
    assert result["next_action"] == "replay"
    assert result["rule_ir"]["probe_variant_ref"] == "fresh_replay"
    assert result["rule_ir"]["oracle_ref"] == "response_shape"


def test_unknown_surface_or_missing_observation_asks() -> None:
    result = derive_abstract_probe_plan({"method": "GET", "surface_context": "unknown"})
    assert result["status"] == "ask_missing_observation"
    assert result["safe_to_send"] is False
    assert result["rule_ir"] is None


def test_raw_or_callback_input_is_rejected_without_wire() -> None:
    result = derive_abstract_probe_plan({"method": "GET", "url": "http://example.invalid", "surface_context": "html_text"})
    assert result["status"] == "rejected_raw_or_evaluator_observation"
    assert result["wire_binding_requested"] is False
    assert result["safe_to_send"] is False
