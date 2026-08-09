"""Abstract probe reasoning for the last evaluator hop.

PG-380 is intentionally a *reasoning* boundary, not a payload generator.  It
turns sanitized surface/transport/filter observations into a bounded Rule-IR
proposal and an explanation of the next one-variable probe.  Concrete bytes
remain in a separately reviewed evaluator template catalog (PG-350).  The
model-facing result therefore contains no URL, payload, response body,
callback, script, or WAF-bypass literal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.pg361_payload_shape_slots import (
    ALLOWED_ENCODINGS,
    ALLOWED_FIELD_ROLES,
    ALLOWED_ORACLES,
    ALLOWED_SHAPES,
    ALLOWED_SYNTAX_CATEGORIES,
    ALLOWED_TRANSPORTS,
    ALLOWED_VARIANTS,
)


SCHEMA_VERSION = "pg380-abstract-probe-reasoning-v1"
MODEL_SLOTS = (
    "question",
    "ask_reason",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "probe_variant_ref",
    "safe_to_send",
    "payload_shape_ref",
    "oracle_ref",
    "negative_control_presence_ref",
)

_FORBIDDEN_KEYS = {
    "payload",
    "raw_payload",
    "raw_value",
    "literal",
    "wire",
    "response",
    "response_body",
    "url",
    "route_literal",
    "evaluator_answer",
    "oracle_answer",
    "callback",
    "webhook",
}
_FORBIDDEN_PARTS = ("raw_", "response_body", "evaluator_", "wire", "payload")
_FORBIDDEN_VALUES = (
    "http://",
    "https://",
    "javascript:",
    "document.cookie",
    "<script",
    "<img",
    "onerror=",
    "powershell",
    "cmd.exe",
    "curl ",
    "wget ",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_raw(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_KEYS or any(part in key_text for part in _FORBIDDEN_PARTS):
                return f"forbidden_key:{path}.{key}"
            found = _reject_raw(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found = _reject_raw(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        folded = value.casefold()
        for fragment in _FORBIDDEN_VALUES:
            if fragment in folded:
                return f"forbidden_value:{fragment}"
    return None


def _normal(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else default).casefold().replace("-", "_")
    return text if text else default


def _method_transport(method: str) -> str:
    if method == "GET":
        return "get_query"
    if method == "POST":
        return "post_form"
    return "unknown"


def _surface_defaults(surface: str, method: str) -> tuple[str, str, str]:
    """Return shape, syntax and oracle classes from abstract surface only."""

    table = {
        "html_text": ("html_text_marker", "marker", "reflection"),
        "html_attribute": ("html_attribute_marker", "structured_value", "dom_shape"),
        "html_dom": ("html_dom_marker", "expression_node", "dom_shape"),
        "json_string": ("json_string_marker", "structured_value", "parser_shape"),
        "sql_string": ("sql_string_marker", "delimiter_boundary", "response_shape"),
        "sql_numeric": ("sql_numeric_marker", "boolean_branch", "response_shape"),
        "path_segment": ("path_segment_marker", "parser_node", "response_shape"),
        "query": ("query_marker", "structured_value", "response_shape"),
        "form": ("html_form_marker", "structured_value", "response_shape"),
        "html_form": ("html_form_marker", "structured_value", "response_shape"),
        "redirect": ("state_transition_marker", "redirect_control", "typed_state_delta"),
    }
    shape, syntax, oracle = table.get(surface, ("unknown", "unknown", "unknown"))
    if method not in {"GET", "POST"}:
        return "unknown", "unknown", "unknown"
    return shape, syntax, oracle


def _missing_observations(observation: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("method", "surface_context", "parameter_role", "filter_feedback", "response_shape"):
        value = observation.get(key)
        if value in (None, "", "unknown", "not_observed"):
            missing.append(key)
    feedback = observation.get("filter_feedback")
    if not isinstance(feedback, Mapping):
        missing.append("filter_feedback")
    return missing


def _repair_from_feedback(feedback: Mapping[str, Any], *, surface: str, method: str) -> tuple[str, str, str, str]:
    state = _normal(feedback.get("state"))
    filter_class = _normal(feedback.get("filter_class"))
    shape, syntax, oracle = _surface_defaults(surface, method)
    encoding = _normal(feedback.get("encoding_observed"), "identity")
    repair = "none"
    action = "select_probe_variant"
    variant = "source_attested_candidate"
    reason = "baseline_observation_available"

    if state in {"filtered", "blocked"}:
        variant = "one_variable_repair"
        if filter_class in {"encoding_normalized", "encoding_filter", "canonicalization"}:
            repair, action, reason = "encoding", "repair", "change_encoding_layer_only"
            encoding = "double_layer_order_sensitive" if encoding != "double_layer_order_sensitive" else "html_entity"
        elif filter_class in {"delimiter_rejected", "syntax_filter", "parser_boundary"}:
            repair, action, reason = "syntax", "repair", "change_syntax_class_only"
            syntax = "structured_value" if syntax in {"delimiter_boundary", "structured_value"} else "parser_node"
        elif filter_class in {"shape_filter", "context_filter", "length_limit"}:
            repair, action, reason = "shape", "repair", "change_shape_or_length_bucket_only"
            shape = "query_marker" if shape not in {"query_marker", "path_segment_marker"} else "html_text_marker"
        else:
            action, reason = "ask", "filter_class_not_observed"
    elif state == "parser_error":
        variant, repair, action, reason = "one_variable_repair", "syntax", "repair", "parser_error_requires_syntax_change"
        syntax = "parser_node"
    elif state in {"no_effect", "reflected"}:
        variant, action, reason = "reference_shape", "repair", "typed_effect_not_confirmed"
    elif state == "typed_effect":
        variant, action, reason = "fresh_replay", "replay", "typed_effect_requires_fresh_replay"
    elif state in {"unknown", "not_observed"}:
        action, reason = "ask", "missing_feedback"

    if encoding not in ALLOWED_ENCODINGS:
        encoding = "unknown"
    if syntax not in ALLOWED_SYNTAX_CATEGORIES:
        syntax = "unknown"
    if shape not in ALLOWED_SHAPES:
        shape = "unknown"
    if oracle not in ALLOWED_ORACLES:
        oracle = "unknown"
    if variant not in ALLOWED_VARIANTS:
        variant = "unknown"
    return encoding, syntax, shape, oracle, variant, repair, action, reason


def derive_abstract_probe_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a bounded reasoning trace and Rule-IR proposal.

    This function is intentionally deterministic so that a demo can show why
    a repair changed.  It never emits a concrete payload or sets
    ``safe_to_send`` to true; only the evaluator can bind a reviewed template
    after its own route/reset/evidence gate.
    """

    if not isinstance(observation, Mapping):
        raise ValueError("PG-380 observation must be a mapping")
    raw_error = _reject_raw(observation)
    if raw_error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected_raw_or_evaluator_observation",
            "question": "ask_sanitized_observation",
            "ask_reason": raw_error,
            "wire_binding_requested": False,
            "safe_to_send": False,
            "abstract_trace": [],
            "input_sha256": _digest({"rejected": raw_error}),
        }

    method = _normal(observation.get("method")).upper()
    surface = _normal(observation.get("surface_context"))
    role = _normal(observation.get("parameter_role"))
    feedback = observation.get("filter_feedback")
    feedback_map = dict(feedback) if isinstance(feedback, Mapping) else {}
    missing = _missing_observations(observation)
    if method not in {"GET", "POST"}:
        missing.append("method(GET_or_POST)")
    if role not in ALLOWED_FIELD_ROLES:
        missing.append("parameter_role")

    if missing:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ask_missing_observation",
            "question": "ask_missing_observation",
            "ask_reason": "missing:" + ",".join(dict.fromkeys(missing)),
            "next_action": "ask",
            "repair_action": "observe",
            "wire_binding_requested": False,
            "safe_to_send": False,
            "abstract_trace": [{"step": "ask", "missing": list(dict.fromkeys(missing))}],
            "rule_ir": None,
            "input_sha256": _digest(observation),
            "raw_payload": None,
        }

    encoding, syntax, shape, oracle, variant, repair, action, reason = _repair_from_feedback(
        feedback_map, surface=surface, method=method
    )
    if action == "ask":
        status = "ask_filter_observation"
        question = "ask_filter_observation"
    elif variant == "fresh_replay":
        status = "ready_for_fresh_replay"
        question = "none"
    else:
        status = "abstract_variant_selected"
        question = "none"

    rule_ir = {
        "question": question,
        "ask_reason": "none" if question == "none" else reason,
        "next_action": action,
        "repair_action": repair,
        "transport_ref": _method_transport(method),
        "field_role_ref": role,
        "encoding_ref": encoding,
        "syntax_category_ref": syntax,
        "probe_variant_ref": variant,
        "safe_to_send": False,
        "payload_shape_ref": shape,
        "oracle_ref": oracle,
        "negative_control_presence_ref": "matched_triplet" if observation.get("negative_control") is True else "unknown",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "question": question,
        "ask_reason": rule_ir["ask_reason"],
        "next_action": action,
        "repair_action": repair,
        "wire_binding_requested": bool(action != "ask"),
        "safe_to_send": False,
        "rule_ir": rule_ir,
        "abstract_trace": [
            {"step": "classify_surface", "surface_context": surface, "method": method, "parameter_role": role},
            {"step": "read_feedback", "state": _normal(feedback_map.get("state")), "filter_class": _normal(feedback_map.get("filter_class"))},
            {"step": "choose_one_variable_change", "reason": reason, "repair_action": repair, "encoding_ref": encoding, "syntax_category_ref": syntax, "payload_shape_ref": shape},
            {"step": "gate_evaluator", "typed_oracle_required": True, "fresh_reset_required": True, "negative_replay_required": True},
        ],
        "evaluator_binding": {
            "template_ref_required": True,
            "source_attested_route_required": True,
            "marker_class": "bounded_loopback_canary",
            "raw_payload_in_model": False,
            "raw_payload_persisted": False,
        },
        "input_sha256": _digest(observation),
        "raw_payload": None,
    }


__all__ = ["MODEL_SLOTS", "SCHEMA_VERSION", "derive_abstract_probe_plan"]
