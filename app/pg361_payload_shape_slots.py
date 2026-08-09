"""PG-361 abstract payload-shape slot contract.

This module is the vocabulary boundary between a causal decoder and the
runtime binder.  It deliberately describes *where/how* a probe fits rather
than the bytes that will be sent.  A reviewed evaluator may later bind these
slots to a one-shot local canary, but a slot sequence can never contain a
literal payload, URL, route, response body, or evaluator answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg361-payload-shape-slots-v1"

SLOT_ORDER = (
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "payload_shape_ref",
    "probe_variant_ref",
    "oracle_ref",
    "negative_control_presence_ref",
    "safe_to_send",
)

ALLOWED_TRANSPORTS = frozenset(
    {"none", "get_query", "get_path", "get_fragment", "post_form", "post_json", "header", "unknown"}
)
ALLOWED_FIELD_ROLES = frozenset(
    {
        "none",
        "query_term",
        "query_text",
        "form_field",
        "display_text",
        "attribute_value",
        "path_segment",
        "fragment_identifier",
        "json_value",
        "header_value",
        "dom_text",
        "filter_choice",
        "sort_direction",
        "record_cursor",
        "profile_key",
        "view_mode",
        "tab_name",
        "step_index",
        "metric_group",
        "note_text",
        "notice_state",
        "status_label",
        "list_item",
        "unknown",
    }
)
ALLOWED_ENCODINGS = frozenset(
    {
        "none",
        "identity",
        "url_percent",
        "form_urlencoded",
        "html_entity",
        "javascript_unicode",
        "json_escape",
        "xml_entity",
        "double_layer_order_sensitive",
        "unknown",
    }
)

# These are grammar *classes*, not payloads.  The deliberately small set is
# useful for cross-surface holdout without teaching the model exploit syntax.
ALLOWED_SYNTAX_CATEGORIES = frozenset(
    {
        "none",
        "unknown",
        "marker",
        "delimiter_boundary",
        "structured_value",
        "expression_node",
        "boolean_branch",
        "parser_node",
        "state_transition",
        "redirect_control",
    }
)
ALLOWED_SHAPES = frozenset(
    {
        "none",
        "unknown",
        "html_text_marker",
        "html_attribute_marker",
        "html_dom_marker",
        "html_fragment_marker",
        "html_form_marker",
        "json_string_marker",
        "path_segment_marker",
        "query_marker",
        "fragment_marker",
        "script_context_marker",
        "style_context_marker",
        "xml_text_marker",
        "xml_attribute_marker",
        "sql_string_marker",
        "sql_numeric_marker",
        "header_marker",
        "state_transition_marker",
    }
)
ALLOWED_VARIANTS = frozenset(
    {
        "none",
        "baseline_marker",
        "reference",
        "reference_shape",
        "source_attested_candidate",
        "matched_negative",
        "negative_control",
        "one_variable_repair",
        "runtime_canary",
        "fresh_replay",
        "unsupported_abstain",
        "unknown",
    }
)
ALLOWED_ORACLES = frozenset(
    {
        "none",
        "reflection",
        "response_shape",
        "parser_shape",
        "dom_shape",
        "typed_state_delta",
        "typed_effect",
        "negative_no_effect",
        "unknown",
    }
)
ALLOWED_NEGATIVE = frozenset({"unknown", "not_observed", "not_required", "matched_triplet"})

_FORBIDDEN_KEY_PARTS = (
    "raw_",
    "raw",
    "literal",
    "wire",
    "response_body",
    "route_literal",
    "evaluator",
    "oracle_answer",
)
_FORBIDDEN_VALUE_PARTS = ("http://", "https://", "javascript:", "<script", "document.cookie")


def _norm(value: Any) -> str:
    return str(value).casefold().replace("-", "_")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_raw(mapping: Mapping[str, Any]) -> None:
    for key, value in mapping.items():
        key_text = str(key).casefold()
        value_text = str(value).casefold()
        if key_text in {"payload", "raw_payload", "raw_value", "literal", "wire", "response", "url"} or any(part in key_text for part in _FORBIDDEN_KEY_PARTS):
            raise ValueError("payload-shape slots contain a raw or evaluator field")
        if any(part in value_text for part in _FORBIDDEN_VALUE_PARTS):
            raise ValueError("payload-shape slots contain a literal execution marker")
        if isinstance(value, (Mapping, list, tuple, set)):
            raise ValueError("payload-shape slots must be bounded scalar values")


def validate_slots(slots: Mapping[str, Any], *, require_syntax: bool = True) -> dict[str, Any]:
    """Validate a model proposal and return canonical abstract slots.

    ``require_syntax=False`` is only for reading legacy PG-349/350 rows.  A
    new candidate or a binder call should keep the default strict mode.
    """

    if not isinstance(slots, Mapping):
        raise ValueError("payload-shape slots must be a mapping")
    _reject_raw(slots)
    allowed = set(SLOT_ORDER) | {"schema_version"}
    unknown = sorted(str(key) for key in slots if str(key) not in allowed)
    if unknown:
        raise ValueError("payload-shape slots contain unsupported fields")
    syntax_present = "syntax_category_ref" in slots
    if require_syntax and not syntax_present:
        raise ValueError("syntax_category_ref is required for PG-361 candidates")
    result = {
        "transport_ref": _norm(slots.get("transport_ref", "unknown")),
        "field_role_ref": _norm(slots.get("field_role_ref", "unknown")),
        "encoding_ref": _norm(slots.get("encoding_ref", "unknown")),
        "syntax_category_ref": _norm(slots.get("syntax_category_ref", "unknown")),
        "payload_shape_ref": _norm(slots.get("payload_shape_ref", "unknown")),
        "probe_variant_ref": _norm(slots.get("probe_variant_ref", "unknown")),
        "oracle_ref": _norm(slots.get("oracle_ref", "unknown")),
        "negative_control_presence_ref": _norm(slots.get("negative_control_presence_ref", "unknown")),
        "safe_to_send": bool(slots.get("safe_to_send") in {True, 1, "1", "true"}),
    }
    checks = (
        ("transport_ref", ALLOWED_TRANSPORTS),
        ("field_role_ref", ALLOWED_FIELD_ROLES),
        ("encoding_ref", ALLOWED_ENCODINGS),
        ("syntax_category_ref", ALLOWED_SYNTAX_CATEGORIES),
        ("payload_shape_ref", ALLOWED_SHAPES),
        ("probe_variant_ref", ALLOWED_VARIANTS),
        ("oracle_ref", ALLOWED_ORACLES),
        ("negative_control_presence_ref", ALLOWED_NEGATIVE),
    )
    for key, values in checks:
        if result[key] not in values:
            raise ValueError(f"{key} is not allow-listed")
    if result["safe_to_send"] and any(result[key] in {"unknown", "none"} for key, _ in checks):
        raise ValueError("safe_to_send requires complete abstract slots")
    result["schema_version"] = SCHEMA_VERSION
    result["syntax_category_present"] = syntax_present
    return result


def target_tokens(slots: Mapping[str, Any], *, require_syntax: bool = True) -> list[str]:
    """Encode only abstract slots for causal next-token training."""

    canonical = validate_slots(slots, require_syntax=require_syntax)
    tokens = ["[TARGET_BOS]"]
    for key in SLOT_ORDER:
        value = int(canonical[key]) if key == "safe_to_send" else canonical[key]
        tokens.append(f"{key}={value}")
    tokens.append("[TARGET_EOS]")
    return tokens


def syntax_attestation(record: Mapping[str, Any], syntax_category_ref: str) -> dict[str, str]:
    """Bind a grammar class to source metadata without exposing raw source."""

    category = _norm(syntax_category_ref)
    if category not in ALLOWED_SYNTAX_CATEGORIES or category in {"none", "unknown"}:
        raise ValueError("syntax category attestation must be concrete and allow-listed")
    source_projection = {
        "implementation_group": str(record.get("implementation_group", "")),
        "surface_template_id": str(record.get("surface_template_id", "")),
        "transport_method": _norm(record.get("transport_method", "")),
        "parameter_role": _norm(record.get("parameter_role", "")),
        "response_shape": _norm(record.get("response_shape", "")),
        "script_surface": _norm(record.get("script_surface", "")),
        "syntax_category_ref": category,
    }
    if not all(source_projection[key] for key in ("implementation_group", "surface_template_id", "transport_method", "parameter_role")):
        raise ValueError("source metadata is insufficient for syntax attestation")
    return {
        "syntax_category_ref": category,
        "syntax_attestation_sha256": _sha(source_projection),
        "syntax_attestation_scope": "evaluator_surface_metadata_only",
    }


__all__ = [
    "ALLOWED_SYNTAX_CATEGORIES",
    "SCHEMA_VERSION",
    "SLOT_ORDER",
    "syntax_attestation",
    "target_tokens",
    "validate_slots",
]
