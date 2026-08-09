"""PG-348 fixture metadata to abstract whole-page projection.

The PG-348 page lanes are intentionally tiny and deterministic.  Their
manifest records are useful provenance for an evaluator, but are not a model
input: identifiers, local paths, source hashes and synthetic oracle names
must remain in a sidecar.  This module converts the bounded, already
de-identified metadata into the same seven-axis token stream used by the
PG-331 web tokenizer.

No request, browser, filesystem or network operation is performed here.  The
only input is an in-memory mapping.  A missing observation is represented by
``not_observed`` in the field manifest/token inventory and drives an explicit
ASK target; it is never silently interpreted as a negative observation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .pg331_web_tokenizer import (
    ONTOLOGY_PATH,
    STATIC_AXIS_ORDER,
    tokenize_web_observation,
)


SCHEMA_VERSION = "pg348-surface-projection-v1"
AXES = tuple(STATIC_AXIS_ORDER)
AXIS_ORDER = AXES
FIELD_STATUS = frozenset({"observed", "absent", "not_observed", "unknown"})
METHODS = frozenset({"GET", "POST"})

# Manifest fields are evaluator/provenance metadata.  They are copied only to
# ``sidecar`` and never interpolated into a context token.
MANIFEST_KEYS = frozenset(
    {
        "challenge_id",
        "local_path",
        "mechanism_id",
        "surface_template_id",
        "implementation_group",
        "transport_method",
        "parameter_role",
        "encoding_chain",
        "response_shape",
        "redirect_shape",
        "script_surface",
        "synthetic_oracle_kind",
        "source_hash",
        "raw_source_for_evaluator_only",
        "training_context_raw",
        # Fixture safety/deduplication attestations.  These are evaluator
        # sidecars and are never model-visible.
        "localhost_only",
        "loopback_only",
        "external_network",
        "state_write",
        "template_index",
        "variant_index",
        "safe_variant_id",
        "training_eligible",
        "memory_promotion_allowed",
        "payload_catalog_promotion_allowed",
        "vulnerability_claim_allowed",
        "observed_fields",
        "field_status",
        "field_capture_manifest",
        # A caller may provide complete axis projections in addition to the
        # compact manifest fields.  They are normalized into observations,
        # not copied verbatim into context.
        "document_structure",
        "navigation",
        "request_transport",
        "response_transport",
        "javascript_surface",
        "failure_feedback",
        "belief_and_replay",
    }
)

# Keys that identify raw/evaluator material.  ``source_hash`` and the
# abstract ``script_surface``/``response_shape`` fields are deliberately not
# matched by these markers.  The scan is recursive so a forbidden value under
# an otherwise harmless nested object cannot reach the tokenizer.
_FORBIDDEN_KEY_EXACT = frozenset(
    {
        "payload",
        "raw_payload",
        "payload_value",
        "probe_value",
        "request_body",
        "request_value",
        "query_value",
        "form_value",
        "response_body",
        "raw_response",
        "body_text",
        "raw_body",
        "html",
        "markup",
        "raw_markup",
        "javascript_source",
        "source_code",
        "oracle",
        "oracle_answer",
        "evaluator",
        "evaluator_answer",
        "expected_answer",
        "target_answer",
        "family",
        "family_label",
        "vulnerability_family",
        "route",
        "route_literal",
        "route_name",
        "url",
        "uri",
        "href",
        "location_url",
        "credential",
        "cookie_value",
        "authorization",
        "authorization_value",
        "secret",
    }
)
_FORBIDDEN_KEY_MARKERS = (
    "raw_",
    "response_body",
    "oracle_answer",
    "evaluator_answer",
    "route_literal",
    "route_name",
    "family_label",
    "vulnerability_family",
    "credential",
    "authorization_value",
    "cookie_value",
)
_FORBIDDEN_VALUE_MARKERS = (
    "<script",
    "javascript:",
    "<img",
    "onerror",
    "select ",
    "union ",
    "union select",
    " or 1=1",
)

# ``raw_source_for_evaluator_only`` is an attestation bit, not source text.
# It is permitted in the manifest only when true and is retained solely in
# the sidecar.  Keep this exception explicit so a similarly named raw field
# cannot bypass the firewall.
_SIDECAR_SAFE_KEYS = frozenset({"raw_source_for_evaluator_only"})

_SYMBOL_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _declared_fields() -> dict[str, tuple[str, ...]]:
    ontology = json.loads(Path(ONTOLOGY_PATH).read_text(encoding="utf-8-sig"))
    axes = dict(ontology.get("axes") or {})
    return {axis: tuple(str(field) for field in list((spec or {}).get("fields") or [])) for axis, spec in axes.items()}


DECLARED_FIELDS = _declared_fields()


def _key_is_forbidden(key: str) -> bool:
    folded = str(key).casefold()
    if folded in _FORBIDDEN_KEY_EXACT:
        return True
    return any(marker in folded for marker in _FORBIDDEN_KEY_MARKERS)


def _scan_forbidden(value: Any, *, path: str = "", allowed_field_keys: frozenset[str] = frozenset()) -> None:
    """Fail closed on raw/evaluator fields before tokenization."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            folded = name.casefold()
            # Ontology field names (for example ``html_lang``) are safe.  A
            # forbidden key remains forbidden even when nested under a valid
            # axis, so this exception is intentionally narrow.
            if _key_is_forbidden(name) and folded not in allowed_field_keys and folded not in _SIDECAR_SAFE_KEYS:
                raise ValueError(f"PG-348 projection rejects raw/evaluator field: {path + '.' if path else ''}{name}")
            _scan_forbidden(child, path=f"{path}.{name}" if path else name, allowed_field_keys=allowed_field_keys)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _scan_forbidden(child, path=f"{path}[{index}]", allowed_field_keys=allowed_field_keys)
    elif isinstance(value, (bytes, bytearray)):
        raise ValueError("PG-348 projection rejects byte/raw material")
    elif isinstance(value, str):
        folded = value.casefold()
        if any(marker in folded for marker in _FORBIDDEN_VALUE_MARKERS):
            raise ValueError(f"PG-348 projection rejects raw executable/query literal at {path or '<value>'}")


def _symbol(value: Any, *, name: str, default: str = "unknown", allow_empty: bool = False) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "present" if value else "absent"
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if not text:
        return "empty" if allow_empty else default
    # A projection value is a category, shape or bounded bucket.  URL/path,
    # query and literal source strings are not abstract categories.
    if any(marker in text for marker in ("://", "javascript:", "<", ">", "?", "#", "&", "=", "\\", '"', "'")):
        raise ValueError(f"PG-348 {name} must be an abstract symbol")
    if text.startswith(("raw_", "payload", "response_body", "oracle", "evaluator", "route", "family")):
        raise ValueError(f"PG-348 {name} must not contain a raw/route/family literal")
    if len(text) > 64 or not _SYMBOL_RE.fullmatch(text):
        raise ValueError(f"PG-348 {name} must be an abstract symbol")
    return text


def _count_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "one" if value else "zero"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= 0:
        return "zero"
    if number == 1:
        return "one"
    if number == 2:
        return "two"
    if number <= 5:
        return "few"
    return "many"


def _length_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = len(str(value))
    if number <= 0:
        return "empty"
    if number <= 64:
        return "short"
    if number <= 1024:
        return "medium"
    return "long"


def _shape(value: Any, *, name: str, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, Mapping):
        # Nested shape projections may carry a bounded ``class``/``kind``
        # value.  Literal members are never copied.
        candidate = value.get("shape", value.get("class", value.get("kind")))
        return _shape(candidate, name=name, default=default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _count_bucket(len(value))
    text = str(value).strip()
    if not text:
        return "empty"
    # Shape categories are symbols.  Permit common abstract shape spellings,
    # but do not let a local route/path or body text through.
    return _symbol(text, name=name, default=default)


def _encoding_chain(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [item for item in re.split(r"[+,|> ]+", value.strip()) if item]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        raise ValueError("PG-348 encoding_chain must be an abstract sequence")
    result: list[str] = []
    for part in parts:
        text = _symbol(part, name="encoding_chain")
        if text not in {"unknown", "not_observed", "empty", "none", "url_percent", "percent", "double_percent", "form_urlencoded", "multipart", "json", "base64", "html_entity", "utf8", "utf_8", "path_segment", "query", "body", "absent"}:
            # Keep append-only abstract categories possible without accepting
            # punctuation/literals; unknown categories are bounded symbols.
            if text.startswith(("http", "www", "route", "url")):
                raise ValueError("PG-348 encoding_chain contains URL/literal material")
        if text == "utf_8":
            text = "utf8"
        if text not in result:
            result.append(text)
    return tuple(result)


def _parameter_roles(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else [value]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(values, 1):
        if isinstance(item, Mapping):
            allowed = {"role", "value_type", "presence", "order", "name_shape"}
            unknown = set(str(key) for key in item) - allowed
            if unknown:
                raise ValueError("PG-348 parameter role contains unsupported fields")
            role = _symbol(item.get("role"), name="parameter_role")
            value_type = _symbol(item.get("value_type", "text"), name="parameter_value_type")
            presence = _symbol(item.get("presence", "present"), name="parameter_presence")
            order = item.get("order", index)
            try:
                order = int(order)
            except (TypeError, ValueError):
                raise ValueError("PG-348 parameter order must be bounded") from None
            if order < 1 or order > 4096:
                raise ValueError("PG-348 parameter order must be bounded")
            name_shape = _shape(item.get("name_shape", "abstract"), name="parameter_name_shape", default="unknown")
        else:
            role = _symbol(item, name="parameter_role")
            value_type, presence, order, name_shape = "text", "present", index, "abstract"
        result.append({"role": role, "value_type": value_type, "presence": presence, "order": order, "name_shape": name_shape})
    return result


def _status(value: Any, *, default: str = "observed") -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().casefold().replace("-", "_")
    if text in FIELD_STATUS:
        return text
    return default


def _declared_status(value: Any, *, name: str) -> str:
    """Normalize an explicit manifest status, failing on typos."""

    text = str(value).strip().casefold().replace("-", "_")
    if text not in FIELD_STATUS:
        raise ValueError(f"PG-348 field status invalid: {name}")
    return text


def _provided_statuses(record: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Read optional field statuses without accepting undeclared fields."""

    raw = record.get("field_capture_manifest", record.get("field_status", record.get("observed_fields")))
    result = {axis: {} for axis in AXES}
    if raw is None:
        return result
    if not isinstance(raw, Mapping):
        raise ValueError("PG-348 observed_fields must be an object")
    # Accept either axis -> field -> status or flat ``axis.field`` keys.
    for axis_key, section in raw.items():
        key = str(axis_key)
        if "." in key and not isinstance(section, Mapping):
            axis, field = key.split(".", 1)
            if axis not in result or field not in DECLARED_FIELDS.get(axis, ()):
                raise ValueError(f"PG-348 observed field is not ontology-declared: {key}")
            result[axis][field] = _declared_status(section, name=key)
            continue
        if key not in result:
            raise ValueError(f"PG-348 observed field axis is unsupported: {key}")
        if not isinstance(section, Mapping):
            raise ValueError(f"PG-348 observed field axis must be an object: {key}")
        for field_key, field_status in section.items():
            field = str(field_key)
            if field not in DECLARED_FIELDS.get(key, ()):
                raise ValueError(f"PG-348 observed field is not ontology-declared: {key}.{field}")
            value = _declared_status(field_status, name=f"{key}.{field}")
            result[key][field] = value
    return result


def _normalize_axis_mapping(value: Any, axis: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-348 {axis} must be an abstract object")
    fields = set(DECLARED_FIELDS.get(axis, ()))
    # Nested structural containers are declared by the ontology emitters even
    # when they are not individual inventory fields.
    extras = {
        "elements",
        "links",
        "parameters",
        "event_handler_kinds",
        # Emitter-level aggregate aliases used by reviewed adapters.  They
        # are still bounded counts/shapes and are normalized below.
        "link_count",
        "script_count",
        "event_handler_count",
        "fetch_count",
        "xhr_count",
        "ast_node_count",
        "body_length",
        "timeout_ms",
        "history_length",
        "probe_count",
        "navigation_event",
        "path_segment_count",
        "query_key_count",
        "form_action_shape",
    }
    unknown = set(str(key) for key in value) - fields - extras
    if unknown:
        raise ValueError(f"PG-348 {axis} contains unsupported fields: {', '.join(sorted(unknown))}")
    result: dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key)
        if name in {"parameters", "elements", "links", "event_handler_kinds"}:
            # Containers are recursively checked by the firewall and then
            # normalized only through the known abstract roles/shapes.
            if name == "parameters":
                result[name] = _parameter_roles(raw)
            elif name == "event_handler_kinds":
                values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) else [raw]
                result[name] = [_symbol(item, name="event_handler_kind") for item in values]
            elif name in {"elements", "links"}:
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
                    raise ValueError(f"PG-348 {axis}.{name} must be an abstract sequence")
                normalized_items: list[dict[str, Any]] = []
                for item in raw:
                    if not isinstance(item, Mapping):
                        raise ValueError(f"PG-348 {axis}.{name} item must be an object")
                    # Keep only ontology-safe structural keys.  Literal names,
                    # text and targets are represented by shape categories.
                    normalized_item: dict[str, Any] = {}
                    for item_key, item_value in item.items():
                        item_name = str(item_key)
                        if item_name not in {"tag", "depth", "sibling_count", "role", "id_shape", "class_shape", "aria_role", "attribute_presence", "text_shape", "text_length", "method", "target_shape", "same_origin", "query_present", "fragment_present"}:
                            raise ValueError(f"PG-348 {axis}.{name} item contains unsupported field: {item_name}")
                        if item_name == "attribute_presence":
                            attrs = item_value if isinstance(item_value, Sequence) and not isinstance(item_value, (str, bytes, bytearray)) else [item_value]
                            normalized_item[item_name] = [_symbol(attr, name="attribute_presence") for attr in attrs]
                        elif item_name in {"depth", "sibling_count", "text_length"}:
                            normalized_item[item_name] = int(item_value) if isinstance(item_value, int) and not isinstance(item_value, bool) else _count_bucket(item_value)
                        elif item_name in {"tag", "method"}:
                            normalized_item[item_name] = _symbol(item_value, name=item_name)
                        elif item_name in {"same_origin", "query_present", "fragment_present"}:
                            normalized_item[item_name] = "present" if item_value is True else "absent" if item_value is False else _symbol(item_value, name=item_name)
                        else:
                            normalized_item[item_name] = _shape(item_value, name=item_name)
                    normalized_items.append(normalized_item)
                result[name] = normalized_items
            continue
        if name in {"method"}:
            result[name] = _symbol(raw, name=f"{axis}.{name}")
        elif name in {"body_length", "content_length", "timeout_ms", "history_length", "probe_count", "link_count", "redirect_hop_count", "query_count", "form_count", "json_field_count", "multipart_part_count", "script_count", "event_handler_count", "fetch_count", "xhr_count", "ast_node_count", "head_count", "meta_count", "style_count", "section_count", "repeated_element_count"}:
            try:
                result[name] = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                result[name] = _count_bucket(raw)
        elif name in {"encoding_chain"}:
            chain = list(_encoding_chain(raw))
            result[name] = "_then_".join(chain) if chain else "absent"
        elif name in {"body_shape", "title_shape", "html_lang", "doctype", "form_action_shape", "target_shape", "path_segment_shape", "query_key_shape", "redirect_chain_shape", "fetch_target_shape", "xhr_target_shape", "id_shape", "class_shape", "text_shape", "element_id_shape", "element_class_shape", "visible_text_shape", "error_shape", "syntax_shape", "body_shape"}:
            result[name] = _shape(raw, name=f"{axis}.{name}")
        elif name in {"status_class"}:
            result[name] = _symbol(raw, name=f"{axis}.{name}")
        else:
            result[name] = _symbol(raw, name=f"{axis}.{name}") if isinstance(raw, (str, bool, int, float)) or raw is None else _shape(raw, name=f"{axis}.{name}")
    return result


def _compact_observation(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, str]], list[str]]:
    """Build the seven-axis observation and its field-status manifest."""

    statuses = _provided_statuses(record)
    observation: dict[str, Any] = {axis: None for axis in AXES}
    supplied: dict[str, set[str]] = {axis: set() for axis in AXES}

    # Full axis projections are accepted when a reviewed collector already
    # has them.  They remain abstract and are normalized field by field.
    for axis in AXES:
        if axis in record:
            observation[axis] = _normalize_axis_mapping(record.get(axis), axis)
            if observation[axis] is not None:
                supplied[axis].update(str(key) for key in observation[axis])
                if axis == "request_transport" and isinstance(observation[axis].get("parameters"), Sequence):
                    supplied[axis].update({"parameter_role", "parameter_name_shape", "parameter_value_type", "parameter_presence", "parameter_order"})
                if axis == "navigation" and isinstance(observation[axis].get("links"), Sequence):
                    supplied[axis].update({"link_count", "link_method", "link_target_shape", "same_origin_bucket", "query_present", "fragment_present"})
                if axis == "javascript_surface" and "event_handler_kinds" in observation[axis]:
                    supplied[axis].add("event_handler_kind")
                aliases = {
                    "body_length": "body_length_bucket",
                    "timeout_ms": "timeout_bucket",
                    "history_length": "history_length_bucket",
                    "probe_count": "probe_count_bucket",
                }
                supplied[axis].update(alias for raw_name, alias in aliases.items() if raw_name in observation[axis])

    method_value = record.get("transport_method")
    role_value = record.get("parameter_role")
    encoding_value = record.get("encoding_chain")
    if method_value is not None or role_value is not None or encoding_value is not None:
        request = dict(observation.get("request_transport") or {})
        if method_value is not None:
            method = _symbol(method_value, name="transport_method").upper()
            if method not in METHODS:
                raise ValueError("PG-348 transport_method must be GET or POST")
            request.update(
                {
                    "method": method,
                    "placement": "query" if method == "GET" else "form",
                    "content_type_class": "none" if method == "GET" else "form",
                }
            )
            supplied["request_transport"].update({"method", "placement", "content_type_class"})
        if encoding_value is not None:
            chain = list(_encoding_chain(encoding_value))
            # PG-331's emitter accepts one abstract category for the encoding
            # field.  Keep the whole ordered chain as a delimiter-free symbol
            # rather than letting Python's list repr become a model token.
            request["encoding_chain"] = "_then_".join(chain) if chain else "absent"
            request["charset_class"] = "utf8"
            supplied["request_transport"].update({"encoding_chain", "charset_class"})
        if role_value is not None:
            parameters = _parameter_roles(role_value)
            request["parameters"] = parameters
            request["query_count"] = len(parameters) if request.get("method") == "GET" else 0
            request["form_count"] = len(parameters) if request.get("method") == "POST" else 0
            request["json_field_count"] = 0
            request["multipart_part_count"] = 0
            request["body_shape"] = "empty" if request.get("method") == "GET" else "form"
            supplied["request_transport"].update({"parameters", "parameter_role", "parameter_name_shape", "parameter_value_type", "parameter_presence", "parameter_order", "query_count", "form_count", "json_field_count", "multipart_part_count", "body_shape"})
        observation["request_transport"] = request

    response_shape = record.get("response_shape")
    redirect_shape = record.get("redirect_shape")
    if response_shape is not None or redirect_shape is not None:
        response = dict(observation.get("response_transport") or {})
        if response_shape is not None:
            if isinstance(response_shape, Mapping):
                nested_response = _normalize_axis_mapping(response_shape, "response_transport")
                response.update(nested_response or {})
                supplied["response_transport"].update(str(key) for key in (nested_response or {}))
                aliases = {"body_length": "body_length_bucket"}
                supplied["response_transport"].update(alias for raw_name, alias in aliases.items() if raw_name in (nested_response or {}))
            else:
                response["body_shape"] = _shape(response_shape, name="response_shape")
                supplied["response_transport"].add("body_shape")
        if redirect_shape is not None:
            if isinstance(redirect_shape, Mapping):
                nested_redirect = _normalize_axis_mapping(redirect_shape, "response_transport")
                response.update(nested_redirect or {})
                supplied["response_transport"].update(str(key) for key in (nested_redirect or {}))
            else:
                redirect = _shape(redirect_shape, name="redirect_shape")
                is_redirect = redirect not in {"none", "empty", "absent", "unknown", "not_observed"}
                response.update(
                    {
                        "redirect_hop_count": 1 if is_redirect else 0,
                        "redirect_location_class": "present" if is_redirect else "none",
                        "redirect_chain_shape": redirect,
                    }
                )
                supplied["response_transport"].update({"redirect_hop_count", "redirect_location_class", "redirect_chain_shape"})
        # Status/content type are intentionally unknown unless supplied by a
        # complete response_transport projection.  Do not infer a typed
        # effect from a fixture shape label.
        observation["response_transport"] = response

    script_value = record.get("script_surface")
    if script_value is not None:
        script = dict(observation.get("javascript_surface") or {})
        if isinstance(script_value, Mapping):
            nested = _normalize_axis_mapping(script_value, "javascript_surface")
            script.update(nested or {})
            supplied["javascript_surface"].update(str(key) for key in (nested or {}))
        else:
            kind = _symbol(script_value, name="script_surface")
            script.update(
                {
                    "script_count": 0 if kind in {"none", "empty", "absent"} else 1,
                    "event_handler_count": 0,
                    "fetch_count": 0,
                    "xhr_count": 0,
                    "ast_node_count": 0,
                    "script_kind": kind,
                    "module_presence": "unknown",
                    "inline_external_class": "unknown",
                    "source_category": "unknown",
                    "sink_category": "unknown",
                    "syntax_shape": "empty" if kind in {"none", "empty", "absent"} else "unknown",
                    "dynamic_code_presence": "unknown",
                    "storage_api_presence": "unknown",
                    "fetch_method": "ABSENT" if kind in {"none", "empty", "absent"} else "unknown",
                    "xhr_method": "ABSENT" if kind in {"none", "empty", "absent"} else "unknown",
                    "fetch_target_shape": "empty" if kind in {"none", "empty", "absent"} else "unknown",
                    "xhr_target_shape": "empty" if kind in {"none", "empty", "absent"} else "unknown",
                    "event_handler_kinds": [],
                }
            )
            supplied["javascript_surface"].update({"script_count", "event_handler_count", "fetch_count", "xhr_count", "ast_node_count", "script_kind", "module_presence", "inline_external_class", "source_category", "sink_category", "syntax_shape", "dynamic_code_presence", "storage_api_presence", "fetch_method", "xhr_method", "fetch_target_shape", "xhr_target_shape", "event_handler_kinds"})
        observation["javascript_surface"] = script

    # Apply explicit status overrides.  A ``not_observed`` field is removed so
    # the tokenizer's inventory emits the explicit marker rather than a
    # synthetic false/zero value.
    manifest: dict[str, dict[str, str]] = {}
    for axis in AXES:
        axis_value = observation.get(axis)
        axis_status: dict[str, str] = {}
        for field in DECLARED_FIELDS.get(axis, ()):
            if field in statuses[axis]:
                status = statuses[axis][field]
            elif axis_value is None:
                status = "not_observed"
            elif field in supplied[axis]:
                # Some ontology fields are emitted from a nested abstract
                # container (for example ``parameter_role`` from
                # ``parameters[*].role``).  ``supplied`` records that the
                # collector supplied such a value even though there is no
                # direct key on the axis mapping.
                raw = axis_value.get(field)
                status = _status(raw, default="observed") if field in axis_value else "observed"
            elif field in axis_value:
                raw = axis_value.get(field)
                status = _status(raw, default="observed")
            else:
                status = "not_observed"
            axis_status[field] = status
            if isinstance(axis_value, dict):
                if status == "not_observed":
                    axis_value.pop(field, None)
                elif status in {"unknown", "absent"}:
                    axis_value[field] = status
        manifest[axis] = axis_status
    return observation, manifest, [f"{axis}.{field}" for axis, fields in manifest.items() for field, status in fields.items() if status in {"not_observed", "unknown"}]


def _axis_presence(tokens: Sequence[str]) -> dict[str, str]:
    keys = {
        "document_presence",
        "navigation_presence",
        "request_transport_presence",
        "response_transport_presence",
        "javascript_presence",
        "failure_feedback_presence",
        "belief_replay_presence",
    }
    result: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in keys:
            result[key] = value
    return result


def _target_for_projection(manifest: Mapping[str, Mapping[str, str]], *, typed_available: bool = False) -> dict[str, Any]:
    missing = {axis: [field for field, status in fields.items() if status in {"not_observed", "unknown"}] for axis, fields in manifest.items()}
    request_missing = missing.get("request_transport", [])
    if "method" in request_missing:
        question = "ask_transport"
    elif "parameter_role" in request_missing:
        question = "ask_parameter_role"
    elif "encoding_chain" in request_missing:
        question = "ask_encoding"
    elif missing.get("response_transport"):
        question = "ask_response"
    elif missing.get("failure_feedback"):
        question = "ask_failure"
    elif missing.get("belief_and_replay"):
        question = "ask_belief"
    elif not typed_available:
        question = "ask_typed"
    else:
        question = "none"

    complete = not any(missing.values()) and typed_available
    if question == "none" and complete:
        return {
            "question": "none",
            "next_action": "assemble_rule_ir",
            "repair_action": "none",
            "transport_ref": "request_method",
            "field_role_ref": "parameter_role",
            "encoding_ref": "encoding_chain",
            "probe_variant_ref": "none",
            "safe_to_send": False,
        }
    return {
        "question": question,
        "next_action": "ask_typed" if question == "ask_typed" else "ask",
        "repair_action": "observe",
        "transport_ref": "request_method" if "method" not in request_missing else "unknown",
        "field_role_ref": "parameter_role" if "parameter_role" not in request_missing else "unknown",
        "encoding_ref": "encoding_chain" if "encoding_chain" not in request_missing else "unknown",
        "probe_variant_ref": "none",
        "safe_to_send": False,
    }


def _target_tokens(target: Mapping[str, Any]) -> list[str]:
    return [
        "[TARGET_BOS]",
        f"question={target['question']}",
        f"next_action={target['next_action']}",
        f"repair_action={target['repair_action']}",
        f"transport_ref={target['transport_ref']}",
        f"field_role_ref={target['field_role_ref']}",
        f"encoding_ref={target['encoding_ref']}",
        f"probe_variant_ref={target['probe_variant_ref']}",
        f"safe_to_send={int(bool(target['safe_to_send']))}",
        "[TARGET_EOS]",
    ]


def project_surface(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one PG-348 manifest record into an abstract seven-axis row.

    The returned ``sidecar`` is evaluator/provenance metadata.  It is kept
    separate from ``context_tokens`` and may contain fixture IDs, paths and a
    synthetic oracle kind, while the model-visible context contains only
    ontology tokens.  Incomplete rows are valid diagnostic output but receive
    an explicit safe ASK target and are never training eligible.
    """

    if not isinstance(record, Mapping):
        raise TypeError("PG-348 surface record must be a mapping")
    # All ontology field names are allowed by the recursive raw-key scanner;
    # all other raw/evaluator keys fail closed.
    allowed_fields = frozenset(field.casefold() for fields in DECLARED_FIELDS.values() for field in fields) | frozenset({"elements", "links", "parameters", "event_handler_kinds", "shape", "class", "kind"})
    _scan_forbidden(record, allowed_field_keys=allowed_fields)
    unknown_top = set(str(key) for key in record) - MANIFEST_KEYS
    if unknown_top:
        raise ValueError(f"PG-348 surface record contains unsupported fields: {', '.join(sorted(unknown_top))}")
    if record.get("training_context_raw") is True:
        raise ValueError("PG-348 training_context_raw must be false")
    if "raw_source_for_evaluator_only" in record and record.get("raw_source_for_evaluator_only") is not True:
        raise ValueError("PG-348 raw_source_for_evaluator_only must be true when present")

    observation, field_manifest, missing_fields = _compact_observation(record)
    tokenized = tokenize_web_observation(observation)
    context_tokens = [str(token) for token in tokenized.get("context_tokens") or []]
    forbidden_tokens = [
        token
        for token in context_tokens
        if any(fragment in token.casefold() for fragment in ("payload", "response_body=", "raw_", "oracle=", "evaluator=", "route=", "family="))
    ]
    if forbidden_tokens:
        raise ValueError("PG-348 context firewall rejected forbidden token")

    # A typed oracle is not copied into context.  It may be attested later by
    # an evaluator-side sidecar; this projection has no authority to send.
    belief = observation.get("belief_and_replay")
    typed_available = bool(isinstance(belief, Mapping) and str(belief.get("typed_available", "")).casefold() in {"present", "true"})
    source_hash = record.get("source_hash")
    source_hash_valid = bool(isinstance(source_hash, str) and _HASH_RE.fullmatch(source_hash))
    if source_hash not in (None, "") and not source_hash_valid:
        raise ValueError("PG-348 source_hash must be a lowercase/uppercase SHA-256 digest")
    target = _target_for_projection(field_manifest, typed_available=typed_available and source_hash_valid)
    target_tokens = _target_tokens(target)
    sidecar_keys = (
        "challenge_id",
        "local_path",
        "mechanism_id",
        "surface_template_id",
        "implementation_group",
        "transport_method",
        "parameter_role",
        "encoding_chain",
        "response_shape",
        "redirect_shape",
        "script_surface",
        "synthetic_oracle_kind",
        "source_hash",
        "raw_source_for_evaluator_only",
        "training_context_raw",
        "localhost_only",
        "loopback_only",
        "external_network",
        "state_write",
        "template_index",
        "variant_index",
        "safe_variant_id",
    )
    sidecar = {key: deepcopy(record[key]) for key in sidecar_keys if key in record}
    local_path = sidecar.get("local_path")
    if isinstance(local_path, str) and "://" in local_path:
        raise ValueError("PG-348 local_path must be a local sidecar path, not a URL")
    sidecar["source_hash_valid"] = source_hash_valid

    status = "complete" if not missing_fields and typed_available and source_hash_valid else "incomplete"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "context_tokens": context_tokens,
        "chunks": tokenized.get("chunks", []),
        "loss_report": {
            **dict(tokenized.get("loss_report") or {}),
            "missing_fields": list(missing_fields),
            "training_eligible": False,
            "status": status,
        },
        "axis_presence": _axis_presence(context_tokens),
        "field_capture_manifest": field_manifest,
        "target_projection": target,
        "target": target,
        "target_tokens": target_tokens,
        "sidecar": sidecar,
        "context_firewall": {
            "forbidden_token_count": 0,
            "sidecars_off_context": True,
            "raw_payload_stored": False,
            "raw_source_stored": False,
            "raw_response_stored": False,
            "raw_response_body_stored": False,
            "raw_oracle_stored": False,
            "raw_url_stored": False,
            "raw_route_stored": False,
            "raw_family_stored": False,
            "oracle_answer_in_context": False,
            "evaluator_answer_in_context": False,
            "raw_url_in_context": False,
            "raw_route_in_context": False,
            "raw_family_in_context": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_payload_stored": False,
        "raw_source_stored": False,
        "raw_response_stored": False,
        "raw_response_body_stored": False,
        "raw_oracle_stored": False,
        "raw_url_stored": False,
        "oracle_answer_in_context": False,
        "evaluator_answer_in_context": False,
    }


# Descriptive aliases make the contract convenient for page collectors while
# keeping one implementation and one schema version.
project_fixture_metadata = project_surface
project_record = project_surface
project_fixture = project_surface


__all__ = [
    "AXES",
    "AXIS_ORDER",
    "DECLARED_FIELDS",
    "FIELD_STATUS",
    "MANIFEST_KEYS",
    "SCHEMA_VERSION",
    "project_fixture_metadata",
    "project_fixture",
    "project_record",
    "project_surface",
]
