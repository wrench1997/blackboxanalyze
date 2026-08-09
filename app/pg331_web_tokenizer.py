"""Ontology-driven whole-web abstract tokenizer for PG-331.

The tokenizer is intentionally structural.  It emits an ordered token stream
for document/DOM, navigation, request, response, JavaScript, failure and
belief/replay observations.  It never emits literal URLs, parameter values,
HTML/JS source, response bodies, credentials, payloads or evaluator answers.
Missing observations are explicit ``<axis>_presence=not_observed`` tokens so
the next-token model can ask instead of silently treating absence as false.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "research" / "pg331_web_token_ontology_v1.json"
SCHEMA_VERSION = "pg331-web-tokenizer-v1"
CHUNK_SIZE = 256
MAX_STRUCTURAL_ITEMS = 4096

STATIC_AXIS_ORDER = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)
RAW_KEYS = frozenset(
    {
        "raw_url",
        "url",
        "href",
        "action",
        "raw_payload",
        "payload",
        "raw_response",
        "response_body",
        "body_text",
        "html_source",
        "javascript_source",
        "source_code",
        "cookie_value",
        "authorization_value",
        "secret",
        "oracle_answer",
        "evaluator_answer",
        # Side-channel labels and literal route metadata are evaluator/source
        # fields, never model-visible webpage structure.
        "family",
        "family_label",
        "vulnerability_family",
        "route_literal",
        "route_name",
        "surface_id",
        "oracle",
        "evaluator",
        "typed_effect",
        "expected_answer",
        "target_answer",
    }
)
FORBIDDEN_LITERAL_MARKERS = ("<script", "javascript:", "select ", "union ", " or 1=1", "<img", "onerror")
SAFE_TAGS = frozenset(
    {
        "a",
        "button",
        "body",
        "div",
        "form",
        "head",
        "html",
        "img",
        "input",
        "label",
        "li",
        "link",
        "main",
        "meta",
        "nav",
        "option",
        "p",
        "script",
        "select",
        "section",
        "style",
        "table",
        "textarea",
        "title",
        "ul",
    }
)
SAFE_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "UNKNOWN"})
SHAPE_SENTINELS = frozenset({"unknown", "not_observed", "empty", "absent", "blocked", "present"})


def _ontology() -> dict[str, Any]:
    value = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema_version") != "pg331-web-token-ontology-v1":
        raise ValueError("PG-331 ontology schema mismatch")
    return value


def ontology_sha256() -> str:
    # Use the same canonical JSON digest as the audit/manifest so formatting
    # changes do not create a false tokenizer/ontology mismatch.
    canonical = json.dumps(_ontology(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bucket_count(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "zero" if number <= 0 else "one" if number == 1 else "two" if number == 2 else "few" if number <= 5 else "many"


def _bucket_length(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        number = int(value)
    except (TypeError, ValueError):
        text = str(value)
        number = len(text)
    return "empty" if number <= 0 else "short" if number <= 64 else "medium" if number <= 1024 else "long"


def _shape(value: Any) -> str:
    if value is None or value == "":
        return "empty"
    text = str(value)
    folded = text.casefold()
    # These values are abstract observation states, not ordinary alpha
    # literals.  Preserve them so an unavailable document field cannot look
    # like a confidently observed word-shaped value.
    if folded in SHAPE_SENTINELS:
        return folded
    if any(marker in folded for marker in FORBIDDEN_LITERAL_MARKERS):
        raise ValueError("raw executable or query-like literal supplied to abstract tokenizer")
    if re.fullmatch(r"[0-9]+", text):
        return "numeric"
    if re.fullmatch(r"[A-Za-z]+", text):
        return "alpha"
    if re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return "word_mixed"
    if "/" in text or "\\" in text:
        return "path_like"
    if "://" in text:
        return "origin_like"
    if "{" in text or "[" in text:
        return "structured_like"
    return "mixed"


def _digest_bucket(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()
    return f"b{digest[:2]}"


def _enum(value: Any, *, allowed: Iterable[str] = ()) -> str:
    text = str(value if value is not None else "unknown").strip().casefold().replace("-", "_").replace(" ", "_")
    choices = {str(item).casefold() for item in allowed}
    return text if not choices or text in choices else "other"


def _token(key: str, value: Any) -> str:
    text = str(value)
    if not text or "=" in text or any(char in text for char in "\r\n"):
        raise ValueError(f"invalid abstract token value for {key}")
    return f"{key}={text}"


def _section(
    tokens: list[str],
    losses: list[dict[str, Any]],
    axis: str,
    presence_key: str,
    value: Any,
    emit: Any,
    fields: Sequence[str] = (),
) -> None:
    """Emit a section or an explicit not-observed marker."""
    if not isinstance(value, Mapping):
        tokens.append(_token(presence_key, "not_observed"))
        losses.append({"axis": axis, "kind": "not_observed"})
        return
    tokens.append(_token(presence_key, "observed"))
    tokens.append(_token("axis_begin", axis))
    emit(value, tokens, losses)
    _emit_declared_fields(axis, value, fields, tokens)
    tokens.append(_token("axis_end", axis))


_NESTED_FIELD_SOURCES: dict[str, dict[str, tuple[str, str | None]]] = {
    "document_structure": {
        "body_section_order": ("section_order", None),
        "dom_tag": ("elements", "tag"),
        "dom_depth_bucket": ("elements", "depth"),
        "dom_sibling_count_bucket": ("elements", "sibling_count"),
        "element_role": ("elements", "role"),
        "element_id_shape": ("elements", "id_shape"),
        "element_class_shape": ("elements", "class_shape"),
        "aria_role": ("elements", "aria_role"),
        "attribute_presence": ("elements", "attribute_presence"),
        "visible_text_shape": ("elements", "text_shape"),
        "text_length_bucket": ("elements", "text_length"),
    },
    "navigation": {
        "link_count": ("links", None),
        "link_method": ("links", "method"),
        "link_target_shape": ("links", "target_shape"),
        "same_origin_bucket": ("links", "same_origin"),
        "query_present": ("links", "query_present"),
        "fragment_present": ("links", "fragment_present"),
        "navigation_event": ("navigation_event", None),
    },
    "request_transport": {
        "content_length_bucket": ("content_length", None),
        "parameter_role": ("parameters", "role"),
        "parameter_name_shape": ("parameters", "name_shape"),
        "parameter_value_type": ("parameters", "value_type"),
        "parameter_presence": ("parameters", "presence"),
        "parameter_order": ("parameters", "order"),
    },
    "response_transport": {
        "body_length_bucket": ("body_length", None),
    },
    "javascript_surface": {
        "event_handler_kind": ("event_handler_kinds", None),
    },
    "failure_feedback": {
        "timeout_bucket": ("timeout_ms", None),
    },
    "belief_and_replay": {
        "history_length_bucket": ("history_length", None),
        "probe_count_bucket": ("probe_count", None),
    },
}

_SHAPE_FIELD_HINTS = (
    "shape",
    "path",
    "target",
    "action",
    "name",
    "id",
    "class",
    "text",
    "location",
    "body",
    "value",
    "hash",
    "origin",
    "route",
    "url",
)


def _nested_field_value(value: Mapping[str, Any], axis: str, field: str) -> tuple[bool, Any]:
    """Return a declared field or a bounded aggregate from its nested rows."""

    if field in value:
        return True, value[field]
    source = _NESTED_FIELD_SOURCES.get(axis, {}).get(field)
    if source is None:
        return False, None
    parent, child = source
    if parent not in value:
        return False, None
    raw = value[parent]
    if child is None:
        return True, raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return True, [item.get(child) for item in raw if isinstance(item, Mapping) and child in item]
    return False, None


def _abstract_declared_field(field: str, raw: Any) -> str:
    """Project a field to a non-literal token value without silent dropping."""

    if raw is None:
        return "unknown"
    if isinstance(raw, bool):
        return "present" if raw else "absent"
    if isinstance(raw, Mapping):
        return "object"
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return _bucket_count(len(raw))
    name = field.casefold()
    if "count" in name or name.endswith(("_order", "_hop_count")):
        return _bucket_count(raw)
    if "length" in name or name.endswith("_bytes"):
        return _bucket_length(raw)
    if any(hint in name for hint in _SHAPE_FIELD_HINTS):
        return _shape(raw)
    value = _enum(raw)
    # A category may be an enum, but never preserve a long or URL-like value.
    if len(value) > 32 or any(char in value for char in ("/", "\\", ":", "?", "#", "&", "<", ">")):
        return _shape(raw)
    return value


def _emit_declared_fields(axis: str, value: Mapping[str, Any], fields: Sequence[str], tokens: list[str]) -> None:
    """Emit one axis-prefixed token for every ontology-declared field.

    The detailed emitters above provide ordered/nested structure.  These
    inventory tokens make the contract explicit: a newly added field cannot
    disappear silently just because an emitter forgot a projection.  Missing
    fields are represented as ``not_observed`` and remain visible to the ASK
    head and the evaluator-side loss report.
    """

    for field in fields:
        name = str(field)
        present, raw = _nested_field_value(value, axis, name)
        projected = _abstract_declared_field(name, raw) if present else "not_observed"
        tokens.append(_token(f"{axis}_field_{name}", projected))


def _raw_field_paths(value: Any, prefix: str = "") -> list[str]:
    """Find raw/evaluator keys recursively so nested metadata cannot leak."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            lowered = name.casefold()
            path = f"{prefix}.{name}" if prefix else name
            if lowered in RAW_KEYS or lowered.startswith("raw_"):
                paths.append(path)
            else:
                paths.extend(_raw_field_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # Scan every nested item.  A bounded raw-key scan would permit a
        # forbidden field after item 64 to escape the context firewall.
        for index, child in enumerate(value):
            paths.extend(_raw_field_paths(child, f"{prefix}[{index}]"))
    return paths


def _emit_document(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    for field in ("doctype", "html_lang", "title_shape"):
        tokens.append(_token(f"doc_{field}", _shape(value.get(field))))
    for field in ("head_count", "meta_count", "style_count", "script_count", "section_count", "repeated_element_count"):
        tokens.append(_token(f"doc_{field}", _bucket_count(value.get(field))))
    elements = value.get("elements")
    if isinstance(elements, Sequence) and not isinstance(elements, (str, bytes)):
        tokens.append(_token("dom_element_count", _bucket_count(len(elements))))
        for index, element in enumerate(elements[:MAX_STRUCTURAL_ITEMS]):
            if not isinstance(element, Mapping):
                losses.append({"axis": "document_structure", "kind": "element_not_mapping", "index": index})
                continue
            tag = _enum(element.get("tag"), allowed=SAFE_TAGS)
            tokens.extend(
                [
                    _token("dom_element_index", _bucket_count(index + 1)),
                    _token("dom_tag", tag),
                    _token("dom_depth", _bucket_count(element.get("depth"))),
                    _token("dom_sibling_count", _bucket_count(element.get("sibling_count"))),
                    _token("element_role", _enum(element.get("role"))),
                    _token("element_id_shape", _shape(element.get("id_shape"))),
                    _token("element_class_shape", _shape(element.get("class_shape"))),
                    _token("text_shape", _shape(element.get("text_shape"))),
                    _token("text_length", _bucket_length(element.get("text_length"))),
                ]
            )
            attributes = list(element.get("attribute_presence") or [])
            for attr in attributes[:MAX_STRUCTURAL_ITEMS]:
                tokens.append(_token("attribute_presence", _enum(attr)))
            if len(attributes) > MAX_STRUCTURAL_ITEMS:
                losses.append({"axis": "document_structure", "kind": "bounded_attribute_overflow", "count": len(attributes) - MAX_STRUCTURAL_ITEMS})
        if len(elements) > MAX_STRUCTURAL_ITEMS:
            tokens.append(_token("dom_overflow_count", _bucket_count(len(elements) - MAX_STRUCTURAL_ITEMS)))
            losses.append({"axis": "document_structure", "kind": "bounded_overflow", "count": len(elements) - MAX_STRUCTURAL_ITEMS})
    else:
        tokens.append(_token("dom_element_count", "not_observed"))


def _emit_navigation(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    links = value.get("links")
    tokens.append(_token("nav_link_count", _bucket_count(len(links) if isinstance(links, Sequence) and not isinstance(links, (str, bytes)) else value.get("link_count"))))
    if isinstance(links, Sequence) and not isinstance(links, (str, bytes)):
        for index, link in enumerate(links[:MAX_STRUCTURAL_ITEMS]):
            if not isinstance(link, Mapping):
                losses.append({"axis": "navigation", "kind": "link_not_mapping", "index": index})
                continue
            tokens.extend(
                [
                    _token("link_index", _bucket_count(index + 1)),
                    _token("link_method", _enum(link.get("method", "GET"), allowed=SAFE_METHODS)),
                    _token("link_target_shape", _shape(link.get("target_shape"))),
                    _token("link_same_origin", _enum(link.get("same_origin", "unknown"))),
                    _token("link_query_present", _enum(link.get("query_present", "unknown"))),
                    _token("link_fragment_present", _enum(link.get("fragment_present", "unknown"))),
                ]
            )
        if len(links) > MAX_STRUCTURAL_ITEMS:
            losses.append({"axis": "navigation", "kind": "bounded_overflow", "count": len(links) - MAX_STRUCTURAL_ITEMS})
    for field in ("path_segment_count", "query_key_count", "form_action_shape"):
        tokens.append(_token(f"nav_{field}", _bucket_count(value.get(field)) if field.endswith("count") else _shape(value.get(field))))


def _emit_request(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    tokens.extend(
        [
            _token("request_method", _enum(value.get("method"), allowed=SAFE_METHODS)),
            _token("request_placement", _enum(value.get("placement"))),
            _token("request_content_type", _enum(value.get("content_type_class"))),
            _token("request_encoding_chain", _enum(value.get("encoding_chain"))),
            _token("request_charset", _enum(value.get("charset_class"))),
            _token("request_body_shape", _shape(value.get("body_shape"))),
            _token("request_query_count", _bucket_count(value.get("query_count"))),
            _token("request_form_count", _bucket_count(value.get("form_count"))),
            _token("request_json_count", _bucket_count(value.get("json_field_count"))),
            _token("request_multipart_count", _bucket_count(value.get("multipart_part_count"))),
            _token("request_header_presence", _enum(value.get("header_presence_class"))),
            _token("request_cookie_presence", _enum(value.get("cookie_presence_class"))),
            _token("request_csrf_presence", _enum(value.get("csrf_presence_class"))),
            _token("request_content_length", _bucket_length(value.get("content_length"))),
        ]
    )
    parameters = value.get("parameters")
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes)):
        for index, parameter in enumerate(parameters[:MAX_STRUCTURAL_ITEMS]):
            if not isinstance(parameter, Mapping):
                losses.append({"axis": "request_transport", "kind": "parameter_not_mapping", "index": index})
                continue
            tokens.extend(
                [
                    _token("param_index", _bucket_count(index + 1)),
                    _token("param_role", _enum(parameter.get("role"))),
                    _token("param_name_shape", _shape(parameter.get("name_shape"))),
                    _token("param_value_type", _enum(parameter.get("value_type"))),
                    _token("param_presence", _enum(parameter.get("presence"))),
                    _token("param_order", _bucket_count(parameter.get("order", index + 1))),
                ]
            )
        if len(parameters) > MAX_STRUCTURAL_ITEMS:
            losses.append({"axis": "request_transport", "kind": "bounded_overflow", "count": len(parameters) - MAX_STRUCTURAL_ITEMS})


def _emit_response(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    for field, allowed in (("status_class", ("1xx", "2xx", "3xx", "4xx", "5xx", "transport", "unknown")), ("content_type_class", ("html", "json", "text", "other", "unknown")), ("connection_outcome", ())):
        tokens.append(_token(f"response_{field}", _enum(value.get(field), allowed=allowed)))
    for field in ("body_length", "redirect_hop_count"):
        tokens.append(_token(f"response_{field}", _bucket_length(value.get(field)) if field == "body_length" else _bucket_count(value.get(field))))
    for field in ("body_shape", "charset_class", "header_presence_class", "cache_shape", "redirect_location_class", "redirect_chain_shape"):
        tokens.append(_token(f"response_{field}", _shape(value.get(field)) if field in {"body_shape", "redirect_chain_shape"} else _enum(value.get(field))))


def _emit_javascript(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    for field in ("script_count", "event_handler_count", "fetch_count", "xhr_count", "ast_node_count"):
        tokens.append(_token(f"js_{field}", _bucket_count(value.get(field))))
    for field in ("script_kind", "module_presence", "inline_external_class", "source_category", "sink_category", "syntax_shape", "dynamic_code_presence", "storage_api_presence"):
        tokens.append(_token(f"js_{field}", _enum(value.get(field))))
    for field in ("fetch_method", "xhr_method"):
        tokens.append(_token(f"js_{field}", _enum(value.get(field), allowed=SAFE_METHODS | {"ABSENT"})))
    for field in ("fetch_target_shape", "xhr_target_shape"):
        tokens.append(_token(f"js_{field}", _shape(value.get(field))))
    events = list(value.get("event_handler_kinds") or [])
    for event in events[:MAX_STRUCTURAL_ITEMS]:
        tokens.append(_token("js_event", _enum(event)))
    if len(events) > MAX_STRUCTURAL_ITEMS:
        losses.append({"axis": "javascript_surface", "kind": "bounded_event_overflow", "count": len(events) - MAX_STRUCTURAL_ITEMS})


def _emit_failure(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    for field in ("failure_class", "failure_stage", "error_shape", "parse_error_class", "encoding_error_class", "redirect_error_class", "blocked_reason_class", "environment_failure_class", "previous_action", "next_action", "repair_delta_axis", "repair_outcome"):
        tokens.append(_token(f"failure_{field}", _enum(value.get(field))))
    tokens.append(_token("failure_timeout", _bucket_length(value.get("timeout_ms"))))


def _emit_belief(value: Mapping[str, Any], tokens: list[str], losses: list[dict[str, Any]]) -> None:
    for field in ("observation_presence", "observation_delta_axis", "belief_prior_bucket", "belief_posterior_bucket", "belief_delta_axis", "history_action", "typed_available", "evidence_present", "negative_control", "fresh_reset", "replay_ready", "reference_present", "candidate_present", "step_budget", "evidence_hash_present"):
        tokens.append(_token(f"belief_{field}", _enum(value.get(field))))
    # PG-343 role/step binding is optional for legacy observations but
    # mandatory for new target-conditioned collection.  These are bounded
    # process symbols, not evaluator answers or probe literals.
    if "probe_role" in value:
        tokens.append(_token("belief_probe_role", _enum(value.get("probe_role"), allowed=("candidate", "reference", "negative", "replay"))))
    if "process_step" in value:
        tokens.append(_token("belief_process_step", _enum(value.get("process_step"), allowed=("preflight", "baseline", "failure", "repair", "replay"))))
    for field in ("history_length", "probe_count"):
        tokens.append(_token(f"belief_{field}", _bucket_count(value.get(field))))


EMITTERS = {
    "document_structure": _emit_document,
    "navigation": _emit_navigation,
    "request_transport": _emit_request,
    "response_transport": _emit_response,
    "javascript_surface": _emit_javascript,
    "failure_feedback": _emit_failure,
    "belief_and_replay": _emit_belief,
}


def tokenize_web_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return model tokens plus evaluator-side loss metadata.

    ``loss_report`` is not part of the model context.  It lets the collector
    see which axes were missing/overflowed and prevents silent information
    loss from becoming a training example.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("PG-331 observation must be a mapping")
    tokens: list[str] = []
    losses: list[dict[str, Any]] = []
    omitted_raw = _raw_field_paths(observation)
    if omitted_raw:
        losses.append({"kind": "raw_field_omitted", "fields": sorted(omitted_raw)})
    ontology = _ontology()
    axes = dict(ontology.get("axes") or {})
    axis_order = tuple(axis for axis in STATIC_AXIS_ORDER if axis in axes) + tuple(axis for axis in axes if axis not in STATIC_AXIS_ORDER)
    for axis in axis_order:
        emitter = EMITTERS.get(axis)
        if emitter is None:
            losses.append({"axis": axis, "kind": "no_emitter"})
            tokens.append(_token(f"{axis}_presence", "not_observed"))
            continue
        spec = dict(axes.get(axis) or {})
        presence_key = str(spec.get("presence_token") or f"{axis}_presence")
        fields = tuple(str(field) for field in list(spec.get("fields") or []))
        _section(tokens, losses, axis, presence_key, observation.get(axis), emitter, fields)
    # The model-visible stream is ordered and canonical.  Long pages are
    # framed into explicit chunks rather than silently truncated; each chunk
    # repeats only bounded boundary metadata and preserves the original order.
    full_tokens = list(tokens)
    chunk_count = max(1, (len(full_tokens) + CHUNK_SIZE - 1) // CHUNK_SIZE)
    chunks: list[list[str]] = []
    chunked_tokens: list[str] = []
    for index in range(chunk_count):
        chunk = full_tokens[index * CHUNK_SIZE : (index + 1) * CHUNK_SIZE]
        framed = [
            _token("chunk_boundary", "begin"),
            _token("chunk_index", _bucket_count(index + 1)),
            _token("chunk_count", _bucket_count(chunk_count)),
            _token("chunk_shape", _bucket_count(len(chunk))),
            _token("chunk_digest", _digest_bucket("|".join(chunk))),
            *chunk,
            _token("chunk_boundary", "end"),
        ]
        chunks.append(framed)
        chunked_tokens.extend(framed)
    # The loss report is explicitly separate so no oracle/raw field can leak
    # through metadata.  ``token_count`` is the unframed canonical count;
    # ``model_token_count`` is what the decoder actually receives.
    return {
        "schema_version": SCHEMA_VERSION,
        "ontology_sha256": ontology_sha256(),
        "context_tokens": chunked_tokens,
        "chunks": chunks,
        "loss_report": {
            "missing_axes": [item["axis"] for item in losses if item.get("kind") == "not_observed"],
            "losses": losses,
            "raw_fields_omitted": sorted(omitted_raw),
            "token_count": len(full_tokens),
            "model_token_count": len(chunked_tokens),
            "chunk_count": chunk_count,
            "chunk_size": CHUNK_SIZE,
            "lossy": bool(losses),
            "training_eligible": not bool(losses),
        },
    }


def build_vocabulary(token_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Build an append-only vocabulary from ontology tokens and observations."""

    ontology = _ontology()
    values: set[str] = set(str(token) for token in (ontology.get("reserved_tokens") or {}).get("universal", []))
    values.update(str(token) for token in (ontology.get("reserved_tokens") or {}).get("bucket_policy", []))
    for record in token_records:
        values.update(str(token) for token in record.get("context_tokens") or [])
    ordered = ["[PAD]", "[UNK]"] + sorted(values - {"[PAD]", "[UNK]"})
    return {token: index for index, token in enumerate(dict.fromkeys(ordered))}


__all__ = ["CHUNK_SIZE", "MAX_STRUCTURAL_ITEMS", "SCHEMA_VERSION", "build_vocabulary", "ontology_sha256", "tokenize_web_observation"]
