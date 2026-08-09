"""PG-350 runtime-only binding from abstract Rule-IR to a local wire.

The model never supplies a literal probe.  It supplies bounded slots such as
``payload_shape_ref=sql_string_marker`` and ``encoding_ref=url_percent``.
An evaluator-side, source-attested template catalog then expands the single
``{{MARKER}}`` placeholder in memory and renders a concrete GET/POST wire.

This module deliberately does not send a request or persist the concrete
value.  ``human_review_wire()`` is the explicit ephemeral display boundary;
``persisted_projection()`` contains hashes and abstract slots only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import html
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit

from app.pg361_payload_shape_slots import ALLOWED_SYNTAX_CATEGORIES


SCHEMA_VERSION = "pg350-runtime-payload-binder-v1"
MARKER_PLACEHOLDER = "{{MARKER}}"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_MARKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")
_FIELD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")

ALLOWED_SHAPES = frozenset(
    {
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
ALLOWED_ENCODINGS = frozenset(
    {
        "identity",
        "url_percent",
        "form_urlencoded",
        "html_entity",
        "javascript_unicode",
        "json_escape",
        "xml_entity",
        "double_layer_order_sensitive",
    }
)
ALLOWED_TRANSPORTS = frozenset({"get_query", "get_path", "get_fragment", "post_form", "post_json", "header"})
ALLOWED_ORACLES = frozenset({"reflection", "response_shape", "parser_shape", "dom_shape", "typed_state_delta", "typed_effect", "negative_no_effect"})
ALLOWED_VARIANTS = frozenset({"reference", "reference_shape", "source_attested_candidate", "runtime_canary", "fresh_replay"})
_SECRET_FIELDS = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "session", "file", "upload"})
_FORBIDDEN_MODEL_KEYS = frozenset({"payload", "raw_payload", "raw_value", "literal", "wire", "response", "response_body", "url", "route_literal", "evaluator_answer", "oracle_answer"})
_FORBIDDEN_TEMPLATE_MARKERS = ("http://", "https://", "javascript:", "document.cookie", "powershell", "cmd.exe", "curl ", "wget ")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical(value))


def _require_hash(value: Any, label: str) -> str:
    text = str(value).casefold()
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return text


def _require_local_origin(value: Any) -> str:
    origin = str(value).rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("runtime binding requires a loopback origin")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("runtime origin must not contain credentials, path, query, or fragment")
    try:
        if not 1 <= int(parsed.port or 0) <= 65535:
            raise ValueError("runtime origin requires a bounded port")
    except (TypeError, ValueError) as error:
        raise ValueError("runtime origin requires a bounded port") from error
    return origin


def _validate_model_slots(rule_ir: Mapping[str, Any]) -> dict[str, str]:
    """Validate only abstract decoder output; no raw value is accepted here."""

    if not isinstance(rule_ir, Mapping):
        raise ValueError("Rule-IR proposal must be a mapping")
    for key, value in rule_ir.items():
        key_text = str(key).casefold()
        if key_text in _FORBIDDEN_MODEL_KEYS or any(part in key_text for part in ("raw_", "response_body", "evaluator_")):
            raise ValueError("Rule-IR proposal contains a raw/evaluator field")
        if isinstance(value, (Mapping, list, tuple)):
            raise ValueError("Rule-IR proposal slots must be bounded scalars")
        value_text = str(value)
        if any(marker in value_text.casefold() for marker in _FORBIDDEN_TEMPLATE_MARKERS):
            raise ValueError("Rule-IR proposal contains a literal execution marker")
    transport = str(rule_ir.get("transport_ref", ""))
    shape = str(rule_ir.get("payload_shape_ref", ""))
    encoding = str(rule_ir.get("encoding_ref", ""))
    oracle = str(rule_ir.get("oracle_ref", ""))
    variant = str(rule_ir.get("probe_variant_ref", ""))
    syntax_present = "syntax_category_ref" in rule_ir
    syntax = str(rule_ir.get("syntax_category_ref", "unknown")).casefold().replace("-", "_")
    safe = rule_ir.get("safe_to_send") in {True, 1, "1", "true"}
    if not safe:
        raise ValueError("Rule-IR is not allowed to bind while safe_to_send is false")
    if transport not in ALLOWED_TRANSPORTS:
        raise ValueError("Rule-IR transport_ref is not allow-listed")
    if shape not in ALLOWED_SHAPES:
        raise ValueError("Rule-IR payload_shape_ref is not allow-listed")
    if encoding not in ALLOWED_ENCODINGS:
        raise ValueError("Rule-IR encoding_ref is not allow-listed")
    if oracle not in ALLOWED_ORACLES:
        raise ValueError("Rule-IR oracle_ref is missing or unknown")
    if syntax_present and syntax not in ALLOWED_SYNTAX_CATEGORIES:
        raise ValueError("Rule-IR syntax_category_ref is not allow-listed")
    if variant not in ALLOWED_VARIANTS:
        raise ValueError("Rule-IR probe_variant_ref is not source-attested")
    result = {
        "transport_ref": transport,
        "payload_shape_ref": shape,
        "encoding_ref": encoding,
        "oracle_ref": oracle,
        "probe_variant_ref": variant,
        "parameter_role_ref": str(rule_ir.get("field_role_ref", "unknown")),
    }
    if syntax_present:
        result["syntax_category_ref"] = syntax
    return result


def _template_body(entry: Mapping[str, Any], shape: str) -> tuple[str, str]:
    template_id = str(entry.get("template_id", ""))
    if not _ID_RE.fullmatch(template_id):
        raise ValueError("template_id must be a bounded identifier")
    if str(entry.get("shape", "")) != shape:
        raise ValueError("template shape does not match Rule-IR payload_shape_ref")
    template = str(entry.get("template", ""))
    if not 1 <= len(template) <= 512 or template.count(MARKER_PLACEHOLDER) != 1:
        raise ValueError("template must contain exactly one bounded {{MARKER}} placeholder")
    if "\x00" in template or any(marker in template.casefold() for marker in _FORBIDDEN_TEMPLATE_MARKERS):
        raise ValueError("template contains an external or credential access marker")
    declared = _require_hash(entry.get("template_sha256"), "template_sha256")
    if declared != _sha256_text(template):
        raise ValueError("template_sha256 does not match evaluator template")
    if entry.get("local_only") is not True or entry.get("non_destructive") is not True:
        raise ValueError("template must be explicitly local_only and non_destructive")
    return template_id, template


def validate_template_catalog(catalog: Mapping[str, Any], *, shape: str, syntax_category: str | None = None) -> dict[str, Any]:
    """Validate an evaluator-side catalog without returning its raw templates."""

    if not isinstance(catalog, Mapping):
        raise ValueError("template catalog must be a mapping")
    entries = catalog.get("templates")
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("template catalog must contain at least one template")
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("template catalog entry must be a mapping")
        entry_shape = str(entry.get("shape", ""))
        if entry_shape not in ALLOWED_SHAPES:
            raise ValueError("template catalog contains an unsupported shape")
        template_id, template = _template_body(entry, entry_shape)
        normalized.append(
            {
                "template_id": template_id,
                "shape": entry_shape,
                "template_sha256": _sha256_text(template),
                "local_only": True,
                "non_destructive": True,
                "stateful_allowed": bool(entry.get("stateful_allowed", False)),
                **(
                    {"syntax_category_ref": str(entry.get("syntax_category_ref")).casefold().replace("-", "_")}
                    if entry.get("syntax_category_ref") is not None
                    else {}
                ),
            }
        )
    matching = [item for item in normalized if item["shape"] == shape]
    if not matching:
        raise ValueError("template catalog has no entry for the requested shape")
    if syntax_category is not None:
        category = str(syntax_category).casefold().replace("-", "_")
        if category not in ALLOWED_SYNTAX_CATEGORIES or category in {"none", "unknown"}:
            raise ValueError("syntax_category must be a concrete allow-listed class")
        if any(item.get("syntax_category_ref") != category for item in matching):
            raise ValueError("template catalog syntax category does not match Rule-IR")
    return {"schema_version": "pg350-evaluator-template-catalog-v1", "shape": shape, "templates": matching}


def _encode(value: str, encoding: str) -> str:
    if encoding == "identity":
        return value
    if encoding == "url_percent":
        return quote(value, safe="")
    if encoding == "form_urlencoded":
        return quote_plus(value, safe="")
    if encoding == "html_entity" or encoding == "xml_entity":
        return html.escape(value, quote=True)
    if encoding == "json_escape":
        return json.dumps(value, ensure_ascii=False)
    if encoding == "javascript_unicode":
        return "".join(f"\\u{ord(char):04x}" for char in value)
    if encoding == "double_layer_order_sensitive":
        return quote(quote(value, safe=""), safe="")
    raise ValueError("unsupported encoding chain")


def _route(runtime: Mapping[str, Any], transport: str) -> dict[str, str]:
    origin = _require_local_origin(runtime.get("target_origin"))
    route = runtime.get("route")
    if not isinstance(route, Mapping):
        raise ValueError("runtime route attestation is missing")
    method = str(route.get("method", "")).upper()
    path = str(route.get("path", ""))
    if method not in {"GET", "POST"} or not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
        raise ValueError("runtime route must be an origin-relative GET/POST path")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("runtime route path must not contain query, fragment, or origin")
    field = str(route.get("field_name", ""))
    if field.casefold() in _SECRET_FIELDS or not _FIELD_RE.fullmatch(field):
        raise ValueError("runtime route field is not an allow-listed non-secret field")
    expected_method = "GET" if transport in {"get_query", "get_path", "get_fragment"} else "POST" if transport in {"post_form", "post_json"} else method
    if method != expected_method:
        raise ValueError("Rule-IR transport and observed route method disagree")
    if transport == "header":
        header_name = str(route.get("header_name", ""))
        if not _HEADER_RE.fullmatch(header_name) or header_name.casefold() in {"cookie", "authorization", "set-cookie"}:
            raise ValueError("runtime header is not allow-listed")
    return {"origin": origin, "method": method, "path": path, "field": field, "header": str(route.get("header_name", ""))}


def _validate_runtime_gate(runtime: Mapping[str, Any]) -> dict[str, Any]:
    required_true = (
        "loopback_only",
        "source_attested",
        "route_attested",
        "field_attested",
        "fresh_reset",
        "candidate_reference_negative",
        "replay_consistency",
    )
    if any(runtime.get(key) is not True for key in required_true) or runtime.get("external_network") is not False:
        raise ValueError("runtime evaluator gate is incomplete")
    authorization_id = str(runtime.get("authorization_id", ""))
    if not _ID_RE.fullmatch(authorization_id):
        raise ValueError("runtime authorization_id is required")
    if bool(runtime.get("stateful_evaluator", False)):
        required_state = ("state_reset_before", "state_reset_after", "database_clean", "teardown")
        if any(runtime.get(key) is not True for key in required_state):
            raise ValueError("stateful evaluator requires reset/clean/teardown attestations")
    return {
        "authorization_id": authorization_id,
        "loopback_only": True,
        "external_network": False,
        "fresh_reset": True,
        "candidate_reference_negative": True,
        "replay_consistency": True,
        "stateful_evaluator": bool(runtime.get("stateful_evaluator", False)),
    }


@dataclass(frozen=True, repr=False)
class RuntimeBoundProbe:
    """An in-memory concrete request; repr and persisted projection hide it."""

    abstract_slots: dict[str, str]
    template_id: str
    marker: str
    method: str
    url: str
    headers: dict[str, str]
    body: str | None
    raw_value: str = field(repr=False)
    wire: str = field(repr=False)
    _attestation: dict[str, Any] = field(repr=False, default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - defensive display behavior
        return f"RuntimeBoundProbe(shape={self.abstract_slots['payload_shape_ref']!r}, method={self.method!r}, raw_value=<ephemeral>)"

    def human_review_wire(self) -> str:
        """Return the concrete wire for an operator's ephemeral local review."""

        return self.wire

    def ephemeral_request(self) -> dict[str, Any]:
        """Return the one-shot request object for the evaluator sender."""

        return {
            "method": self.method,
            "url": self.url,
            "headers": dict(self.headers),
            "body": self.body,
            "raw_value": self.raw_value,
            "ephemeral_only": True,
            "loopback_only": True,
        }

    def persisted_projection(self) -> dict[str, Any]:
        """Return the only shape allowed into a trace/catalog/checkpoint."""

        projection = {
            "schema_version": SCHEMA_VERSION,
            "abstract_slots": dict(self.abstract_slots),
            "template_id": self.template_id,
            "marker_sha256": _sha256_text(self.marker),
            "raw_value_sha256": _sha256_text(self.raw_value),
            "wire_sha256": _sha256_text(self.wire),
            "method": self.method,
            "transport": self.abstract_slots["transport_ref"],
            "evaluator_authorization_id": self._attestation["authorization_id"],
            "loopback_only": True,
            "external_network": False,
            "raw_payload_stored": False,
            "raw_wire_stored": False,
            "evaluator_only": True,
            "training_context_allowed": False,
            "memory_promotion_allowed": False,
        }
        projection["projection_sha256"] = _sha256_json(projection)
        return projection


def bind_runtime_probe(
    rule_ir: Mapping[str, Any],
    runtime: Mapping[str, Any],
    template_catalog: Mapping[str, Any],
    *,
    marker: str,
) -> RuntimeBoundProbe:
    """Expand abstract slots to a concrete local request without sending it."""

    slots = _validate_model_slots(rule_ir)
    gate = _validate_runtime_gate(runtime)
    if not _MARKER_RE.fullmatch(str(marker)):
        raise ValueError("marker must be a bounded runtime canary")
    route = _route(runtime, slots["transport_ref"])
    syntax_category = slots.get("syntax_category_ref")
    catalog = validate_template_catalog(
        template_catalog,
        shape=slots["payload_shape_ref"],
        syntax_category=syntax_category,
    )
    entries = list(template_catalog.get("templates") or [])
    matching = [entry for entry in entries if str(entry.get("shape")) == slots["payload_shape_ref"]]
    # A caller may explicitly restrict which source-attested template IDs are
    # usable for this route.  No unrestricted template substitution is allowed.
    allowed_ids = runtime.get("allowed_template_ids")
    if not isinstance(allowed_ids, (list, tuple, set)):
        raise ValueError("runtime allowed_template_ids is required")
    matching = [entry for entry in matching if str(entry.get("template_id")) in {str(item) for item in allowed_ids}]
    if syntax_category is not None:
        matching = [
            entry
            for entry in matching
            if str(entry.get("syntax_category_ref", "")).casefold().replace("-", "_") == syntax_category
        ]
    if len(matching) != 1:
        raise ValueError("runtime must select exactly one source-attested template")
    template_id, template = _template_body(matching[0], slots["payload_shape_ref"])
    if bool(runtime.get("stateful_evaluator", False)) and matching[0].get("stateful_allowed") is not True:
        raise ValueError("stateful evaluator requires a stateful-allowed template")
    raw_value = template.replace(MARKER_PLACEHOLDER, str(marker))
    encoded = _encode(raw_value, slots["encoding_ref"])
    headers: dict[str, str] = {"Accept": "*/*", "X-Blackbox-Research": "loopback-ephemeral"}
    body: str | None = None
    wire_component = encoded if slots["encoding_ref"] == "url_percent" else quote(encoded, safe="")
    if slots["transport_ref"] == "get_query":
        url = f"{route['origin']}{route['path']}?{quote(route['field'], safe='')}={wire_component}"
    elif slots["transport_ref"] == "get_path":
        url = f"{route['origin']}{route['path'].rstrip('/')}/{wire_component}"
    elif slots["transport_ref"] == "get_fragment":
        url = f"{route['origin']}{route['path']}#{wire_component}"
    elif slots["transport_ref"] == "post_form":
        url = f"{route['origin']}{route['path']}"
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        form_value = encoded if slots["encoding_ref"] == "form_urlencoded" else quote_plus(encoded, safe="")
        body = f"{quote_plus(route['field'])}={form_value}"
    elif slots["transport_ref"] == "post_json":
        url = f"{route['origin']}{route['path']}"
        headers["Content-Type"] = "application/json"
        body = json.dumps({route["field"]: raw_value}, ensure_ascii=False, separators=(",", ":"))
    elif slots["transport_ref"] == "header":
        url = f"{route['origin']}{route['path']}"
        headers[route["header"]] = encoded
    else:  # _validate_model_slots already constrains this
        raise ValueError("unsupported Rule-IR transport")
    if body is None:
        wire = f"{route['method']} {url}\n" + "\n".join(f"{key}: {value}" for key, value in headers.items())
    else:
        wire = f"{route['method']} {url}\n" + "\n".join(f"{key}: {value}" for key, value in headers.items()) + f"\n\n{body}"
    # ``catalog`` is intentionally only used for validation; retaining the
    # normalized shape prevents an accidental raw-template persistence path.
    _ = catalog
    return RuntimeBoundProbe(
        abstract_slots=slots,
        template_id=template_id,
        marker=str(marker),
        method=route["method"],
        url=url,
        headers=headers,
        body=body,
        raw_value=raw_value,
        wire=wire,
        _attestation=gate,
    )


__all__ = [
    "ALLOWED_ENCODINGS",
    "ALLOWED_ORACLES",
    "ALLOWED_SHAPES",
    "ALLOWED_TRANSPORTS",
    "RuntimeBoundProbe",
    "SCHEMA_VERSION",
    "bind_runtime_probe",
    "validate_template_catalog",
]
