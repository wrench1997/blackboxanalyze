"""Layered source-token -> Rule IR -> IR-token compression.

This is a local, non-executing projection adapter.  It may inspect an
authorized page snapshot, GET/POST manifest and JavaScript source, but it
emits only bounded categorical tokens and hashes.  The raw page/code is never
returned to the model-facing object.  Typed oracle authority remains outside
the IR token stream and is still required by the replay acceptance gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from typing import Any, Mapping, Sequence


SOURCE_SCHEMA_VERSION = "sift-layered-source-token-v1"
IR_SCHEMA_VERSION = "sift-layered-rule-ir-token-v1"
LAYERED_SCHEMA_VERSION = "sift-layered-token-ir-v1"
MAX_SOURCE_TOKENS = 256
MAX_IR_TOKENS = 96
_METHODS = frozenset({"GET", "POST"})
_ID_RE = re.compile(r"^[A-Za-z0-9_.:/+\-]{1,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset({"body", "raw_body", "raw_probe", "request_body", "response_body", "password", "token", "cookie", "authorization", "source", "code"})
_JS_APIS = ("document", "fetch", "xmlhttprequest", "location", "innerhtml", "eval", "settimeout", "postmessage")
_JS_KEYWORDS = ("if", "else", "for", "while", "function", "return", "const", "let", "var", "try", "catch", "throw", "new")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _bucket(number: int) -> str:
    if number <= 0:
        return "0"
    if number <= 4:
        return "1-4"
    if number <= 16:
        return "5-16"
    return "17+"


def _name_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _id(value: Any, label: str) -> str:
    result = str(value or "")
    if not _ID_RE.fullmatch(result):
        raise ValueError(f"{label} is not a bounded identifier")
    return result


class _HtmlProjectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[dict[str, Any]] = []
        self.script_lengths: list[int] = []
        self.text_length = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tokens.append({"layer": "source", "modality": "html", "kind": "tag", "value": str(tag).casefold()})
        for name, value in attrs:
            name_text = str(name).casefold()
            token = {"layer": "source", "modality": "html", "kind": "attribute", "value": name_text}
            if name_text in {"name", "id", "class", "action", "href", "src"}:
                token["value_hash"] = _name_hash(value or "")
            self.tokens.append(token)
            if name_text == "method":
                method = str(value or "GET").upper()
                self.tokens.append({"layer": "source", "modality": "html", "kind": "form_method", "value": method if method in _METHODS else "OTHER"})

    def handle_data(self, data: str) -> None:
        length = len(str(data))
        self.text_length += length
        if self.get_starttag_text() and self.get_starttag_text().lower().startswith("<script"):
            self.script_lengths.append(length)


def tokenize_html_snapshot(html: str) -> dict[str, Any]:
    """Parse structure only; never emit literal text or attribute values."""

    text = str(html)
    parser = _HtmlProjectionParser()
    parser.feed(text)
    parser.close()
    tokens = parser.tokens[:MAX_SOURCE_TOKENS]
    tokens.append({"layer": "source", "modality": "html", "kind": "text_length_bucket", "value": _bucket(parser.text_length)})
    tokens.append({"layer": "source", "modality": "html", "kind": "script_count", "value": _bucket(len(parser.script_lengths))})
    return {"schema_version": SOURCE_SCHEMA_VERSION, "modality": "html", "source_sha256": _sha256(text), "token_count": len(tokens), "tokens": tokens, "raw_retained": False}


def tokenize_action_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    method = str(manifest.get("method", "")).upper()
    if method not in _METHODS:
        raise ValueError("action manifest method must be GET or POST")
    placement = _id(manifest.get("placement"), "action.placement")
    route_id = _id(manifest.get("route_template_id"), "action.route_template_id")
    tokens = [
        {"layer": "source", "modality": "transport", "kind": "method", "value": method},
        {"layer": "source", "modality": "transport", "kind": "placement", "value": placement},
        {"layer": "source", "modality": "transport", "kind": "route_template", "value_hash": _name_hash(route_id)},
    ]
    fields = [str(item) for item in manifest.get("form_field_names", [])]
    if method == "POST":
        tokens.append({"layer": "source", "modality": "transport", "kind": "form_field_count", "value": _bucket(len(fields))})
        tokens.extend({"layer": "source", "modality": "transport", "kind": "form_field", "value_hash": _name_hash(field)} for field in fields[:16])
    return {"schema_version": SOURCE_SCHEMA_VERSION, "modality": "transport", "token_count": len(tokens), "tokens": tokens, "raw_retained": False}


def tokenize_javascript_snapshot(js_source: str) -> dict[str, Any]:
    """Lex a local script into allow-listed categories without executing it."""

    text = str(js_source)
    lower = text.casefold()
    tokens: list[dict[str, Any]] = []
    for api in _JS_APIS:
        count = len(re.findall(rf"\b{re.escape(api)}\b", lower))
        if count:
            tokens.append({"layer": "source", "modality": "javascript", "kind": "api", "value": api, "count_bucket": _bucket(count)})
    for keyword in _JS_KEYWORDS:
        count = len(re.findall(rf"\b{re.escape(keyword)}\b", lower))
        if count:
            tokens.append({"layer": "source", "modality": "javascript", "kind": "keyword", "value": keyword, "count_bucket": _bucket(count)})
    event_count = len(re.findall(r"\bon[a-z][a-z0-9_]*\s*=", lower))
    if event_count:
        tokens.append({"layer": "source", "modality": "javascript", "kind": "event_handler", "value": "present", "count_bucket": _bucket(event_count)})
    tokens.append({"layer": "source", "modality": "javascript", "kind": "length_bucket", "value": _bucket(len(text))})
    return {"schema_version": SOURCE_SCHEMA_VERSION, "modality": "javascript", "source_sha256": _sha256(text), "token_count": len(tokens), "tokens": tokens[:MAX_SOURCE_TOKENS], "raw_retained": False, "execution_performed": False}


def _failure_state(signature: Mapping[str, Any]) -> bool:
    kind = str(signature.get("kind", ""))
    return kind in {"candidate_without_typed_effect", "oracle_unavailable", "method_disagreement", "budget_exhausted"} or (kind == "no_surface_delta" and str(signature.get("failed_gate", "")) != "matched_negative_control")


def summarize_rule_ir(source_layers: Sequence[Mapping[str, Any]], response_projection: Mapping[str, Any], failure_signature: Mapping[str, Any]) -> dict[str, Any]:
    """Compress source tokens into family-free abstract Rule IR slots."""

    observed_methods = {
        str(item).upper()
        for item in failure_signature.get("methods_seen", [])
        if str(item).upper() in _METHODS
    }
    source_methods = {
        str(token.get("value", "")).upper()
        for layer in source_layers
        for token in layer.get("tokens", [])
        if token.get("kind") == "method" and str(token.get("value", "")).upper() in _METHODS
    }
    methods = sorted(observed_methods or source_methods)
    modalities = sorted({str(layer.get("modality", "")) for layer in source_layers})
    failure = _failure_state(failure_signature)
    transition = str(response_projection.get("transition_delta", "none"))
    slots = [
        {"slot_id": "surface.modalities", "value": "+".join(modalities) if modalities else "none", "weight": 1.0},
        {"slot_id": "transport.methods_seen", "value": "+".join(methods) if methods else "none", "weight": 1.0},
        {"slot_id": "response.transition_delta", "value": transition if transition else "none", "weight": 1.0},
        {"slot_id": "failure.kind", "value": str(failure_signature.get("kind", "unknown")), "weight": 2.0 if failure else 1.0},
        {"slot_id": "failure.failed_gate", "value": str(failure_signature.get("failed_gate", "unknown")), "weight": 2.0 if failure else 1.0},
        {"slot_id": "failure.recovery_phase", "value": "failure_adjusted" if failure else "forward_baseline", "weight": 2.0 if failure else 1.0},
        {"slot_id": "probe.remaining_budget", "value": _bucket(int(failure_signature.get("remaining_probe_budget", 0) or 0)), "weight": 1.0},
    ]
    return {"schema_version": IR_SCHEMA_VERSION, "ir_id": f"ir-{_sha256(slots)[:16]}", "slots": slots[:MAX_IR_TOKENS], "source_modalities": modalities, "oracle_authority_included": False, "raw_retained": False}


def rule_ir_tokens(rule_ir: Mapping[str, Any]) -> dict[str, Any]:
    if str(rule_ir.get("schema_version")) != IR_SCHEMA_VERSION:
        raise ValueError("unexpected Rule IR schema")
    tokens = []
    for slot in rule_ir.get("slots", []):
        token = {"layer": "ir", "kind": "slot", "slot_id": _id(slot.get("slot_id"), "ir.slot_id"), "value": _id(slot.get("value"), "ir.value"), "weight": max(0.0, min(float(slot.get("weight", 1.0)), 2.0))}
        tokens.append(token)
    return {"schema_version": IR_SCHEMA_VERSION, "token_count": len(tokens), "tokens": tokens, "raw_retained": False, "oracle_authority_included": False, "ir_sha256": _sha256(rule_ir)}


def layered_compress(*, html_snapshot: str, javascript_snapshot: str, action_manifests: Sequence[Mapping[str, Any]], response_projection: Mapping[str, Any], failure_signature: Mapping[str, Any]) -> dict[str, Any]:
    source_layers = [tokenize_html_snapshot(html_snapshot), tokenize_javascript_snapshot(javascript_snapshot)]
    source_layers.extend(tokenize_action_manifest(manifest) for manifest in action_manifests)
    ir = summarize_rule_ir(source_layers, response_projection, failure_signature)
    ir_layer = rule_ir_tokens(ir)
    return {"schema_version": LAYERED_SCHEMA_VERSION, "layers": {"source_token_layers": source_layers, "rule_ir": ir, "ir_tokens": ir_layer}, "compression_contract": {"source_to_ir": True, "ir_to_ir_tokens": True, "raw_source_retained": False, "script_execution": False, "external_network": False, "oracle_authority_included": False, "memory_promotion_allowed": False}, "manifest_sha256": _sha256({"source": source_layers, "ir": ir, "ir_tokens": ir_layer})}


def validate_layered_compression(value: Mapping[str, Any]) -> dict[str, Any]:
    if str(value.get("schema_version")) != LAYERED_SCHEMA_VERSION:
        raise ValueError("layered compression schema is invalid")
    contract = dict(value.get("compression_contract") or {})
    for key in ("source_to_ir", "ir_to_ir_tokens", "raw_source_retained", "script_execution", "external_network", "oracle_authority_included", "memory_promotion_allowed"):
        if key not in contract:
            raise ValueError(f"layered compression contract misses {key}")
    if any(bool(contract[key]) for key in ("raw_source_retained", "script_execution", "external_network", "oracle_authority_included", "memory_promotion_allowed")):
        raise ValueError("layered compression safety contract failed")
    encoded = _canonical(value)
    if len(encoded) > 65536:
        raise ValueError("layered compression object is too large")
    def _find_forbidden(node: Any) -> str | None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if str(key).casefold() in _FORBIDDEN_KEYS:
                    return str(key)
                found = _find_forbidden(child)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = _find_forbidden(child)
                if found:
                    return found
        return None

    forbidden = _find_forbidden(value)
    if forbidden:
        raise ValueError(f"layered output contains forbidden field {forbidden}")
    declared = str(value.get("manifest_sha256", ""))
    if not _HASH_RE.fullmatch(declared):
        raise ValueError("layered manifest hash is invalid")
    body = dict(value)
    body.pop("manifest_sha256", None)
    if _sha256({"source": body.get("layers", {}).get("source_token_layers"), "ir": body.get("layers", {}).get("rule_ir"), "ir_tokens": body.get("layers", {}).get("ir_tokens")}) != declared:
        raise ValueError("layered manifest hash mismatch")
    return dict(value)


__all__ = [
    "IR_SCHEMA_VERSION",
    "LAYERED_SCHEMA_VERSION",
    "MAX_IR_TOKENS",
    "MAX_SOURCE_TOKENS",
    "SOURCE_SCHEMA_VERSION",
    "layered_compress",
    "rule_ir_tokens",
    "summarize_rule_ir",
    "tokenize_action_manifest",
    "tokenize_html_snapshot",
    "tokenize_javascript_snapshot",
    "validate_layered_compression",
]
