"""CTF-like frontend context projection for the PG-387 candidate dataset.

The input may be a bounded local fixture script or an already parsed abstract
surface.  The output deliberately drops source text, URLs, route literals,
wire values, response bodies and evaluator answers.  It is a context reader,
not an exploit generator: unsafe loader/persistence/dynamic-code surfaces are
represented as blocked/ASK states.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "pg387-ctf-frontend-context-v1"
SOURCE_LIMIT = 64 * 1024

_CASES: tuple[dict[str, str], ...] = (
    {"case_ref": "dom_text_reflection", "sink_kind": "dom_text", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "search_params_then_decode", "transport": "GET_query", "response_shape": "bounded_text_projection", "failure_shape": "none", "oracle_shape": "marker_reflection", "default_action": "select_probe_variant", "repair_action": "none"},
    {"case_ref": "dom_attribute_reflection", "sink_kind": "dom_attribute", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "attribute_escape_then_render", "transport": "GET_query", "response_shape": "bounded_attribute_projection", "failure_shape": "attribute_filter", "oracle_shape": "attribute_shape", "default_action": "ask_context", "repair_action": "none"},
    {"case_ref": "html_fragment_guard", "sink_kind": "dom_html_guarded", "loader_policy": "dynamic_blocked", "state_policy": "ephemeral", "normalization": "sanitize_then_parse", "transport": "POST_form", "response_shape": "bounded_fragment_projection", "failure_shape": "html_sink_blocked", "oracle_shape": "no_execution", "default_action": "abstain", "repair_action": "none"},
    {"case_ref": "url_attribute_guard", "sink_kind": "url_attribute", "loader_policy": "external_blocked", "state_policy": "ephemeral", "normalization": "scheme_allowlist", "transport": "GET_query", "response_shape": "redirect_shape", "failure_shape": "scheme_blocked", "oracle_shape": "redirect_shape", "default_action": "ask_context", "repair_action": "none"},
    {"case_ref": "script_loader_policy", "sink_kind": "script_loader", "loader_policy": "dynamic_blocked", "state_policy": "ephemeral", "normalization": "token_boundary", "transport": "POST_form", "response_shape": "policy_error", "failure_shape": "loader_blocked", "oracle_shape": "no_load", "default_action": "abstain", "repair_action": "none"},
    {"case_ref": "module_import_policy", "sink_kind": "module_loader", "loader_policy": "external_blocked", "state_policy": "ephemeral", "normalization": "specifier_allowlist", "transport": "GET_query", "response_shape": "policy_error", "failure_shape": "external_loader_blocked", "oracle_shape": "no_load", "default_action": "abstain", "repair_action": "none"},
    {"case_ref": "json_parser_boundary", "sink_kind": "json_value", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "json_parse_then_validate", "transport": "POST_json", "response_shape": "parser_error", "failure_shape": "json_parse_error", "oracle_shape": "parser_shape", "default_action": "ask_context", "repair_action": "syntax"},
    {"case_ref": "form_redirect_control", "sink_kind": "form_state", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "form_decode_then_route", "transport": "POST_form", "response_shape": "redirect_shape", "failure_shape": "redirect_blocked", "oracle_shape": "redirect_shape", "default_action": "select_probe_variant", "repair_action": "encoding"},
    {"case_ref": "fragment_router_guard", "sink_kind": "fragment_state", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "fragment_decode_once", "transport": "GET_fragment", "response_shape": "route_state_projection", "failure_shape": "fragment_mismatch", "oracle_shape": "state_shape", "default_action": "select_probe_variant", "repair_action": "encoding"},
    {"case_ref": "post_dom_update", "sink_kind": "dom_text", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "form_decode_then_render", "transport": "POST_form", "response_shape": "bounded_text_projection", "failure_shape": "field_missing", "oracle_shape": "marker_reflection", "default_action": "ask_context", "repair_action": "none"},
    {"case_ref": "client_normalizer_order", "sink_kind": "dom_text", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "double_decode_order_sensitive", "transport": "GET_query", "response_shape": "bounded_text_projection", "failure_shape": "raw_delimiter_blocked", "oracle_shape": "marker_reflection", "default_action": "repair", "repair_action": "encoding"},
    {"case_ref": "template_escape_boundary", "sink_kind": "template_text", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "template_escape", "transport": "GET_query", "response_shape": "bounded_template_projection", "failure_shape": "escaped_output", "oracle_shape": "no_execution", "default_action": "abstain", "repair_action": "none"},
    {"case_ref": "shadow_dom_text", "sink_kind": "shadow_text", "loader_policy": "static_only", "state_policy": "ephemeral", "normalization": "search_params_then_decode", "transport": "GET_query", "response_shape": "bounded_shadow_projection", "failure_shape": "shadow_boundary", "oracle_shape": "marker_reflection", "default_action": "ask_context", "repair_action": "none"},
    {"case_ref": "sandbox_frame", "sink_kind": "sandboxed_frame", "loader_policy": "dynamic_blocked", "state_policy": "ephemeral", "normalization": "attribute_escape_then_render", "transport": "POST_form", "response_shape": "sandbox_shape", "failure_shape": "execution_isolated", "oracle_shape": "no_execution", "default_action": "abstain", "repair_action": "none"},
    {"case_ref": "storage_policy_guard", "sink_kind": "dom_text", "loader_policy": "static_only", "state_policy": "persistent_blocked", "normalization": "search_params_then_decode", "transport": "POST_form", "response_shape": "policy_error", "failure_shape": "persistent_write_blocked", "oracle_shape": "no_write", "default_action": "abstain", "repair_action": "none"},
    {"case_ref": "dynamic_code_guard", "sink_kind": "dynamic_code", "loader_policy": "dynamic_blocked", "state_policy": "persistent_blocked", "normalization": "token_boundary", "transport": "POST_form", "response_shape": "policy_error", "failure_shape": "dynamic_code_blocked", "oracle_shape": "no_execution", "default_action": "abstain", "repair_action": "none"},
)

CTF_CASES = tuple(dict(item) for item in _CASES)

_SOURCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("dom_html_guarded", r"(?:innerHTML|insertAdjacentHTML)"),
    ("dom_attribute", r"(?:setAttribute|\.href\s*=|\.src\s*=)"),
    ("dom_text", r"(?:textContent|innerText|createTextNode)"),
    ("script_loader", r"(?:createElement\s*\(\s*['\"]script|appendChild\s*\()"),
    ("dynamic_code", r"(?:eval\s*\(|new\s+Function\s*\()"),
)

# These are deliberately abstract labels.  They describe the path from an
# input source through a client-side normalizer/guard to a sink, but never
# preserve an identifier, literal, URL, regex, or source-code substring.
_JS_SOURCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("location_search", r"(?:location\.(?:search|href)|urlsearchparams)"),
    ("location_hash", r"(?:location\.hash|hashchange)"),
    ("form_input", r"(?:formdata|\.elements\b|input\.value|textarea\.value)"),
    ("message_event", r"(?:messageevent|addEventListener\s*\(\s*['\"]message)"),
    ("client_storage", r"(?:localstorage|sessionstorage|indexeddb|document\.cookie)"),
)

_JS_PARSER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("url_search_params", r"urlsearchparams"),
    ("json_parse", r"json\.parse\s*\("),
    ("form_decode", r"formdata|formdecode|decodeform"),
    ("fragment_parse", r"location\.hash|hashchange"),
    ("template_parse", r"template|domparser"),
)

_JS_NORMALIZATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("percent_decode", r"decodeuricomponent|decodeuri|unescape"),
    ("url_search_parse", r"urlsearchparams"),
    ("json_parse", r"json\.parse\s*\("),
    ("html_escape", r"escapehtml|htmlspecialchars|textcontent|createTextNode"),
    ("url_scheme_check", r"(?:scheme|protocol|startsWith\s*\(\s*['\"]https?)"),
    ("casefold", r"toLowerCase|toUpperCase"),
    ("trim", r"\.trim\s*\("),
    ("replace_normalize", r"\.replace\s*\("),
)


def _first_label(source: str, patterns: tuple[tuple[str, str], ...], default: str) -> str:
    """Return a bounded abstract class, never the matched source text."""

    for label, pattern in patterns:
        if re.search(pattern, source, re.IGNORECASE):
            return label
    return default


def _ordered_labels(source: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    """Return an ordered, de-duplicated normalization chain."""

    hits: list[tuple[int, int, str]] = []
    for order, (label, pattern) in enumerate(patterns):
        match = re.search(pattern, source, re.IGNORECASE)
        if match:
            hits.append((match.start(), order, label))
    hits.sort()
    result: list[str] = []
    for _position, _order, label in hits:
        if label not in result:
            result.append(label)
    return result


def _semantic_js_projection(source: str, *, sink: str, external: bool, dynamic_loader: bool, persistent: bool, dynamic_code: bool) -> dict[str, Any]:
    """Build the append-only semantic JS overlay used by PG-387.

    This is intentionally not a JavaScript lexer/AST serializer.  It is a
    stable, low-leakage source→transform→guard→sink representation so the
    model can reason about client context while the evaluator retains source
    and concrete test values off-context.
    """

    lowered = source.casefold()
    source_kind = _first_label(source, _JS_SOURCE_PATTERNS, "unknown_source")
    parser_kind = _first_label(source, _JS_PARSER_PATTERNS, "none_observed")
    chain = _ordered_labels(source, _JS_NORMALIZATION_PATTERNS)
    if not chain:
        chain = ["identity_observed"]

    if re.search(r"(?:allowlist|allowed|startsWith\s*\(|endsWith\s*\(|includes\s*\()", source, re.IGNORECASE):
        filter_shape = "allowlist_or_membership"
    elif re.search(r"(?:\.test\s*\(|\.replace\s*\(|blocklist|deny|forbidden)", source, re.IGNORECASE):
        filter_shape = "blocklist_or_regex"
    elif re.search(r"(?:sanitize|dompurify|escape)", source, re.IGNORECASE):
        filter_shape = "sanitizer_or_escape"
    else:
        filter_shape = "none_observed"

    guard_bits: list[str] = []
    if re.search(r"\bif\s*\(|\bswitch\s*\(|\?", source, re.IGNORECASE):
        guard_bits.append("conditional")
    if re.search(r"try\s*\{|catch\s*\(", source, re.IGNORECASE):
        guard_bits.append("exception_boundary")
    if re.search(r"preventDefault\s*\(|return\s+false", source, re.IGNORECASE):
        guard_bits.append("event_block")
    guard_shape = "+".join(guard_bits) if guard_bits else "none_observed"

    control_bits: list[str] = []
    if re.search(r"\bif\s*\(|\bswitch\s*\(|\?", source, re.IGNORECASE):
        control_bits.append("branch")
    if re.search(r"\b(?:for|while|do)\s*\(", source, re.IGNORECASE):
        control_bits.append("loop")
    if re.search(r"try\s*\{|catch\s*\(", source, re.IGNORECASE):
        control_bits.append("exception")
    control_flow = "+".join(control_bits) if control_bits else "straight_line"

    if re.search(r"addEventListener\s*\(\s*['\"](?:input|submit|click)", source, re.IGNORECASE) or re.search(r"onsubmit|oninput|onclick", source, re.IGNORECASE):
        event_shape = "input_or_submit_handler"
    elif re.search(r"DOMContentLoaded|window\.onload", source, re.IGNORECASE):
        event_shape = "document_ready_handler"
    elif "message_event" == source_kind:
        event_shape = "message_handler"
    else:
        event_shape = "none_observed"

    ast_bits: list[str] = []
    if re.search(r"\b(?:function|=>)\b|=>", source, re.IGNORECASE):
        ast_bits.append("function_or_arrow")
    if re.search(r"\b(?:if|switch|for|while|try)\b", source, re.IGNORECASE):
        ast_bits.append("control_node")
    if re.search(r"\.[A-Za-z_$][\w$]*\s*\(", source):
        ast_bits.append("call_chain")
    if re.search(r"\b(?:const|let|var)\b", source, re.IGNORECASE):
        ast_bits.append("binding_node")
    ast_shape = "+".join(ast_bits) if ast_bits else "expression_only"

    if sink in {"dom_html_guarded", "dom_attribute", "dom_text", "script_loader", "dynamic_code"}:
        sink_context = {
            "dom_html_guarded": "html_fragment_sink",
            "dom_attribute": "attribute_sink",
            "dom_text": "text_sink",
            "script_loader": "loader_sink",
            "dynamic_code": "code_sink",
        }[sink]
    else:
        sink_context = "sink_not_observed"
    if source_kind == "unknown_source":
        source_to_sink = "unknown_source_path"
    elif sink_context == "sink_not_observed":
        source_to_sink = "source_without_sink"
    else:
        source_to_sink = f"{source_kind}_to_{sink_context}"

    semantic = {
        "source_kind": source_kind,
        "parser_kind": parser_kind,
        "normalization_chain": chain,
        "filter_shape": filter_shape,
        "guard_shape": guard_shape,
        "control_flow_shape": control_flow,
        "event_shape": event_shape,
        "ast_shape": ast_shape,
        "source_to_sink_shape": source_to_sink,
        "sink_context": sink_context,
        "external_or_dynamic_loader": bool(external or dynamic_loader),
        "persistent_state": bool(persistent),
        "dynamic_code": bool(dynamic_code),
    }
    tokens = [
        f"js_source={source_kind}",
        f"js_parser={parser_kind}",
        *(f"js_normalization_step={step}" for step in chain),
        f"js_filter_shape={filter_shape}",
        f"js_guard_shape={guard_shape}",
        f"js_control_flow={control_flow}",
        f"js_event_shape={event_shape}",
        f"js_ast_shape={ast_shape}",
        f"js_source_to_sink={source_to_sink}",
        f"js_sink_context={sink_context}",
        f"js_persistent_state={'blocked' if persistent else 'absent'}",
    ]
    semantic["tokens"] = tokens
    return semantic


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tokens(case: Mapping[str, str]) -> list[str]:
    ordered = (
        ("case_surface", case["case_ref"]),
        ("js_sink", case["sink_kind"]),
        ("loader_policy", case["loader_policy"]),
        ("state_policy", case["state_policy"]),
        ("normalization", case["normalization"]),
        ("transport", case["transport"]),
        ("response_shape", case["response_shape"]),
        ("failure_shape", case["failure_shape"]),
        ("oracle_shape", case["oracle_shape"]),
    )
    # Case fixtures carry only abstract labels.  Derive the same semantic
    # overlay used by source projection so synthetic CTF rows and local JS
    # source rows share one token vocabulary.
    semantic = _case_semantic_projection(case)
    semantic_tokens = [
        f"js_source={semantic['source_kind']}",
        f"js_parser={semantic['parser_kind']}",
        *(f"js_normalization_step={step}" for step in semantic["normalization_chain"]),
        f"js_filter_shape={semantic['filter_shape']}",
        f"js_guard_shape={semantic['guard_shape']}",
        f"js_control_flow={semantic['control_flow_shape']}",
        f"js_event_shape={semantic['event_shape']}",
        "js_ast_shape=case_fixture",
        f"js_source_to_sink={semantic['source_to_sink_shape']}",
        f"js_sink_context={semantic['sink_context']}",
        f"js_persistent_state={'blocked' if case.get('state_policy') != 'ephemeral' else 'absent'}",
    ]
    return ["[BOS]", *(f"{key}={value}" for key, value in ordered), *semantic_tokens, "[CTX_END]"]


def _case_semantic_projection(case: Mapping[str, Any]) -> dict[str, Any]:
    """Map a synthetic case to the same abstract JS overlay as source code."""

    transport = str(case.get("transport", ""))
    source_kind = "location_hash" if "fragment" in transport.casefold() else "form_input" if transport.startswith("POST") else "location_search"
    parser_kind = "json_parse" if "json" in transport.casefold() else "fragment_parse" if "fragment" in transport.casefold() else "url_search_params" if "query" in transport.casefold() else "form_decode"
    normalization = str(case.get("normalization", "none_observed"))
    failure = str(case.get("failure_shape", ""))
    filter_shape = "sanitizer_or_escape" if any(word in normalization for word in ("sanitize", "escape", "template")) else "blocklist_or_regex" if any(word in failure for word in ("blocked", "filter", "mismatch")) else "none_observed"
    guard_shape = "policy_gate" if case.get("loader_policy") != "static_only" or case.get("state_policy") != "ephemeral" else "conditional" if case.get("default_action") in {"ask_context", "repair", "abstain"} else "none_observed"
    sink_context = {
        "dom_html_guarded": "html_fragment_sink",
        "dom_attribute": "attribute_sink",
        "dom_text": "text_sink",
        "script_loader": "loader_sink",
        "module_loader": "loader_sink",
        "dynamic_code": "code_sink",
    }.get(str(case.get("sink_kind", "")), "sink_not_observed")
    return {
        "source_kind": source_kind,
        "parser_kind": parser_kind,
        "normalization_chain": [normalization],
        "filter_shape": filter_shape,
        "guard_shape": guard_shape,
        "control_flow_shape": "branch" if guard_shape != "none_observed" else "straight_line",
        "event_shape": "input_or_submit_handler" if transport.startswith("POST") else "route_or_load_handler",
        "ast_shape": "case_fixture",
        "source_to_sink_shape": f"{source_kind}_to_{sink_context}",
        "sink_context": sink_context,
    }


def project_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict abstract projection for one CTF-like surface."""

    required = {item["case_ref"] for item in _CASES}
    case_ref = str(case.get("case_ref", ""))
    if case_ref not in required:
        raise ValueError("unknown_pg387_case")
    canonical = next(item for item in _CASES if item["case_ref"] == case_ref)
    context = dict(canonical)
    safe = context["loader_policy"] == "static_only" and context["state_policy"] == "ephemeral" and context["sink_kind"] not in {"dom_html_guarded", "dynamic_code"}
    if context["default_action"] == "repair":
        next_action = "repair"
        probe_variant = "one_variable_context_bound"
    elif safe and context["default_action"] != "abstain":
        next_action = context["default_action"]
        probe_variant = "one_variable_context_bound"
    else:
        next_action = "ask"
        probe_variant = "none"
    target = [
        "[TARGET_BEGIN]",
        f"next_action={next_action}",
        f"repair_action={context['repair_action']}",
        f"probe_variant_ref={probe_variant}",
        f"safe_to_send={int(safe)}",
        f"ask_reason={'context_or_policy_missing' if not safe else 'none'}",
        "[TARGET_END]",
    ]
    semantic = _case_semantic_projection(context)
    projection = {
        "schema_version": SCHEMA_VERSION,
        "context_tokens": _tokens(context),
        "target_tokens": target,
        "javascript_surface": {
            "sink_kind": context["sink_kind"],
            "loader_policy": context["loader_policy"],
            "state_policy": context["state_policy"],
            "normalization": context["normalization"],
            "source_text_stored": False,
            "external_network": False,
            "raw_response_stored": False,
        },
        "javascript_context": {
            **semantic,
            "ast_shape": "case_fixture",
            "semantic_overlay_only": True,
            "source_text_stored": False,
        },
        "training_eligible": False,
        "typed_evaluator_observed": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    projection["projection_sha256"] = _sha(projection)
    return projection


def project_js_source(source: str, *, local_fixture: bool = True) -> dict[str, Any]:
    """Read a bounded local script and emit only abstract JS-context tokens.

    This intentionally does not return the source or any matched substring.
    External references, dynamic code and persistence APIs make the result
    ASK/blocked; they never become a model-send permission.
    """

    if not isinstance(source, str) or len(source.encode("utf-8")) > SOURCE_LIMIT:
        raise ValueError("source_limit_exceeded")
    lowered = source.casefold()
    sink = "none"
    for candidate, pattern in _SOURCE_PATTERNS:
        if re.search(pattern, source, re.IGNORECASE):
            sink = candidate
            break
    external = bool(re.search(r"(?:https?://|//[^/]|websocket\s*\()", lowered))
    dynamic_loader = bool(re.search(r"(?:import\s*\(|fetch\s*\(|xmlhttprequest|createelement\s*\(\s*['\"]script)", lowered))
    persistent = bool(re.search(r"(?:localstorage|sessionstorage|indexeddb|document\.cookie)", lowered))
    dynamic_code = bool(re.search(r"(?:eval\s*\(|new\s+function\s*\()", lowered))
    normalizer = "decode_or_normalize_observed" if re.search(r"(?:decodeuricomponent|urlsearchparams|normalize|replace\s*\()", lowered) else "none_observed"
    if external or dynamic_loader:
        loader_policy = "external_or_dynamic_blocked"
    else:
        loader_policy = "static_only"
    state_policy = "persistent_blocked" if persistent else "ephemeral"
    safe = bool(local_fixture) and not (external or dynamic_loader or persistent or dynamic_code) and sink in {"dom_text", "dom_attribute"}
    semantic = _semantic_js_projection(
        source,
        sink=sink,
        external=external,
        dynamic_loader=dynamic_loader,
        persistent=persistent,
        dynamic_code=dynamic_code,
    )
    projection = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_text_stored": False,
        "local_fixture": bool(local_fixture),
        "javascript_surface": {
            "sink_kind": sink,
            "loader_policy": loader_policy,
            "state_policy": state_policy,
            "dynamic_code": "present_blocked" if dynamic_code else "absent",
            "normalization": normalizer,
            "external_network": external,
            "persistent_write": persistent,
        },
        "javascript_context": {
            key: value for key, value in semantic.items() if key != "tokens"
        },
        "js_semantic_tokens": semantic["tokens"],
        "context_tokens": [
            "[BOS]",
            f"js_sink={sink}",
            f"loader_policy={loader_policy}",
            f"state_policy={state_policy}",
            f"normalization={normalizer}",
            f"external_network={'present' if external else 'absent'}",
            f"dynamic_code={'blocked' if dynamic_code else 'absent'}",
            *semantic["tokens"],
            "[CTX_END]",
        ],
        "next_action": "select_probe_variant" if safe else "ask",
        "safe_to_send": safe,
        "ask_reason": "none" if safe else "js_context_or_policy_missing",
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    projection["projection_sha256"] = _sha(projection)
    return projection


tokenize_js_source = project_js_source


__all__ = ["CTF_CASES", "SCHEMA_VERSION", "SOURCE_LIMIT", "project_case", "project_js_source", "tokenize_js_source"]
