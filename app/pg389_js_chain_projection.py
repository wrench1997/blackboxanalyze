"""Abstract JavaScript decode/filter-chain projection for PG-389.

The module describes *where* decoding, filtering, guards and sinks occur.  It
never returns source text, literals, URLs, wire values or evaluator answers.
Concrete canaries, if ever used, remain a separate local evaluator concern.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg389-js-decode-filter-chain-v1"
SOURCE_LIMIT = 64 * 1024
CHAIN_VARIANTS = ("chain_order_a", "chain_order_b")


def _case(
    case_ref: str,
    *,
    source_kind: str,
    transport: str,
    decoder_chain: tuple[str, ...],
    filter_stage: str,
    guard_precedence: str,
    sink_context: str,
    state_policy: str,
    failure_signature: str,
    oracle_shape: str,
    next_action: str,
    repair_action: str,
    probe_variant: str,
    ask_reason: str,
    safe_fixture: bool,
    observation_sequence: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "case_ref": case_ref,
        "source_kind": source_kind,
        "transport": transport,
        "decoder_chain": list(decoder_chain),
        "filter_stage": filter_stage,
        "guard_precedence": guard_precedence,
        "sink_context": sink_context,
        "state_policy": state_policy,
        "failure_signature": failure_signature,
        "oracle_shape": oracle_shape,
        "next_action": next_action,
        "repair_action": repair_action,
        "probe_variant_ref": probe_variant,
        "ask_reason": ask_reason,
        "safe_fixture": bool(safe_fixture),
        "observation_sequence": list(observation_sequence),
    }


CHAIN_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "query_decode_then_filter",
        source_kind="location_search",
        transport="GET_query",
        decoder_chain=("query_parse", "percent_decode"),
        filter_stage="after_decode",
        guard_precedence="filter_before_sink",
        sink_context="text_sink",
        state_policy="ephemeral",
        failure_signature="filtered_after_decode",
        oracle_shape="bounded_text_shape",
        next_action="repair",
        repair_action="encoding",
        probe_variant="fixture_shape_encoded",
        ask_reason="none",
        safe_fixture=True,
        observation_sequence=("input_seen", "decoded_once", "filter_rejected", "sink_not_reached"),
    ),
    _case(
        "query_filter_then_decode",
        source_kind="location_search",
        transport="GET_query",
        decoder_chain=("query_parse", "percent_decode"),
        filter_stage="before_decode",
        guard_precedence="filter_before_sink",
        sink_context="text_sink",
        state_policy="ephemeral",
        failure_signature="raw_shape_blocked",
        oracle_shape="bounded_text_shape",
        next_action="select_probe_variant",
        repair_action="encoding",
        probe_variant="fixture_shape_one_variable",
        ask_reason="none",
        safe_fixture=True,
        observation_sequence=("input_seen", "filter_rejected", "decoder_not_reached", "sink_not_reached"),
    ),
    _case(
        "double_decode_order",
        source_kind="location_search",
        transport="GET_query",
        decoder_chain=("query_parse", "percent_decode", "percent_decode"),
        filter_stage="between_decode_steps",
        guard_precedence="guard_after_normalize",
        sink_context="text_sink",
        state_policy="ephemeral",
        failure_signature="second_decode_changes_shape",
        oracle_shape="bounded_marker_shape",
        next_action="repair",
        repair_action="encoding",
        probe_variant="fixture_shape_layered",
        ask_reason="none",
        safe_fixture=True,
        observation_sequence=("input_seen", "decoded_once", "guard_checked", "decoded_twice", "sink_shape_observed"),
    ),
    _case(
        "json_parse_before_guard",
        source_kind="form_input",
        transport="POST_json",
        decoder_chain=("json_parse", "field_extract"),
        filter_stage="after_parse",
        guard_precedence="schema_before_sink",
        sink_context="structured_value_sink",
        state_policy="ephemeral",
        failure_signature="schema_rejected",
        oracle_shape="parser_error_shape",
        next_action="ask_context",
        repair_action="syntax",
        probe_variant="none",
        ask_reason="parser_observation_missing",
        safe_fixture=True,
        observation_sequence=("input_seen", "parsed", "schema_rejected", "sink_not_reached"),
    ),
    _case(
        "form_decode_then_route",
        source_kind="form_input",
        transport="POST_form",
        decoder_chain=("form_decode", "trim"),
        filter_stage="before_route",
        guard_precedence="allowlist_before_route",
        sink_context="route_state_sink",
        state_policy="ephemeral",
        failure_signature="route_guard_rejected",
        oracle_shape="bounded_redirect_shape",
        next_action="ask_context",
        repair_action="encoding",
        probe_variant="none",
        ask_reason="route_guard_context_missing",
        safe_fixture=True,
        observation_sequence=("input_seen", "decoded_once", "guard_rejected", "route_not_taken"),
    ),
    _case(
        "fragment_decode_once",
        source_kind="location_hash",
        transport="GET_fragment",
        decoder_chain=("fragment_parse", "percent_decode"),
        filter_stage="after_decode",
        guard_precedence="sink_guard_before_render",
        sink_context="attribute_sink",
        state_policy="ephemeral",
        failure_signature="attribute_guard_rejected",
        oracle_shape="bounded_attribute_shape",
        next_action="ask_context",
        repair_action="none",
        probe_variant="none",
        ask_reason="sink_context_missing",
        safe_fixture=False,
        observation_sequence=("input_seen", "decoded_once", "guard_rejected", "sink_not_reached"),
    ),
    _case(
        "trim_then_casefold_allowlist",
        source_kind="form_input",
        transport="POST_form",
        decoder_chain=("form_decode", "trim", "casefold"),
        filter_stage="after_normalize",
        guard_precedence="allowlist_before_sink",
        sink_context="text_sink",
        state_policy="ephemeral",
        failure_signature="allowlist_miss",
        oracle_shape="bounded_text_shape",
        next_action="select_probe_variant",
        repair_action="normalization",
        probe_variant="fixture_shape_casefold",
        ask_reason="none",
        safe_fixture=True,
        observation_sequence=("input_seen", "trimmed", "casefolded", "guard_rejected", "sink_not_reached"),
    ),
    _case(
        "escape_then_text_sink",
        source_kind="location_search",
        transport="GET_query",
        decoder_chain=("query_parse", "percent_decode", "html_escape"),
        filter_stage="escape_at_sink",
        guard_precedence="escape_before_sink",
        sink_context="text_sink",
        state_policy="ephemeral",
        failure_signature="escaped_output",
        oracle_shape="bounded_text_shape",
        next_action="abstain",
        repair_action="none",
        probe_variant="none",
        ask_reason="sink_is_non_executable",
        safe_fixture=False,
        observation_sequence=("input_seen", "decoded_once", "escaped", "text_shape_observed"),
    ),
    _case(
        "url_scheme_guard",
        source_kind="form_input",
        transport="POST_form",
        decoder_chain=("form_decode", "trim", "scheme_parse"),
        filter_stage="before_attribute_sink",
        guard_precedence="scheme_allowlist_before_sink",
        sink_context="url_attribute_sink",
        state_policy="ephemeral",
        failure_signature="scheme_rejected",
        oracle_shape="bounded_redirect_shape",
        next_action="abstain",
        repair_action="none",
        probe_variant="none",
        ask_reason="external_scheme_blocked",
        safe_fixture=False,
        observation_sequence=("input_seen", "decoded_once", "scheme_checked", "guard_rejected", "sink_not_reached"),
    ),
    _case(
        "parser_error_short_circuit",
        source_kind="form_input",
        transport="POST_json",
        decoder_chain=("json_parse",),
        filter_stage="parser_boundary",
        guard_precedence="parser_before_filter",
        sink_context="structured_value_sink",
        state_policy="ephemeral",
        failure_signature="parser_error",
        oracle_shape="parser_error_shape",
        next_action="repair",
        repair_action="syntax",
        probe_variant="fixture_shape_well_formed",
        ask_reason="none",
        safe_fixture=True,
        observation_sequence=("input_seen", "parser_error", "filter_not_reached", "sink_not_reached"),
    ),
    _case(
        "persistent_state_guard",
        source_kind="form_input",
        transport="POST_form",
        decoder_chain=("form_decode", "trim"),
        filter_stage="before_state_write",
        guard_precedence="persistence_block_before_sink",
        sink_context="persistent_state_sink",
        state_policy="persistent_blocked",
        failure_signature="persistent_write_blocked",
        oracle_shape="no_write_shape",
        next_action="abstain",
        repair_action="none",
        probe_variant="none",
        ask_reason="persistent_state_not_authorized",
        safe_fixture=False,
        observation_sequence=("input_seen", "decoded_once", "persistence_guard", "write_not_attempted"),
    ),
    _case(
        "dynamic_code_guard",
        source_kind="form_input",
        transport="POST_form",
        decoder_chain=("form_decode", "dynamic_code_boundary"),
        filter_stage="before_code_sink",
        guard_precedence="dynamic_code_block_before_sink",
        sink_context="dynamic_code_sink",
        state_policy="persistent_blocked",
        failure_signature="dynamic_code_blocked",
        oracle_shape="no_execution_shape",
        next_action="abstain",
        repair_action="none",
        probe_variant="none",
        ask_reason="dynamic_code_not_authorized",
        safe_fixture=False,
        observation_sequence=("input_seen", "decoded_once", "code_guard_rejected", "sink_not_reached"),
    ),
)

_CASE_BY_REF = {item["case_ref"]: item for item in CHAIN_CASES}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _semantic_tokens(case: Mapping[str, Any], variant: str) -> list[str]:
    tokens = [
        "[BOS]",
        f"chain_case={case['case_ref']}",
        f"chain_variant={variant}",
        f"source_kind={case['source_kind']}",
        f"transport={case['transport']}",
        *(f"decoder_step_{index}={step}" for index, step in enumerate(case["decoder_chain"], start=1)),
        f"filter_stage={case['filter_stage']}",
        f"guard_precedence={case['guard_precedence']}",
        f"sink_context={case['sink_context']}",
        f"state_policy={case['state_policy']}",
        f"failure_signature={case['failure_signature']}",
        f"oracle_shape={case['oracle_shape']}",
        *(f"observation_{index}={step}" for index, step in enumerate(case["observation_sequence"], start=1)),
        "[CTX_END]",
    ]
    return tokens


def project_chain_case(case: Mapping[str, Any], *, variant: str = "chain_order_a") -> dict[str, Any]:
    """Return the bounded model projection for one known chain case."""

    case_ref = str(case.get("case_ref", ""))
    if case_ref not in _CASE_BY_REF:
        raise ValueError("unknown_pg389_case")
    if variant not in CHAIN_VARIANTS:
        raise ValueError("unknown_pg389_variant")
    canonical = dict(_CASE_BY_REF[case_ref])
    safe = bool(canonical["safe_fixture"] and canonical["state_policy"] == "ephemeral")
    target = [
        "[TARGET_BEGIN]",
        f"next_action={canonical['next_action']}",
        f"repair_action={canonical['repair_action']}",
        f"probe_variant_ref={canonical['probe_variant_ref'] if safe else 'none'}",
        f"safe_to_send={int(safe)}",
        f"ask_reason={canonical['ask_reason'] if not safe or canonical['ask_reason'] != 'none' else 'none'}",
        f"observation_goal={canonical['oracle_shape']}",
        "[TARGET_END]",
    ]
    projection = {
        "schema_version": SCHEMA_VERSION,
        "case_ref": case_ref,
        "chain_variant": variant,
        "context_tokens": _semantic_tokens(canonical, variant),
        "target_tokens": target,
        "decode_filter_context": {
            "source_kind": canonical["source_kind"],
            "transport": canonical["transport"],
            "decoder_chain": list(canonical["decoder_chain"]),
            "filter_stage": canonical["filter_stage"],
            "guard_precedence": canonical["guard_precedence"],
            "sink_context": canonical["sink_context"],
            "observation_sequence": list(canonical["observation_sequence"]),
            "source_text_stored": False,
            "raw_value_stored": False,
        },
        "javascript_surface": {
            "source_kind": canonical["source_kind"],
            "decoder_chain_length": len(canonical["decoder_chain"]),
            "sink_context": canonical["sink_context"],
            "state_policy": canonical["state_policy"],
            "external_network": False,
            "persistent_write": canonical["state_policy"] != "ephemeral",
        },
        "source_text_stored": False,
        "typed_evaluator_observed": False,
        "safe_to_send": safe,
        "training_eligible": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    projection["projection_sha256"] = _sha(projection)
    return projection


def _ordered_steps(source: str) -> list[tuple[int, str]]:
    patterns = (
        ("query_parse", r"urlsearchparams|location\.search"),
        ("fragment_parse", r"location\.hash|hashchange"),
        ("form_decode", r"formdata|formdecode|decodeform"),
        ("json_parse", r"json\.parse\s*\("),
        ("percent_decode", r"decodeuricomponent|decodeuri|unescape"),
        ("trim", r"\.trim\s*\("),
        ("casefold", r"tolowercase|touppercase"),
        ("html_escape", r"escapehtml|htmlspecialchars|textcontent|createtextnode"),
        ("scheme_parse", r"protocol|scheme"),
        ("dynamic_code_boundary", r"eval\s*\(|new\s+function"),
        ("field_extract", r"\.get\s*\(|\.value\b"),
    )
    hits: list[tuple[int, str]] = []
    lowered = source.casefold()
    for label, pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            hits.append((match.start(), label))
    return sorted(hits)


def project_js_chain_source(source: str, *, local_fixture: bool = True, variant: str = "chain_order_a") -> dict[str, Any]:
    """Project bounded local JS into the same chain vocabulary.

    This is a lexical semantic projection, not an AST or exploit generator.
    The returned object contains only abstract labels and a source hash.
    """

    if not isinstance(source, str) or len(source.encode("utf-8")) > SOURCE_LIMIT:
        raise ValueError("source_limit_exceeded")
    if variant not in CHAIN_VARIANTS:
        raise ValueError("unknown_pg389_variant")
    lowered = source.casefold()
    ordered = _ordered_steps(source)
    steps = [label for _position, label in ordered]
    if not steps:
        steps = ["identity_observed"]
    if "fragment_parse" in steps:
        source_kind = "location_hash"
        transport = "GET_fragment"
    elif any(label in steps for label in ("form_decode",)):
        source_kind = "form_input"
        transport = "POST_form"
    elif "json_parse" in steps:
        source_kind = "form_input"
        transport = "POST_json"
    else:
        source_kind = "location_search" if "query_parse" in steps else "unknown_source"
        transport = "GET_query" if source_kind == "location_search" else "unknown_transport"

    if re.search(r"(?:innerhtml|insertadjacenthtml|\.href\s*=|\.src\s*=)", lowered):
        sink_context = "html_or_attribute_sink"
    elif re.search(r"(?:textcontent|innertext|createtextnode)", lowered):
        sink_context = "text_sink"
    elif "dynamic_code_boundary" in steps:
        sink_context = "dynamic_code_sink"
    else:
        sink_context = "sink_not_observed"
    if re.search(r"(?:allowlist|allowed|startswith|endswith|includes|\.test\s*\(|blocklist|deny|forbidden)", lowered):
        filter_stage = "guard_or_filter_observed"
    else:
        filter_stage = "none_observed"
    external = bool(re.search(r"(?:https?://|websocket\s*\(|fetch\s*\(|xmlhttprequest)", lowered))
    persistent = bool(re.search(r"(?:localstorage|sessionstorage|indexeddb|document\.cookie)", lowered))
    dynamic = "dynamic_code_boundary" in steps
    safe = bool(local_fixture and not external and not persistent and not dynamic and sink_context == "text_sink")
    if external or persistent or dynamic:
        next_action, ask_reason = "ask", "js_policy_or_context_missing"
    elif safe:
        next_action, ask_reason = "select_probe_variant", "none"
    else:
        next_action, ask_reason = "ask", "sink_context_missing"
    target = [
        "[TARGET_BEGIN]",
        f"next_action={next_action}",
        "repair_action=encoding" if "percent_decode" in steps else "repair_action=none",
        f"probe_variant_ref={'fixture_shape_one_variable' if safe else 'none'}",
        f"safe_to_send={int(safe)}",
        f"ask_reason={ask_reason}",
        f"observation_goal={'bounded_text_shape' if safe else 'context_projection'}",
        "[TARGET_END]",
    ]
    projection = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_text_stored": False,
        "context_tokens": [
            "[BOS]",
            f"chain_variant={variant}",
            f"source_kind={source_kind}",
            f"transport={transport}",
            *(f"decoder_step_{index}={step}" for index, step in enumerate(steps, start=1)),
            f"filter_stage={filter_stage}",
            f"sink_context={sink_context}",
            f"external_network={'present' if external else 'absent'}",
            f"state_policy={'persistent_blocked' if persistent else 'ephemeral'}",
            f"dynamic_code={'blocked' if dynamic else 'absent'}",
            "[CTX_END]",
        ],
        "target_tokens": target,
        "decode_filter_context": {
            "source_kind": source_kind,
            "transport": transport,
            "decoder_chain": steps,
            "filter_stage": filter_stage,
            "guard_precedence": "guard_observed" if filter_stage != "none_observed" else "not_observed",
            "sink_context": sink_context,
            "source_text_stored": False,
            "raw_value_stored": False,
        },
        "javascript_surface": {
            "source_kind": source_kind,
            "decoder_chain_length": len(steps),
            "sink_context": sink_context,
            "external_network": external,
            "persistent_write": persistent,
            "dynamic_code": dynamic,
        },
        "safe_to_send": safe,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    projection["projection_sha256"] = _sha(projection)
    return projection


__all__ = [
    "CHAIN_CASES",
    "CHAIN_VARIANTS",
    "SCHEMA_VERSION",
    "SOURCE_LIMIT",
    "project_chain_case",
    "project_js_chain_source",
]
