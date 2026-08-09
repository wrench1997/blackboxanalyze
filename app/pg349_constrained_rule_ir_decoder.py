"""PG-349 constrained Rule-IR decoding for abstract payload shapes.

The decoder is deliberately a *safety and evidence layer* around a causal
model.  A model may propose a bounded Rule-IR slot sequence, but this module
forces ASK/repair/abstain when observations are missing or when the evaluator
contract is incomplete.  It never accepts, stores, or returns a literal
payload, URL, response body, route, family label, or evaluator answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "pg349-constrained-rule-ir-decoder-v1"

TARGET_KEYS = (
    "question",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "probe_variant_ref",
    "payload_shape_ref",
    "oracle_ref",
    "safe_to_send",
)

FORBIDDEN_MARKERS = frozenset(
    {
        "payload",
        "raw_payload",
        "response",
        "response_body",
        "route",
        "url",
        "family",
        "oracle_answer",
        "evaluator_answer",
        "source_code",
        "credentials",
        "cookie",
        "sql",
        "xss",
        "xxe",
    }
)

_ALLOWED = {
    "question": frozenset({"none", "ask_typed", "ask_failure", "ask_surface", "ask_oracle"}),
    "next_action": frozenset(
        {
            "ask_typed",
            "ask_failure",
            "assemble_rule_ir",
            "select_probe_variant",
            "send_probe",
            "repair",
            "replay",
            "abstain",
        }
    ),
    "repair_action": frozenset({"none", "observe", "encoding", "channel", "field", "reset", "one_variable"}),
    "transport_ref": frozenset({"none", "get_query", "get_path", "get_fragment", "post_form", "post_json", "header", "unknown"}),
    "field_role_ref": frozenset({"none", "query_term", "query_text", "form_field", "display_text", "attribute_value", "path_segment", "fragment_identifier", "json_value", "header_value", "unknown"}),
    "encoding_ref": frozenset({"none", "identity", "url_percent", "form_urlencoded", "html_entity", "javascript_unicode", "json_escape", "xml_entity", "double_layer_order_sensitive", "unknown"}),
    "probe_variant_ref": frozenset({"none", "baseline_marker", "reference", "reference_shape", "source_attested_candidate", "matched_negative", "negative_control", "one_variable_repair", "runtime_canary", "fresh_replay", "unsupported_abstain"}),
    "payload_shape_ref": frozenset(
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
    ),
    "oracle_ref": frozenset({"none", "reflection", "response_shape", "parser_shape", "dom_shape", "typed_state_delta", "typed_effect", "negative_no_effect", "unknown"}),
    "safe_to_send": frozenset({"0", "1"}),
}

_REQUIRED_EVIDENCE = (
    "typed_available",
    "fresh_reset",
    "replay_ready",
    "evidence_present",
    "reference_present",
    "candidate_present",
)


def _pairs(tokens: Sequence[str]) -> dict[str, str]:
    """Read abstract key/value tokens, including belief-prefixed aliases."""

    values: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
        for prefix in ("belief_", "belief_and_replay_field_", "failure_", "request_", "response_"):
            if key.startswith(prefix):
                values.setdefault(key[len(prefix) :], value)
    return values


def _missing_fields(values: Mapping[str, str]) -> list[str]:
    missing: list[str] = []
    for key in _REQUIRED_EVIDENCE:
        value = values.get(key)
        if value not in {"present", "1", "true", "clean"}:
            missing.append(key)
    for key, value in values.items():
        if key.endswith("_presence") and value in {"unknown", "not_observed"}:
            missing.append(key)
    return sorted(set(missing))


def _failure(values: Mapping[str, str]) -> bool:
    failure = values.get("failure_class") or values.get("failure_failure_class") or "none"
    return failure not in {"none", "absent", "not_applicable", "clean"}


def _forbidden(values: Mapping[str, Any]) -> list[str]:
    bad: list[str] = []
    for key, value in values.items():
        key_text = str(key).lower()
        value_text = str(value).lower()
        if key_text in FORBIDDEN_MARKERS or any(marker in key_text for marker in ("raw_", "response_body", "oracle_answer", "evaluator_")):
            bad.append(str(key))
        if any(marker in value_text for marker in ("<script", "javascript:", "document.cookie", "alert(", "http://", "https://")):
            bad.append(str(key))
    return sorted(set(bad))


def _canonical_proposal(proposal: Mapping[str, Any] | None) -> dict[str, str]:
    proposal = proposal or {}
    values = {key: str(proposal.get(key, "unknown")) for key in TARGET_KEYS}
    for key, allowed in _ALLOWED.items():
        if values[key] not in allowed:
            values[key] = "unknown" if key != "safe_to_send" else "0"
    return values


def constrain_rule_ir(context_tokens: Sequence[str], proposal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply fail-closed evidence constraints to a bounded Rule-IR proposal."""

    context = _pairs(context_tokens)
    proposal_mapping = proposal or {}
    proposed = _canonical_proposal(proposal_mapping)
    forbidden = _forbidden(proposal_mapping)
    missing = _missing_fields(context)
    has_failure = _failure(context)
    reasons: list[str] = []
    output = dict(proposed)
    if output["oracle_ref"] == "unknown" and context.get("typed_available") in {"present", "1", "true"} and context.get("evidence_present") in {"present", "1", "true"}:
        # The old PG-348 rows did not expose an explicit oracle slot.  This is
        # a bounded abstract default, never an evaluator answer or effect
        # literal; new collectors should still emit oracle_ref explicitly.
        output["oracle_ref"] = "typed_effect"
    if forbidden:
        reasons.append("forbidden_literal_or_sidecar")
    if has_failure:
        reasons.append("failure_requires_repair")
    if missing:
        reasons.append("missing_required_observation")
    if forbidden:
        output.update(question="ask_typed", next_action="ask_typed", repair_action="observe", safe_to_send="0")
    elif has_failure:
        output.update(question="ask_failure", next_action="repair", repair_action="one_variable", probe_variant_ref="one_variable_repair", safe_to_send="0")
    elif missing:
        output.update(question="ask_typed", next_action="ask_typed", repair_action="observe", probe_variant_ref="unsupported_abstain", safe_to_send="0")
    elif output["safe_to_send"] == "1":
        if output["probe_variant_ref"] not in {"reference", "reference_shape", "source_attested_candidate", "runtime_canary", "fresh_replay"}:
            reasons.append("unbound_probe_variant")
            output.update(next_action="abstain", probe_variant_ref="unsupported_abstain", safe_to_send="0")
        elif output["oracle_ref"] in {"none", "unknown"}:
            reasons.append("typed_oracle_missing")
            output.update(next_action="ask_typed", question="ask_oracle", probe_variant_ref="unsupported_abstain", safe_to_send="0")
    else:
        output["next_action"] = output["next_action"] if output["next_action"] in _ALLOWED["next_action"] else "abstain"
    return {
        "schema_version": SCHEMA_VERSION,
        "target": output,
        "missing_fields": missing,
        "failure_detected": has_failure,
        "forbidden_fields": forbidden,
        "reasons": sorted(set(reasons)),
        "safe_to_send": output["safe_to_send"] == "1",
        "evaluator_binding_required": True,
        "raw_payload_in_output": False,
    }


def decode_rule_ir(context_tokens: Sequence[str], proposal: Mapping[str, Any] | None = None) -> list[str]:
    """Return a stable abstract target token sequence for next-token evaluation."""

    result = constrain_rule_ir(context_tokens, proposal)
    target = result["target"]
    return ["[TARGET_BOS]", *[f"{key}={target[key]}" for key in TARGET_KEYS], "[TARGET_EOS]"]


def audit_rule_ir_output(context_tokens: Sequence[str], proposal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = constrain_rule_ir(context_tokens, proposal)
    target = result["target"]
    invalid = [key for key, allowed in _ALLOWED.items() if target.get(key) not in allowed]
    safe_gate = bool(result["safe_to_send"]) and not result["missing_fields"] and not result["failure_detected"] and not result["forbidden_fields"]
    return {
        "schema_version": f"{SCHEMA_VERSION}-audit",
        "status": "passed" if not invalid and (not target["safe_to_send"] or safe_gate) else "blocked",
        "invalid_slots": invalid,
        "safe_gate": safe_gate,
        "missing_fields": result["missing_fields"],
        "failure_detected": result["failure_detected"],
        "forbidden_fields": result["forbidden_fields"],
        "promotion": {"training": False, "memory": False, "payload": False, "vulnerability": False},
    }


__all__ = ["SCHEMA_VERSION", "TARGET_KEYS", "audit_rule_ir_output", "constrain_rule_ir", "decode_rule_ir"]
