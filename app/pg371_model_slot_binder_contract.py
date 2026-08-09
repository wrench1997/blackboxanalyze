"""PG-371 abstract model-slot selection to allowlisted binder contract.

This module is deliberately a contract adapter, not a network client.  It
accepts only bounded Rule-IR slots, delegates allowlist validation to the
reviewed PG-350 binder, and stops before template expansion or wire creation
until an independent evaluator supplies all typed evidence.  Thus a
``model_selected=True`` decision is distinguishable from both ``ASK`` and a
confirmed evaluator result.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.pg350_runtime_payload_binder import (
    ALLOWED_ENCODINGS,
    ALLOWED_ORACLES,
    ALLOWED_SHAPES,
    ALLOWED_TRANSPORTS,
    ALLOWED_VARIANTS,
    _validate_model_slots,
)
from app.pg361_payload_shape_slots import ALLOWED_SYNTAX_CATEGORIES

SCHEMA_VERSION = "pg371-model-slot-binder-contract-v1"
ROLES = ("candidate", "reference", "negative", "replay")
REQUIRED_SLOTS = (
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "payload_shape_ref",
    "oracle_ref",
    "probe_variant_ref",
    "safe_to_send",
)
# PG-370's shared heads supervise all 13 ordered Rule-IR slots.  The binder
# itself gates only the eight transport/action slots above, but a model
# proposal is not considered complete if it silently drops ASK, repair, or
# negative-control state.
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
TARGET_BOS = "[TARGET_BOS]"
TARGET_EOS = "[TARGET_EOS]"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
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
}
_FORBIDDEN_VALUE_FRAGMENTS = (
    "http://",
    "https://",
    "javascript:",
    "document.cookie",
    "<script",
    "probe.invalid",
)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _raw_or_forbidden(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_KEYS or any(part in key_text for part in ("raw_", "response_body", "evaluator_")):
                return f"forbidden_key:{path}.{key}"
            found = _raw_or_forbidden(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found = _raw_or_forbidden(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        folded = value.casefold()
        for fragment in _FORBIDDEN_VALUE_FRAGMENTS:
            if fragment in folded:
                return f"forbidden_value:{fragment}"
    return None


def parse_target_slots(tokens: Sequence[str]) -> dict[str, Any] | None:
    """Parse a bounded abstract target sequence; never accepts raw literals."""

    values = [str(token) for token in tokens]
    if not values or values[0] != TARGET_BOS or values[-1] != TARGET_EOS:
        return None
    result: dict[str, Any] = {}
    for token in values[1:-1]:
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            return None
        result[key] = value
    if _raw_or_forbidden(result):
        return None
    if not set(MODEL_SLOTS).issubset(result):
        return None
    return result


def _expected_transport(method: str) -> str:
    normalized = str(method).upper()
    if normalized == "GET":
        return "get_query"
    if normalized == "POST":
        return "post_form"
    raise ValueError("PG-371 only supports GET/POST")


def _evidence_complete(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    if not isinstance(evidence, Mapping):
        return False, "missing_typed_evidence"
    if evidence.get("typed_available") is not True:
        return False, "typed_oracle_unavailable"
    if evidence.get("fresh_reset") is not True:
        return False, "fresh_reset_unobserved"
    if evidence.get("candidate_reference_negative_replay") is not True:
        return False, "role_triplet_or_replay_unobserved"
    if evidence.get("network_none") is not True or evidence.get("loopback_only") is not True:
        return False, "network_contract_unobserved"
    digest = str(evidence.get("evidence_sha256", "")).casefold()
    if not _HEX64.fullmatch(digest):
        return False, "evidence_sha256_missing"
    if evidence.get("context_firewall_closed") is not True:
        return False, "context_firewall_unobserved"
    return True, "complete"


def select_and_bind_model_slots(
    rule_ir: Mapping[str, Any],
    *,
    expected_method: str,
    role: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select model slots and fail closed before any template/wire operation.

    ``model_selected`` records that a bounded decoder output was received;
    ``typed_effect_confirmed`` is never set here.  A complete evidence record
    still returns ``template_binding_required`` because PG-371 deliberately
    owns the contract boundary, not a live evaluator.
    """

    role_text = str(role)
    if role_text not in ROLES:
        raise ValueError("PG-371 role is not allow-listed")
    if not isinstance(rule_ir, Mapping):
        raise ValueError("PG-371 Rule-IR must be a mapping")
    raw_error = _raw_or_forbidden(rule_ir)
    if raw_error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected_raw_or_evaluator_slot",
            "model_selected": True,
            "safe_to_send": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "role": role_text,
            "reason": raw_error,
        }
    missing = [slot for slot in MODEL_SLOTS if slot not in rule_ir]
    if missing:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected_incomplete_slots",
            "model_selected": True,
            "safe_to_send": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "role": role_text,
            "missing_slots": missing,
        }
    if rule_ir.get("safe_to_send") not in {True, 1, "1", "true"}:
        # A decoder may intentionally abstain after an ASK/negative decision.
        # Preserve that model selection state without invoking the sendable
        # PG-350 validator, which correctly rejects ``safe_to_send=false``.
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "abstain_safe_to_send_false",
            "model_selected": True,
            "safe_to_send": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "role": role_text,
            "selected_slots": {key: rule_ir.get(key) for key in REQUIRED_SLOTS},
        }
    try:
        normalized = _validate_model_slots(rule_ir)
        expected_transport = _expected_transport(expected_method)
        if normalized["transport_ref"] != expected_transport:
            raise ValueError("transport_method_mismatch")
        syntax = str(rule_ir["syntax_category_ref"]).casefold().replace("-", "_")
        if syntax not in ALLOWED_SYNTAX_CATEGORIES:
            raise ValueError("syntax_category_ref_not_allowlisted")
        if str(rule_ir["payload_shape_ref"]) not in ALLOWED_SHAPES:
            raise ValueError("payload_shape_ref_not_allowlisted")
        if str(rule_ir["oracle_ref"]) not in ALLOWED_ORACLES:
            raise ValueError("oracle_ref_not_allowlisted")
        if str(rule_ir["encoding_ref"]) not in ALLOWED_ENCODINGS:
            raise ValueError("encoding_ref_not_allowlisted")
        if str(rule_ir["transport_ref"]) not in ALLOWED_TRANSPORTS:
            raise ValueError("transport_ref_not_allowlisted")
        if str(rule_ir["probe_variant_ref"]) not in ALLOWED_VARIANTS:
            raise ValueError("probe_variant_ref_not_allowlisted")
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected_invalid_allowlist_slot",
            "model_selected": True,
            "safe_to_send": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "role": role_text,
            "reason": str(error),
        }
    selected = {
        "transport_ref": str(rule_ir["transport_ref"]),
        "field_role_ref": str(rule_ir["field_role_ref"]),
        "encoding_ref": str(rule_ir["encoding_ref"]),
        "syntax_category_ref": str(rule_ir["syntax_category_ref"]).casefold().replace("-", "_"),
        "payload_shape_ref": str(rule_ir["payload_shape_ref"]),
        "oracle_ref": str(rule_ir["oracle_ref"]),
        "probe_variant_ref": str(rule_ir["probe_variant_ref"]),
        "safe_to_send": rule_ir.get("safe_to_send") in {True, 1, "1", "true"},
    }
    if selected["safe_to_send"] is not True:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "abstain_safe_to_send_false",
            "model_selected": True,
            "safe_to_send": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "role": role_text,
            "selected_slots": selected,
        }
    complete, reason = _evidence_complete(evidence or {})
    if not complete:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_missing_typed_evidence",
            "model_selected": True,
            "safe_to_send": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "role": role_text,
            "selected_slots": selected,
            "reason": reason,
        }
    # PG-371 stops at the binder boundary.  Even complete evidence cannot
    # create a wire in this planning/contract adapter.
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "template_binding_required",
        "model_selected": True,
        "safe_to_send": False,
        "typed_effect_confirmed": False,
        "wire_created": False,
        "role": role_text,
        "selected_slots": selected,
        "evidence_sha256": str(dict(evidence or {}).get("evidence_sha256", "")).casefold(),
        "reason": "pg371_contract_stops_before_wire",
    }


def planning_binding_row(*, seed: int, route_ref_sha256: str, method: str, response_shape: str, role: str) -> dict[str, Any]:
    """Build a non-sendable PG-368 plan row with explicit model selection state."""

    if str(role) not in ROLES:
        raise ValueError("invalid role")
    return {
        "seed": int(seed),
        "route_ref_sha256": str(route_ref_sha256),
        "method": str(method).upper(),
        "response_shape_ref": str(response_shape),
        "role": str(role),
        "model_selected": False,
        "model_selection_status": "not_available_planning_only",
        "binding_status": "ASK_missing_model_slots_and_typed_evaluator",
        "safe_to_send": False,
        "typed_effect_confirmed": False,
        "wire_created": False,
        "fresh_reset_required": True,
        "fresh_reset_observed": False,
        "candidate_reference_negative_replay_required": True,
        "candidate_reference_negative_replay_observed": False,
        "typed_evidence_sha256_required": True,
        "typed_evidence_sha256_observed": False,
        "context_firewall_closed": True,
        "target_contacted": False,
    }


__all__ = [
    "ALLOWED_ENCODINGS",
    "ALLOWED_ORACLES",
    "ALLOWED_SHAPES",
    "ALLOWED_SYNTAX_CATEGORIES",
    "REQUIRED_SLOTS",
    "MODEL_SLOTS",
    "ROLES",
    "SCHEMA_VERSION",
    "parse_target_slots",
    "planning_binding_row",
    "select_and_bind_model_slots",
    "sha256_json",
]
