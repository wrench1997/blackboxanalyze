"""Dependency-free dynamic backend for the PG-388 logic lab.

The service is deliberately a *logic-state simulator*, not a vulnerable
production application.  It accepts only case/role/feedback enums and emits
bounded Rule-IR projections.  State is an in-memory episode counter; no
credentials, identifiers, raw input, response bodies, network calls or
persistent writes are handled.  A reviewed operator may run it in a fresh
network-none container for a frontend demonstration.
"""

from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

from app.pg388_logic_invariant_projection import (
    ALL_LOGIC_CASES,
    FEEDBACK_STATES,
    LOGIC_CASES,
    ROLES,
    SUPPLEMENTAL_LOGIC_CASES,
    project_logic_case,
    project_logic_observation,
)


IMPLEMENTATION_ID = "pg388-logic-lab-backend-a"
SCHEMA_VERSION = "pg388-logic-lab-backend-v1"
MAX_BODY_BYTES = 4096
_CASE_REFS = frozenset(item["case_ref"] for item in ALL_LOGIC_CASES)
_state = {"episode_count": 0, "reset_count": 0}

# Concrete, bounded state-machine canaries for the frontend demo.  The
# evaluator chooses only scenario/role/phase enums; it never accepts a user
# supplied identifier, price, coupon, token, or arbitrary request value.
_CANARY_CASES = {
    "nonce_replay": "replay_protection",
    "coupon_reuse_boundary": "coupon_reuse",
    "subject_resource_scope": "horizontal_authorization",
    "install_reentry_gate": "installation_gate",
    "purchase_price_binding": "transaction_price",
    "purchase_concurrency_lock": "transaction_concurrency",
    "purchase_status_transition": "transaction_status",
    "purchase_quantity_floor": "transaction_quantity",
    "identity_canonicalization": "identity_canonicalization",
    "password_reset_subject_binding": "password_reset",
    "two_factor_reset_binding": "two_factor_reset",
    "oauth_second_factor": "oauth_second_factor",
    "activation_link_second_factor": "activation_second_factor",
    "csrf_disable_second_factor": "factor_settings_csrf",
    "captcha_reuse": "captcha_state",
    "captcha_predictability": "captcha_entropy",
    "captcha_response_exposure": "captcha_response",
    "captcha_client_validation": "captcha_validation",
    "captcha_delivery_abuse": "captcha_delivery",
    "session_fixation_boundary": "session_rotation",
    "session_guessing": "session_entropy",
    "session_forgery": "session_integrity",
    "session_leakage": "session_exposure",
    "query_object_scope": "query_authorization",
    "vertical_role_scope": "vertical_authorization",
    "query_identifier_entropy": "identifier_enumeration",
    "execution_order": "execution_order",
    "sensitive_projection": "information_projection",
}
# These cases deliberately model a bounded defective branch in the local
# simulator.  The values are abstract effect/state buckets only.
_CANARY_RISK = {
    "install_reentry_gate": ("setup_reentered", "unexpected_reconfigure"),
    "purchase_price_binding": ("client_total_accepted", "total_mismatch"),
    "purchase_concurrency_lock": ("duplicate_commit", "order_version_reused"),
    "purchase_status_transition": ("status_advanced", "payment_order_violation"),
    "purchase_quantity_floor": ("negative_quantity_accepted", "quantity_delta_nonzero"),
    "identity_canonicalization": ("duplicate_identity", "normalization_bypass"),
    "password_reset_subject_binding": ("subject_mismatch_accepted", "reset_scope_crossed"),
    "two_factor_reset_binding": ("session_upgraded_without_factor", "factor_order_bypassed"),
    "oauth_second_factor": ("session_upgraded_without_factor", "oauth_factor_skipped"),
    "activation_link_second_factor": ("activation_upgraded_without_factor", "activation_order_bypassed"),
    "csrf_disable_second_factor": ("factor_disabled_without_csrf", "csrf_binding_missing"),
    "captcha_reuse": ("challenge_reused", "verification_replay"),
    "captcha_predictability": ("challenge_predictable", "challenge_entropy_low"),
    "captcha_response_exposure": ("challenge_returned", "challenge_response_exposed"),
    "captcha_client_validation": ("client_validation_accepted", "server_validation_missing"),
    "captcha_delivery_abuse": ("delivery_unlimited", "delivery_budget_bypassed"),
    "session_fixation_boundary": ("session_not_rotated", "identity_transition_shared"),
    "session_guessing": ("session_predictable", "session_entropy_low"),
    "session_forgery": ("session_forged", "session_integrity_missing"),
    "session_leakage": ("session_exposed", "session_secret_projected"),
    "query_object_scope": ("cross_scope_read", "owner_binding_bypassed"),
    "vertical_role_scope": ("admin_action_allowed", "role_guard_bypassed"),
    "query_identifier_entropy": ("ordered_identifier", "enumeration_signal"),
    "execution_order": ("mutate_before_deny", "side_effect_before_check"),
    "sensitive_projection": ("secret_shape_exposed", "response_projection_verbose"),
}
_CANARY_PHASES = frozenset({"baseline", "candidate", "reference", "negative", "replay"})
_canary_state = {"nonce_effect_count": 0, "coupon_use_count": 0, "generic_effect_counts": {key: 0 for key in _CANARY_RISK}}


def _network_mode() -> str:
    return os.environ.get("PG388_NETWORK_MODE", "none")


def _loopback_only() -> bool:
    return _network_mode() == "none"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _json_response(start_response: Callable[..., Any], status: int, document: Mapping[str, Any]) -> list[bytes]:
    body = _canonical(document)
    phrase = "OK" if status < 400 else "Bad Request" if status < 500 else "Server Error"
    start_response(f"{status} {phrase}", [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body))), ("Access-Control-Allow-Origin", "http://localhost:3000"), ("Access-Control-Allow-Methods", "GET, POST")])
    return [body]


def _read_json(environ: Mapping[str, Any], *, allowed: set[str] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        length = min(max(int(str(environ.get("CONTENT_LENGTH", "0") or "0")), 0), MAX_BODY_BYTES)
    except ValueError:
        length = 0
    stream = environ.get("wsgi.input")
    raw = stream.read(length) if stream is not None and length else b""
    if len(raw) > MAX_BODY_BYTES:
        return None, "body_too_large"
    try:
        value = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "json_invalid"
    if not isinstance(value, dict):
        return None, "json_object_required"
    allowed_keys = allowed or {"case_ref", "role", "feedback_state"}
    if set(value) - allowed_keys:
        return None, "abstract_enum_fields_only"
    return value, None


def _validate_request(document: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    case_ref = str(document.get("case_ref", ""))
    role = str(document.get("role", ""))
    feedback = str(document.get("feedback_state", ""))
    if case_ref not in _CASE_REFS:
        return None, "unknown_case_ref"
    if role not in ROLES:
        return None, "unknown_role"
    if feedback not in FEEDBACK_STATES:
        return None, "unknown_feedback_state"
    return {"case_ref": case_ref, "role": role, "feedback_state": feedback}, None


def _validate_canary_request(document: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    case_ref = str(document.get("case_ref", ""))
    role = str(document.get("role", ""))
    phase = str(document.get("phase", ""))
    if case_ref not in _CANARY_CASES:
        return None, "unknown_canary_case"
    if role not in ROLES:
        return None, "unknown_role"
    if phase not in _CANARY_PHASES:
        return None, "unknown_canary_phase"
    if phase == "baseline" and role != "candidate":
        return None, "baseline_role_required"
    if phase != "baseline" and phase != role:
        return None, "canary_role_phase_mismatch"
    return {"case_ref": case_ref, "role": role, "phase": phase}, None


def _canary_state_bucket(case_ref: str) -> str:
    if case_ref == "nonce_replay":
        return "zero_effect" if _canary_state["nonce_effect_count"] == 0 else "one_or_more_effects"
    if case_ref == "coupon_reuse_boundary":
        return "unused" if _canary_state["coupon_use_count"] == 0 else "consumed_once_or_more"
    if case_ref in _CANARY_RISK:
        return "zero_effect" if _canary_state["generic_effect_counts"][case_ref] == 0 else "one_or_more_effects"
    return "subject_scope_unmodified"


def _canary_result(request: Mapping[str, str]) -> dict[str, Any]:
    case_ref = request["case_ref"]
    role = request["role"]
    phase = request["phase"]
    before = _canary_state_bucket(case_ref)
    violated = False
    state_delta = "zero"
    effect_shape = "denied_shape"
    action_shape = "observe_only"

    if phase == "baseline":
        effect_shape = "baseline_shape"
    elif case_ref == "nonce_replay":
        if role == "candidate":
            _canary_state["nonce_effect_count"] += 1
            state_delta, effect_shape, action_shape = "one_effect", "accepted_once", "candidate_apply"
        elif role == "replay":
            violated = _canary_state["nonce_effect_count"] > 0
            if violated:
                _canary_state["nonce_effect_count"] += 1
                state_delta, effect_shape, action_shape = "duplicate_effect", "accepted_replay", "candidate_replay"
            else:
                state_delta, effect_shape, action_shape = "zero", "missing_baseline", "ask_baseline"
        elif role == "reference":
            state_delta, effect_shape, action_shape = "zero", "rejected_replay", "reference_guard"
        else:
            state_delta, effect_shape, action_shape = "zero", "rejected_replay", "negative_guard"
    elif case_ref == "coupon_reuse_boundary":
        if role == "candidate":
            _canary_state["coupon_use_count"] += 1
            state_delta, effect_shape, action_shape = "discount_once", "benefit_applied", "candidate_apply"
        elif role == "replay":
            violated = _canary_state["coupon_use_count"] > 0
            if violated:
                _canary_state["coupon_use_count"] += 1
                state_delta, effect_shape, action_shape = "discount_reused", "benefit_applied_again", "candidate_replay"
            else:
                state_delta, effect_shape, action_shape = "zero", "missing_baseline", "ask_baseline"
        elif role == "reference":
            state_delta, effect_shape, action_shape = "zero", "coupon_reuse_denied", "reference_guard"
        else:
            state_delta, effect_shape, action_shape = "zero", "coupon_reuse_denied", "negative_guard"
    elif case_ref in _CANARY_RISK:
        if role in {"candidate", "replay"}:
            _canary_state["generic_effect_counts"][case_ref] += 1
            effect_shape, state_delta = _CANARY_RISK[case_ref]
            action_shape = "candidate_apply" if role == "candidate" else "candidate_replay"
            violated = True
        elif role == "reference":
            state_delta, effect_shape, action_shape = "zero", "reference_denied_shape", "reference_guard"
        else:
            state_delta, effect_shape, action_shape = "zero", "negative_denied_shape", "negative_guard"
    else:  # subject_resource_scope: deterministic cross-owner authorization canary.
        if role in {"candidate", "replay"}:
            violated = True
            state_delta, effect_shape, action_shape = "read_cross_scope", "resource_visible", "candidate_scope_bypass"
        elif role == "reference":
            state_delta, effect_shape, action_shape = "zero", "resource_denied", "reference_scope_guard"
        else:
            state_delta, effect_shape, action_shape = "zero", "resource_denied", "negative_scope_guard"

    after = _canary_state_bucket(case_ref)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "typed_local_canary_result",
        "canary_case": _CANARY_CASES[case_ref],
        "case_ref": case_ref,
        "role": role,
        "phase": phase,
        "state_before": before,
        "state_after": after,
        "state_delta": state_delta,
        "effect_shape": effect_shape,
        "action_shape": action_shape,
        "invariant_holds": not violated,
        "vulnerable_effect": violated,
        "typed_observation": True,
        "negative_control_clean": role == "negative" and not violated,
        "safe_to_send": False,
        "target_contacted": False,
        "external_network": False,
        "persistent_storage": False,
        "fresh_reset_required": True,
        "evaluator_sidecar": {"scope": "local_disposable_canary_only", "raw_values_stored": False},
    }


def _canary_manifest() -> dict[str, Any]:
    return {
        "endpoint": "/api/canary",
        "cases": [{"case_ref": key, "surface": value} for key, value in _CANARY_CASES.items()],
        "phases": sorted(_CANARY_PHASES),
        "roles": list(ROLES),
        "input_policy": "case_ref_role_phase_enums_only",
        "concrete_effect_cases": list(_CANARY_RISK),
        "vulnerability_claim": "local_typed_state_shape_only",
        "safe_to_send": False,
    }


def manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dynamic_fixture_only_unbound",
        "implementation_id": IMPLEMENTATION_ID,
        "source_sha256": source_digest(),
        "case_count": len(LOGIC_CASES),
        "supplemental_case_count": len(SUPPLEMENTAL_LOGIC_CASES),
        "roles": list(ROLES),
        "feedback_states": list(FEEDBACK_STATES),
        "transport": ["GET", "POST"],
        "network": {"mode": _network_mode(), "loopback_only": _loopback_only(), "external_network": False},
        "state": {"persistent_storage": False, "business_write": False, "disposable_episode_only": True},
        "model_boundary": {"raw_values": False, "raw_response": False, "oracle_answer_in_context": False, "safe_to_send": False},
        "route_classes": ["health", "manifest", "cases", "supplemental-cases", "reset", "observe", "episode", "canary"],
        "canary": _canary_manifest(),
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def _episode_result(request: Mapping[str, str], *, endpoint: str) -> dict[str, Any]:
    global _state
    _state["episode_count"] += 1
    projection = project_logic_case(request["case_ref"], role=request["role"], feedback_state=request["feedback_state"])
    typed_effect = request["feedback_state"] == "typed_effect" and request["role"] != "negative"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_logic_projection",
        "endpoint": endpoint,
        "episode_index_bucket": "first" if _state["episode_count"] == 1 else "repeated",
        "case_ref": request["case_ref"],
        "role": request["role"],
        "feedback_state": request["feedback_state"],
        "next_action": projection["target_projection"]["next_action"],
        "repair_action": projection["target_projection"]["repair_action"],
        "safe_to_send": False,
        "typed_effect": typed_effect,
        "negative_control_clean": request["role"] == "negative" and not typed_effect,
        "state_transition": "abstract_typed_shape" if typed_effect else "no_business_state_write",
        "state_delta": "zero",
        "fresh_reset_required": True,
        "context_tokens": projection["context_tokens"],
        "target_tokens": projection["target_tokens"],
        "raw_values_stored": False,
        "external_network": False,
        "persistent_storage": False,
        "evaluator_sidecar": {"typed_effect": typed_effect, "evidence_scope": "local_episode_only", "evidence_sha256": hashlib.sha256(_canonical({"case_ref": request["case_ref"], "role": request["role"], "feedback_state": request["feedback_state"], "episode": _state["episode_count"]})).hexdigest()},
    }


def application(environ: Mapping[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    path = str(environ.get("PATH_INFO", ""))
    if path == "/health" and method == "GET":
        return _json_response(start_response, 200, {"status": "ok", "implementation_id": IMPLEMENTATION_ID, "network_mode": _network_mode(), "loopback_only": _loopback_only(), "external_network": False, "persistent_storage": False})
    if path == "/api/manifest" and method == "GET":
        return _json_response(start_response, 200, manifest())
    if path == "/api/cases" and method == "GET":
        return _json_response(start_response, 200, {"status": "ok", "cases": [{key: item[key] for key in ("case_ref", "surface", "state_model", "invariant", "precondition", "transition", "counterfactual", "observation", "failure")} for item in LOGIC_CASES]})
    if path == "/api/supplemental-cases" and method == "GET":
        return _json_response(start_response, 200, {"status": "supplemental_candidate_only", "training_eligible": False, "cases": [{key: item[key] for key in ("case_ref", "surface", "state_model", "invariant", "precondition", "transition", "counterfactual", "observation", "failure")} for item in SUPPLEMENTAL_LOGIC_CASES]})
    if path == "/api/reset" and method == "POST":
        _state["episode_count"] = 0
        _state["reset_count"] += 1
        _canary_state["nonce_effect_count"] = 0
        _canary_state["coupon_use_count"] = 0
        _canary_state["generic_effect_counts"] = {key: 0 for key in _CANARY_RISK}
        return _json_response(start_response, 200, {"status": "fresh_reset", "reset_count_bucket": "first" if _state["reset_count"] == 1 else "repeated", "state_clean": True, "state_delta": "zero", "persistent_storage": False, "external_network": False})
    if path == "/api/canary" and method == "POST":
        document, error = _read_json(environ, allowed={"case_ref", "role", "phase"})
        if error:
            return _json_response(start_response, 400, {"status": "ask", "ask_reason": error, "safe_to_send": False, "raw_values_stored": False})
        request, error = _validate_canary_request(document or {})
        if error:
            return _json_response(start_response, 400, {"status": "ask", "ask_reason": error, "safe_to_send": False, "raw_values_stored": False})
        return _json_response(start_response, 200, _canary_result(request or {}))
    if path in {"/api/observe", "/api/episode"} and method == "POST":
        document, error = _read_json(environ)
        if error:
            return _json_response(start_response, 400, {"status": "ask", "ask_reason": error, "safe_to_send": False, "raw_values_stored": False})
        request, error = _validate_request(document or {})
        if error:
            return _json_response(start_response, 400, {"status": "ask", "ask_reason": error, "safe_to_send": False, "raw_values_stored": False})
        if path == "/api/observe":
            result = project_logic_observation(request)
            result.update({"backend_status": "abstract_observation", "safe_to_send": False, "external_network": False, "persistent_storage": False})
            return _json_response(start_response, 200, result)
        return _json_response(start_response, 200, _episode_result(request, endpoint="episode"))
    return _json_response(start_response, 404, {"status": "not_found", "safe_to_send": False, "external_network": False})


app = application
wsgi_app = application


def serve() -> None:  # pragma: no cover
    from wsgiref.simple_server import make_server

    host = os.environ.get("PG388_BIND", "127.0.0.1")
    port = int(os.environ.get("PG388_PORT", "8088"))
    with make_server(host, port, application) as server:
        server.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    serve()


__all__ = ["IMPLEMENTATION_ID", "SCHEMA_VERSION", "ROLES", "FEEDBACK_STATES", "application", "app", "wsgi_app", "manifest", "serve", "source_digest"]
