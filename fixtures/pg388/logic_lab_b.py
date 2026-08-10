"""Independent bounded implementation for the PG-388 logic lab.

Implementation B deliberately uses a different state-transition table from
``logic_lab.py`` while preserving the same enum-only evaluator contract.  It
is a local holdout fixture: there are no accounts, prices, cookies, tokens,
database writes, external calls, or arbitrary request values.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from app.pg388_logic_invariant_projection import (
    ALL_LOGIC_CASES,
    FEEDBACK_STATES,
    LOGIC_CASES,
    ROLES,
    SUPPLEMENTAL_LOGIC_CASES,
)


IMPLEMENTATION_ID = "pg388-logic-lab-backend-b"
SCHEMA_VERSION = "pg388-logic-lab-backend-b-v1"
MAX_BODY_BYTES = 4096
_CASE_REFS = tuple(item["case_ref"] for item in ALL_LOGIC_CASES)
_CASE_SET = frozenset(_CASE_REFS)
_PHASES = frozenset({"baseline", "candidate", "reference", "negative", "replay"})
_state = {"reset_count": 0, "episode_count": 0, "candidate_seen": frozenset()}

# B has its own effect vocabulary and does not import A's state table.  The
# values are typed state shapes only; they are not payloads or business data.
_EFFECTS = {
    case_ref: (f"{case_ref}_guard_gap", "state_transition_shape")
    for case_ref in _CASE_REFS
}
_EFFECTS.update(
    {
        "purchase_concurrency_lock": ("version_guard_gap", "commit_order_shape"),
        "execution_order": ("check_order_gap", "side_effect_order_shape"),
        "query_object_scope": ("owner_binding_gap", "scope_read_shape"),
        "vertical_role_scope": ("role_binding_gap", "admin_transition_shape"),
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _network_mode() -> str:
    return os.environ.get("PG388_NETWORK_MODE", "none")


def _json_response(start_response: Callable[..., Any], status: int, document: Mapping[str, Any]) -> list[bytes]:
    body = _canonical(document)
    phrase = "OK" if status < 400 else "Bad Request" if status < 500 else "Server Error"
    start_response(
        f"{status} {phrase}",
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "http://localhost:3000"),
            ("Access-Control-Allow-Methods", "GET, POST"),
        ],
    )
    return [body]


def _read_json(environ: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
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
    if set(value) - {"case_ref", "role", "phase"}:
        return None, "abstract_enum_fields_only"
    return value, None


def _validate(document: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    case_ref, role, phase = (str(document.get(key, "")) for key in ("case_ref", "role", "phase"))
    if case_ref not in _CASE_SET:
        return None, "unknown_case_ref"
    if role not in ROLES:
        return None, "unknown_role"
    if phase not in _PHASES:
        return None, "unknown_phase"
    if phase == "baseline" and role != "candidate":
        return None, "baseline_role_required"
    if phase != "baseline" and phase != role:
        return None, "phase_role_mismatch"
    return {"case_ref": case_ref, "role": role, "phase": phase}, None


def _result(request: Mapping[str, str]) -> dict[str, Any]:
    global _state
    _state["episode_count"] += 1
    case_ref, role, phase = request["case_ref"], request["role"], request["phase"]
    before = "baseline_shape" if phase == "baseline" else "clean_state_shape"
    vulnerable = role in {"candidate", "replay"} and phase in {"candidate", "replay"}
    if role == "negative":
        vulnerable = False
    effect_shape, transition_shape = _EFFECTS[case_ref]
    if vulnerable:
        _state["candidate_seen"] = frozenset(set(_state["candidate_seen"]) | {case_ref})
        state_delta = "bounded_transition"
        action_shape = "candidate_apply" if role == "candidate" else "replay_apply"
        effect = effect_shape
    elif role == "reference":
        state_delta, action_shape, effect = "zero", "reference_guard", "reference_clean_shape"
    elif role == "negative":
        state_delta, action_shape, effect = "zero", "negative_guard", "negative_clean_shape"
    else:
        state_delta, action_shape, effect = "zero", "baseline_observe", "observe_shape"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "typed_local_holdout_result",
        "implementation_id": IMPLEMENTATION_ID,
        "case_ref": case_ref,
        "role": role,
        "phase": phase,
        "state_before": before,
        "state_after": "bounded_transition_shape" if vulnerable else "clean_state_shape",
        "state_delta": state_delta,
        "effect_shape": effect,
        "transition_shape": transition_shape,
        "action_shape": action_shape,
        "invariant_holds": not vulnerable,
        "vulnerable_effect": vulnerable,
        "typed_observation": True,
        "negative_control_clean": role == "negative" and not vulnerable,
        "safe_to_send": False,
        "target_contacted": False,
        "external_network": False,
        "persistent_storage": False,
        "fresh_reset_required": True,
        "evaluator_sidecar": {
            "scope": "local_disposable_holdout_only",
            "evidence_sha256": hashlib.sha256(_canonical({"case": case_ref, "role": role, "phase": phase, "n": _state["episode_count"]})).hexdigest(),
            "raw_values_stored": False,
        },
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
        "network": {"mode": _network_mode(), "loopback_only": _network_mode() == "none", "external_network": False},
        "state": {"persistent_storage": False, "business_write": False, "disposable_episode_only": True},
        "model_boundary": {"raw_values": False, "raw_response": False, "oracle_answer_in_context": False, "safe_to_send": False},
        "route_classes": ["health", "manifest", "cases", "supplemental-cases", "reset", "canary"],
        "implementation_independence": "separate_transition_table_and_module",
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def application(environ: Mapping[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    path = str(environ.get("PATH_INFO", ""))
    if path == "/health" and method == "GET":
        return _json_response(start_response, 200, {"status": "ok", "implementation_id": IMPLEMENTATION_ID, "external_network": False, "persistent_storage": False})
    if path == "/api/manifest" and method == "GET":
        return _json_response(start_response, 200, manifest())
    if path == "/api/cases" and method == "GET":
        return _json_response(start_response, 200, {"status": "ok", "cases": [{"case_ref": item["case_ref"], "surface": item["surface"]} for item in LOGIC_CASES]})
    if path == "/api/supplemental-cases" and method == "GET":
        return _json_response(start_response, 200, {"status": "supplemental_candidate_only", "training_eligible": False, "cases": [{"case_ref": item["case_ref"], "surface": item["surface"]} for item in SUPPLEMENTAL_LOGIC_CASES]})
    if path == "/api/reset" and method == "POST":
        _state["episode_count"] = 0
        _state["candidate_seen"] = frozenset()
        _state["reset_count"] += 1
        return _json_response(start_response, 200, {"status": "fresh_reset", "reset_count_bucket": "first" if _state["reset_count"] == 1 else "repeated", "state_clean": True, "persistent_storage": False, "external_network": False})
    if path == "/api/canary" and method == "POST":
        document, error = _read_json(environ)
        if error:
            return _json_response(start_response, 400, {"status": "ask", "ask_reason": error, "safe_to_send": False})
        request, error = _validate(document or {})
        if error:
            return _json_response(start_response, 400, {"status": "ask", "ask_reason": error, "safe_to_send": False})
        return _json_response(start_response, 200, _result(request or {}))
    return _json_response(start_response, 404, {"status": "not_found", "safe_to_send": False, "external_network": False})


app = application
wsgi_app = application


def serve() -> None:  # pragma: no cover
    from wsgiref.simple_server import make_server

    host = os.environ.get("PG388_BIND", "127.0.0.1")
    port = int(os.environ.get("PG388_PORT", "8089"))
    with make_server(host, port, application) as server:
        server.serve_forever()


__all__ = ["IMPLEMENTATION_ID", "SCHEMA_VERSION", "application", "app", "wsgi_app", "manifest", "serve", "source_digest"]


if __name__ == "__main__":  # pragma: no cover
    serve()
