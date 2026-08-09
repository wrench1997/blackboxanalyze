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
    allowed = {"case_ref", "role", "feedback_state"}
    if set(value) - allowed:
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
        "route_classes": ["health", "manifest", "cases", "supplemental-cases", "reset", "observe", "episode"],
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
        return _json_response(start_response, 200, {"status": "fresh_reset", "reset_count_bucket": "first" if _state["reset_count"] == 1 else "repeated", "state_clean": True, "state_delta": "zero", "persistent_storage": False, "external_network": False})
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
