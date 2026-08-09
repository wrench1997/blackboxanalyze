"""Independent PG-122 local target for an authorization-transition family."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request


SCHEMA_VERSION = "pg122-theta-authorization-target-v1"
EVIDENCE_SCHEMA_VERSION = "pg122-theta-authorization-evidence-v1"
RESET_SCHEMA_VERSION = "pg122-theta-reset-v1"
SURFACES = ("authorization", "decoy", "steady", "blind")
_MARKER = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass
class _State:
    seed: int
    strength: int
    target_instance_id: str
    epoch: int = 0


async def _payload(request: Request) -> dict[str, Any]:
    if request.method == "GET":
        return {str(key): str(value) for key, value in request.query_params.items()}
    try:
        value = await request.json()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(400, "POST body must be an object") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "POST body must be an object")
    return {str(key): item for key, item in value.items()}


def _marker(value: Any) -> str:
    text = str(value or "")
    if not _MARKER.fullmatch(text):
        raise HTTPException(400, "marker must be bounded")
    return text


def _reset(state: _State) -> dict[str, Any]:
    body = {"schema_version": RESET_SCHEMA_VERSION, "target_instance_id": state.target_instance_id, "reset_epoch": state.epoch, "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "state_change_allowed": False, "external_network": False}
    return {**body, "evidence_hash": _hash_json(body)}


def create_app(seed: int, *, decoy_strength: int = 1) -> FastAPI:
    strength = max(0, min(int(decoy_strength), 2))
    state = _State(seed=int(seed), strength=strength, target_instance_id=f"pg122-theta-target-{int(seed)}-d{strength}")
    app = FastAPI(title="PG-122 Theta Authorization Target", version="1.0")

    def observe(surface: str, payload: dict[str, Any]) -> dict[str, Any]:
        if surface not in SURFACES:
            raise HTTPException(404, "unknown PG-122 surface")
        if surface == "authorization":
            candidate = str(payload.get("access_mode", "normal")) == "elevated"
            available, effect, atom, shape, status, shape_changed, authorization_changed = True, candidate, "authorization_transition", "decision-v5", "2xx", False, candidate
        elif surface == "decoy":
            candidate = str(payload.get("layout_mode", "normal")) == "elevated"
            available, effect, atom, shape, status, shape_changed, authorization_changed = True, False, "layout_only", "decision-v5", "2xx", False, False
        elif surface == "steady":
            candidate = False
            available, effect, atom, shape, status, shape_changed, authorization_changed = True, False, "no_effect", "decision-v5", "2xx", False, False
        else:
            candidate = str(payload.get("phase", "normal")) == "canary" and bool(_marker(payload.get("marker")))
            available, effect, atom, shape, status, shape_changed, authorization_changed = False, False, "withheld_effect", "decision-v5", "2xx", False, False
        slot = f"pg122-theta-slot-{_hash_json(surface)[:16]}"
        safety = {"external_network": False, "script_execution": False, "database_write": False, "navigation": False, "credentials_accessed": False, "state_mutated": False}
        evaluator = {"schema_version": EVIDENCE_SCHEMA_VERSION, "oracle_id": f"pg122-theta-{surface}-typed-v1", "surface_slot": slot, "typed_available": available, "candidate_signal": candidate, "safety": safety}
        if available:
            evaluator["typed_effect"] = effect
            evaluator["effect_atoms"] = [atom] if effect else []
        response_projection = {"status_class": status, "body_length_bucket": "256-4095", "shape_class": shape, "candidate_signal": candidate, "policy_header_changed": False, "shape_changed": shape_changed, "location_changed": False, "metadata_changed": False, "authorization_changed": authorization_changed, "transition_delta": "authorization" if authorization_changed else "none", "noise_bucket": (state.seed * 7 + state.epoch * 19 + strength * 3 + len(surface)) % 11}
        evidence_body = {"schema_version": EVIDENCE_SCHEMA_VERSION, "target_instance_id": state.target_instance_id, "surface": surface, "decoy_strength": strength, "reset_epoch": state.epoch, "candidate_signal": candidate, "typed_evaluator_available": available, "evaluator": evaluator, "response_projection": response_projection, "safety": safety}
        evidence = {**evidence_body, "evidence_hash": _hash_json(evidence_body)}
        return {"schema_version": SCHEMA_VERSION, "implementation": "pg122-theta-authorization-target", "surface": surface, "surface_slot": slot, "candidate_signal": candidate, "response_projection": response_projection, "evaluator": evaluator, "evidence": evidence}

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "implementation": "pg122-theta-authorization-target", "schema_version": SCHEMA_VERSION}

    @app.post("/v9/reset")
    async def reset() -> dict[str, Any]:
        state.epoch += 1
        return {"status": "reset", "reset": _reset(state)}

    @app.get("/v9/check/{surface}")
    async def check_get(surface: str, request: Request) -> dict[str, Any]:
        return observe(surface, await _payload(request))

    @app.post("/v9/check/{surface}")
    async def check_post(surface: str, request: Request) -> dict[str, Any]:
        return observe(surface, await _payload(request))

    app.state.pg122_seed = int(seed)
    app.state.pg122_decoy_strength = strength
    return app


__all__ = ["EVIDENCE_SCHEMA_VERSION", "RESET_SCHEMA_VERSION", "SCHEMA_VERSION", "SURFACES", "create_app"]
