"""Independent PG-117 target with a route-like double-encoding surface.

The route surface is an inert projection only: it reports a bounded location
delta and never navigates, redirects a browser, touches a network or executes
anything.  Its implementation and names are intentionally separate from the
PG-116 alpha/beta targets.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request


SCHEMA_VERSION = "pg117-gamma-independent-target-v1"
EVIDENCE_SCHEMA_VERSION = "pg117-gamma-typed-evidence-v1"
RESET_SCHEMA_VERSION = "pg117-gamma-reset-v1"
SURFACES = ("route", "decoy", "steady", "blind")
_MARKER = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass
class _State:
    seed: int
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
    body = {
        "schema_version": RESET_SCHEMA_VERSION,
        "target_instance_id": state.target_instance_id,
        "reset_epoch": state.epoch,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }
    return {**body, "evidence_hash": _hash_json(body)}


def create_app(seed: int) -> FastAPI:
    state = _State(seed=int(seed), target_instance_id=f"pg117-gamma-target-{int(seed)}")
    app = FastAPI(title="PG-117 Gamma Independent Target", version="1.0")

    def inspect(surface: str, payload: dict[str, Any]) -> dict[str, Any]:
        if surface not in SURFACES:
            raise HTTPException(404, "unknown PG-117 surface")
        if surface == "route":
            candidate = str(payload.get("route_mode", "baseline")) == "alternate"
            available, effect, atom, shape, status, shape_changed, location_changed = True, candidate, "route_transition", "route-transition", "3xx", True, candidate
        elif surface == "decoy":
            candidate = str(payload.get("shape_mode", "baseline")) == "alternate"
            available, effect, atom, shape, status, shape_changed, location_changed = True, False, "shape_delta", "route-decoy", "2xx", candidate, False
        elif surface == "steady":
            candidate = False
            available, effect, atom, shape, status, shape_changed, location_changed = True, False, "no_effect", "steady", "2xx", False, False
        else:
            candidate = str(payload.get("mode", "control")) == "canary" and bool(_marker(payload.get("marker")))
            available, effect, atom, shape, status, shape_changed, location_changed = False, False, "withheld_effect", "opaque-route", "2xx", candidate, False
        slot = f"pg117-gamma-slot-{_hash_json(surface)[:16]}"
        evaluator = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "oracle_id": f"pg117-gamma-{surface}-typed-v1",
            "surface_slot": slot,
            "typed_available": available,
            "candidate_signal": candidate,
            "safety": {"external_network": False, "script_execution": False, "database_write": False, "navigation": False, "credentials_accessed": False, "state_mutated": False},
        }
        if available:
            evaluator["typed_effect"] = effect
            evaluator["effect_atoms"] = [atom] if effect else []
        response_projection = {
            "status_class": status,
            "body_length_bucket": "256-4095",
            "shape_class": shape,
            "candidate_signal": candidate,
            "policy_header_changed": False,
            "shape_changed": shape_changed,
            "location_changed": location_changed,
            "noise_bucket": (state.seed + state.epoch * 7 + len(surface)) % 11,
        }
        evidence_body = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "target_instance_id": state.target_instance_id,
            "surface": surface,
            "reset_epoch": state.epoch,
            "candidate_signal": candidate,
            "typed_evaluator_available": available,
            "evaluator": evaluator,
            "response_projection": response_projection,
            "safety": evaluator["safety"],
        }
        evidence = {**evidence_body, "evidence_hash": _hash_json(evidence_body)}
        return {"schema_version": SCHEMA_VERSION, "implementation": "pg117-gamma-independent-target", "surface": surface, "surface_slot": slot, "candidate_signal": candidate, "response_projection": response_projection, "evaluator": evaluator, "evidence": evidence}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "implementation": "pg117-gamma-independent-target", "schema_version": SCHEMA_VERSION}

    @app.post("/v5/reset")
    async def reset() -> dict[str, Any]:
        state.epoch += 1
        return {"status": "reset", "reset": _reset(state)}

    @app.get("/v5/inspect/{surface}")
    async def inspect_get(surface: str, request: Request) -> dict[str, Any]:
        return inspect(surface, await _payload(request))

    @app.post("/v5/inspect/{surface}")
    async def inspect_post(surface: str, request: Request) -> dict[str, Any]:
        return inspect(surface, await _payload(request))

    app.state.pg117_seed = int(seed)
    return app


__all__ = ["EVIDENCE_SCHEMA_VERSION", "RESET_SCHEMA_VERSION", "SCHEMA_VERSION", "SURFACES", "create_app"]

