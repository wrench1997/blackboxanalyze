"""Two local-only target profiles for PG-116 training trace collection.

The profiles intentionally use different route names and abstract parameter
names, while both expose the same bounded typed-effect contract.  Inputs are
inert role markers only; the target never executes markup, SQL, redirects or
external requests.  This module exists to produce real GET/POST/reset traces,
not to provide an attack surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request


SCHEMA_VERSION = "pg116-multisource-training-target-v1"
EVIDENCE_SCHEMA_VERSION = "pg116-multisource-typed-evidence-v1"
RESET_SCHEMA_VERSION = "pg116-multisource-reset-v1"
_MARKER = re.compile(r"^[A-Za-z0-9._-]{4,64}$")

SOURCE_SURFACES: dict[str, tuple[str, ...]] = {
    "alpha": ("policy", "decoy", "neutral", "opaque"),
    "beta": ("boundary", "layout", "steady", "blind"),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _marker(value: Any) -> str:
    text = str(value or "")
    if not _MARKER.fullmatch(text):
        raise HTTPException(400, "marker must be a bounded inert identifier")
    return text


@dataclass
class _State:
    source: str
    seed: int
    target_instance_id: str
    epoch: int = 0


def _reset_body(state: _State) -> dict[str, Any]:
    body = {
        "schema_version": RESET_SCHEMA_VERSION,
        "source": state.source,
        "target_instance_id": state.target_instance_id,
        "reset_epoch": state.epoch,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }
    return {**body, "evidence_hash": _hash_json(body)}


async def _read_payload(request: Request) -> dict[str, Any]:
    if request.method == "GET":
        return {str(key): str(value) for key, value in request.query_params.items()}
    try:
        body = await request.json()
    except Exception as exc:  # pragma: no cover - transport error branch
        raise HTTPException(400, "POST body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "POST body must be an object")
    return {str(key): value for key, value in body.items()}


def _classify(source: str, surface: str, payload: dict[str, Any]) -> tuple[bool, bool, bool, str, str]:
    """Return candidate, typed_available, typed_effect, atom, shape."""

    if source == "alpha":
        if surface == "policy":
            candidate = str(payload.get("policy_mode", "baseline")) == "alternate"
            return candidate, True, candidate, "policy_transition", "policy-transition"
        if surface == "decoy":
            candidate = str(payload.get("view_mode", "baseline")) == "alternate"
            return candidate, True, False, "shape_delta", "shape-decoy"
        if surface == "neutral":
            return False, True, False, "no_effect", "stable"
        if surface == "opaque":
            candidate = str(payload.get("mode", "control")) == "canary" and bool(_marker(payload.get("marker")))
            return candidate, False, False, "withheld_effect", "opaque-signal"
    elif source == "beta":
        if surface == "boundary":
            candidate = str(payload.get("phase", "baseline")) == "candidate"
            return candidate, True, candidate, "boundary_transition", "boundary-transition"
        if surface == "layout":
            candidate = str(payload.get("layout", "base")) == "variant"
            return candidate, True, False, "layout_delta", "layout-decoy"
        if surface == "steady":
            return False, True, False, "no_effect", "steady"
        if surface == "blind":
            candidate = str(payload.get("probe", "control")) == "canary" and bool(_marker(payload.get("marker")))
            return candidate, False, False, "withheld_effect", "blind-signal"
    raise HTTPException(404, "unknown PG-116 surface")


def create_target_app(source: str, seed: int) -> FastAPI:
    source = str(source).casefold()
    if source not in SOURCE_SURFACES:
        raise ValueError(f"unsupported PG-116 source: {source}")
    state = _State(source=source, seed=int(seed), target_instance_id=f"pg116-{source}-target-{int(seed)}")
    app = FastAPI(title=f"PG-116 {source} local training target", version="1.0")

    def response(surface: str, payload: dict[str, Any]) -> dict[str, Any]:
        candidate, typed_available, typed_effect, atom, shape = _classify(source, surface, payload)
        surface_slot = f"pg116-{source}-slot-{_hash_json(surface)[:16]}"
        evaluator = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "oracle_id": f"pg116-{source}-{surface}-typed-v1",
            "surface_slot": surface_slot,
            "typed_available": typed_available,
            "candidate_signal": candidate,
            "safety": {
                "external_network": False,
                "script_execution": False,
                "database_write": False,
                "navigation": False,
                "credentials_accessed": False,
                "state_mutated": False,
            },
        }
        if typed_available:
            evaluator["typed_effect"] = typed_effect
            evaluator["effect_atoms"] = [atom] if typed_effect else []
        response_projection = {
            "status_class": "2xx",
            "body_length_bucket": "1-255" if source == "alpha" else "256-4095",
            "shape_class": shape,
            "candidate_signal": candidate,
            "policy_header_changed": bool(typed_effect),
            "shape_changed": bool(candidate and not typed_effect),
            "noise_bucket": (state.seed + state.epoch * 5 + len(surface) * 3) % 9,
        }
        evidence_body = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "source": source,
            "target_instance_id": state.target_instance_id,
            "reset_epoch": state.epoch,
            "surface_slot": surface_slot,
            "candidate_signal": candidate,
            "typed_evaluator_available": typed_available,
            "evaluator": evaluator,
            "response_projection": response_projection,
            "safety": evaluator["safety"],
        }
        evidence = {**evidence_body, "evidence_hash": _hash_json(evidence_body)}
        return {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "implementation": f"pg116-{source}-target",
            "surface": surface,
            "surface_slot": surface_slot,
            "candidate_signal": candidate,
            "response_projection": response_projection,
            "evaluator": evaluator,
            "evidence": evidence,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "source": source,
            "implementation": f"pg116-{source}-target",
            "schema_version": SCHEMA_VERSION,
        }

    @app.post("/v4/reset")
    async def reset() -> dict[str, Any]:
        state.epoch += 1
        return {"status": "reset", "reset": _reset_body(state)}

    @app.get("/v4/probe/{surface}")
    async def get_probe(surface: str, request: Request) -> dict[str, Any]:
        if surface not in SOURCE_SURFACES[source]:
            raise HTTPException(404, "unknown PG-116 surface")
        return response(surface, await _read_payload(request))

    @app.post("/v4/probe/{surface}")
    async def post_probe(surface: str, request: Request) -> dict[str, Any]:
        if surface not in SOURCE_SURFACES[source]:
            raise HTTPException(404, "unknown PG-116 surface")
        return response(surface, await _read_payload(request))

    app.state.pg116_source = source
    app.state.pg116_target_state = state
    return app


__all__ = ["EVIDENCE_SCHEMA_VERSION", "RESET_SCHEMA_VERSION", "SCHEMA_VERSION", "SOURCE_SURFACES", "create_target_app"]

