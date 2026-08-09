"""PG-133 local history-latch target.

The target is a deliberately small, non-executing workflow fixture.  It
records only an episode-local sequence of safe ``control``/``candidate``
observations.  The current response projection is identical for the two
history variants; a typed evaluator-only workflow oracle (kept outside model
input) supplies the next safe action.  This lets the replay bridge test
whether a model actually uses a Rule-IR prefix instead of memorising the
current surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request


SCHEMA_VERSION = "pg133-history-latch-target-v1"
EVIDENCE_SCHEMA_VERSION = "pg133-history-latch-evidence-v1"
RESET_SCHEMA_VERSION = "pg133-history-latch-reset-v1"
SURFACES = ("latch", "decoy", "steady", "blind")
_ROLES = frozenset({"control", "candidate"})
_PHASES = frozenset({"shadow", "confirm"})
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
    history: list[dict[str, str]] = field(default_factory=list)


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
    state.history.clear()
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


def _workflow_action(surface: str, role: str, *, control_seen: bool, typed_available: bool) -> str:
    """Return the evaluator-only safe action from observable workflow facts.

    A completed control probe is a generic matched-negative prerequisite for
    every typed surface.  Keeping this rule surface-independent is important:
    otherwise two surfaces with the same visible response/Rule-IR tokens would
    acquire different labels, creating an impossible training example.
    """

    if not typed_available:
        return "abstain_unknown_oracle"
    if role == "control":
        return "repeat_matched_negative_pair"
    if control_seen:
        return "repeat_matched_negative_pair"
    return "probe_candidate_other_method"


def create_app(seed: int, *, decoy_strength: int = 1) -> FastAPI:
    strength = max(0, min(int(decoy_strength), 2))
    state = _State(seed=int(seed), strength=strength, target_instance_id=f"pg133-latch-target-{int(seed)}-d{strength}")
    app = FastAPI(title="PG-133 History Latch Target", version="1.0")

    def observe(surface: str, payload: dict[str, Any], method: str) -> dict[str, Any]:
        if surface not in SURFACES:
            raise HTTPException(404, "unknown PG-133 surface")
        method = str(method).upper()
        if method not in {"GET", "POST"}:
            raise HTTPException(400, "method must be GET or POST")
        role = str(payload.get("role", "control"))
        phase = str(payload.get("probe_phase", "shadow"))
        if role not in _ROLES or phase not in _PHASES:
            raise HTTPException(400, "role or probe_phase is invalid")
        if role == "candidate":
            _marker(payload.get("marker", "pg133-safe-marker"))
        prior = list(state.history)
        control_seen = any(item.get("role") == "control" for item in prior)
        methods_seen = sorted({item.get("method", "") for item in prior} | {method})
        candidate = role == "candidate"
        if surface == "latch":
            typed_available, typed_effect = True, False
        elif surface in {"decoy", "steady"}:
            typed_available, typed_effect = True, False
        else:
            typed_available, typed_effect = False, False
        workflow_action = _workflow_action(surface, role, control_seen=control_seen, typed_available=typed_available)
        history_stage = "control_complete" if control_seen else "candidate_first"
        if role == "control":
            history_stage = "control_probe"
        slot = f"pg133-latch-slot-{_hash_json(surface)[:16]}"
        safety = {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "navigation": False,
            "credentials_accessed": False,
            "state_mutated": False,
        }
        # The history stage and workflow action are evaluator-only authority;
        # they are deliberately absent from response_projection.
        evaluator = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "oracle_id": f"pg133-latch-{surface}-workflow-v1",
            "surface_slot": slot,
            "typed_available": typed_available,
            "candidate_signal": candidate,
            "typed_effect": typed_effect,
            "effect_atoms": [],
            "history_stage": history_stage,
            "workflow_action": workflow_action,
            "methods_seen": methods_seen,
            "safety": safety,
        }
        response_projection = {
            "status_class": "2xx",
            "body_length_bucket": "256-4095",
            "shape_class": "decision-v8",
            "candidate_signal": candidate,
            "policy_header_changed": False,
            "shape_changed": False,
            "location_changed": False,
            "metadata_changed": False,
            "authorization_changed": False,
            "scope_changed": False,
            "visibility_changed": False,
            "transition_delta": "none",
            # Deliberately independent of reset epoch/history: the current
            # observation is the same across the two counterfactual prefixes.
            "noise_bucket": (state.seed * 29 + strength * 11 + len(surface) + len(method)) % 19,
        }
        evidence_body = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "target_instance_id": state.target_instance_id,
            "surface": surface,
            "method": method,
            "role": role,
            "probe_phase": phase,
            "decoy_strength": strength,
            "reset_epoch": state.epoch,
            "candidate_signal": candidate,
            "typed_evaluator_available": typed_available,
            "evaluator": evaluator,
            "response_projection": response_projection,
            "safety": safety,
        }
        evidence = {**evidence_body, "evidence_hash": _hash_json(evidence_body)}
        state.history.append({"method": method, "role": role, "surface": surface})
        return {
            "schema_version": SCHEMA_VERSION,
            "implementation": "pg133-history-latch-target",
            "surface": surface,
            "surface_slot": slot,
            "candidate_signal": candidate,
            "response_projection": response_projection,
            "evaluator": evaluator,
            "evidence": evidence,
        }

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "implementation": "pg133-history-latch-target", "schema_version": SCHEMA_VERSION}

    @app.post("/v13/reset")
    async def reset() -> dict[str, Any]:
        state.epoch += 1
        return {"status": "reset", "reset": _reset(state)}

    @app.get("/v13/observe/{surface}")
    async def check_get(surface: str, request: Request) -> dict[str, Any]:
        return observe(surface, await _payload(request), "GET")

    @app.post("/v13/observe/{surface}")
    async def check_post(surface: str, request: Request) -> dict[str, Any]:
        return observe(surface, await _payload(request), "POST")

    app.state.pg133_seed = int(seed)
    app.state.pg133_decoy_strength = strength
    return app


__all__ = ["EVIDENCE_SCHEMA_VERSION", "RESET_SCHEMA_VERSION", "SCHEMA_VERSION", "SURFACES", "create_app"]
