"""PG-114 family-heldout local target with a negative decoy.

The target implements a small response-policy surface that is absent from the
PG-112/PG-113 replay matrices.  It accepts only inert abstract classes.  A
typed evaluator can attest a bounded policy transition, a decoy can change
the response shape without an effect, and an opaque surface exposes only an
anonymous signal.  No script, SQL, network, redirect, credential, or state
mutation is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request


SCHEMA_VERSION = "pg114-family-holdout-target-v1"
EVIDENCE_SCHEMA_VERSION = "pg114-family-holdout-evidence-v1"
RESET_SCHEMA_VERSION = "pg114-family-holdout-reset-v1"
_MARKER = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
_SURFACE_SEMANTICS = {
    "policy": "security_policy_transition",
    "decoy": "shape_only_change",
    "neutral": "no_effect",
    "opaque": "withheld_policy",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass
class _State:
    seed: int
    target_instance_id: str
    epoch: int = 0


def _new_state() -> _State:
    seed = int(os.environ.get("PG114_SEED", "11401"))
    return _State(seed=seed, target_instance_id=os.environ.get("PG114_TARGET_INSTANCE", f"pg114-target-{seed}"))


state = _new_state()
app = FastAPI(title="PG-114 Family Holdout Target", version="1.0")


def _marker(value: Any) -> str:
    marker = str(value or "")
    if not _MARKER.fullmatch(marker):
        raise HTTPException(400, "marker must be a bounded inert identifier")
    return marker


async def _payload(request: Request) -> dict[str, Any]:
    if request.method == "GET":
        return {str(key): str(value) for key, value in request.query_params.items()}
    try:
        body = await request.json()
    except Exception as exc:  # pragma: no cover - transport error branch
        raise HTTPException(400, "POST body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "POST body must be an object")
    return {str(key): value for key, value in body.items()}


def _slot(kind: str) -> str:
    return f"slot-{_hash_json({'implementation': 'pg114-family-holdout', 'surface': kind})[:16]}"


def _reset() -> dict[str, Any]:
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


def _classify(kind: str, payload: dict[str, Any]) -> tuple[bool, bool, bool, str, str]:
    """Return candidate_signal, typed_available, typed_effect, atom, shape."""

    if kind == "policy":
        candidate = str(payload.get("policy_class", "baseline")) == "alternate"
        return candidate, True, candidate, "security_policy_transition", "header-policy"
    if kind == "decoy":
        candidate = str(payload.get("shape_class", "baseline")) == "alternate"
        return candidate, True, False, "shape_only_change", "shape-decoy"
    if kind == "neutral":
        return False, True, False, "no_effect", "stable"
    if kind == "opaque":
        candidate = str(payload.get("mode", "control")) == "canary" and bool(_marker(payload.get("marker")))
        return candidate, False, False, "withheld_policy", "opaque-signal"
    raise HTTPException(404, "unknown PG-114 surface")


def _response(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    candidate, typed_available, typed_effect, atom, shape = _classify(kind, payload)
    slot = _slot(kind)
    evaluator = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "oracle_id": f"pg114-independent-{kind}-v1",
        "surface_slot": slot,
        "typed_available": typed_available,
        "candidate_signal": candidate,
        "typed_effect": typed_effect,
        "effect_atoms": [atom] if typed_effect else [],
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "navigation": False, "credentials_accessed": False, "state_mutated": False},
    }
    if not typed_available:
        evaluator.pop("typed_effect")
        evaluator.pop("effect_atoms")
    response_projection = {"status_class": "2xx", "shape_class": shape, "policy_header_changed": bool(typed_effect), "shape_changed": bool(candidate and not typed_effect), "noise_bucket": (state.seed + state.epoch * 3 + len(kind)) % 7}
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "target_instance_id": state.target_instance_id,
        "reset_epoch": state.epoch,
        "surface_slot": slot,
        "candidate_signal": candidate,
        "typed_evaluator_available": typed_available,
        "evaluator": evaluator,
        "response_projection": response_projection,
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False},
    }
    return {"schema_version": SCHEMA_VERSION, "implementation": "pg114-family-holdout-target", "surface_slot": slot, "candidate_signal": candidate, "response_projection": response_projection, "evaluator": evaluator, "evidence": {**evidence, "evidence_hash": _hash_json(evidence)}}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "implementation": "pg114-family-holdout-target", "schema_version": SCHEMA_VERSION}


@app.post("/v2/reset")
async def reset() -> dict[str, Any]:
    state.epoch += 1
    return {"status": "reset", "reset": _reset()}


@app.get("/v2/inspect/{kind}")
async def inspect_get(kind: str, request: Request) -> dict[str, Any]:
    return _response(kind, await _payload(request))


@app.post("/v2/inspect/{kind}")
async def inspect_post(kind: str, request: Request) -> dict[str, Any]:
    return _response(kind, await _payload(request))


__all__ = ["SCHEMA_VERSION", "app", "state"]
