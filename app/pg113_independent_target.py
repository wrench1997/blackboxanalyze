"""PG-113 independent local target implementation.

This module is deliberately independent from ``app.main`` and the PG-112
oracle code.  It is a tiny, read-only HTTP target with its own typed-evidence
contract.  The accepted inputs are abstract canary classes, never executable
markup, SQL, JavaScript, credentials, or URLs.  The reset endpoint creates a
new hidden evaluator epoch; action responses expose only bounded projections
and a target-side SHA-256 commitment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request


SCHEMA_VERSION = "pg113-independent-target-v1"
EVIDENCE_SCHEMA_VERSION = "pg113-independent-typed-evidence-v1"
RESET_SCHEMA_VERSION = "pg113-independent-reset-v1"
_SAFE_MARKER = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
_KNOWN_SURFACES = {"markup": "render_context", "parser": "interpreter_context", "boundary": "state_transition"}
_ALL_SURFACES = {**_KNOWN_SURFACES, "opaque": "withheld_context"}


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
    seed = int(os.environ.get("PG113_SEED", "113"))
    target = os.environ.get("PG113_TARGET_INSTANCE", f"pg113-target-{seed}")
    return _State(seed=seed, target_instance_id=target)


state = _new_state()
app = FastAPI(title="PG-113 Independent Local Target", version="1.0")


def _safe_marker(value: Any) -> str:
    marker = str(value or "")
    if not _SAFE_MARKER.fullmatch(marker):
        raise HTTPException(400, "marker must be a bounded inert identifier")
    return marker


async def _payload(request: Request) -> dict[str, Any]:
    if request.method == "GET":
        return {str(key): str(value) for key, value in request.query_params.items()}
    try:
        value = await request.json()
    except Exception as exc:  # pragma: no cover - transport-specific branch
        raise HTTPException(400, "POST body must be JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "POST body must be an object")
    return {str(key): value for key, value in value.items()}


def _reset_evidence() -> dict[str, Any]:
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


def _surface_slot(kind: str) -> str:
    return f"slot-{_hash_json({'implementation': 'pg113-independent', 'surface': kind})[:16]}"


def _classify(kind: str, payload: dict[str, Any]) -> tuple[bool, bool, str]:
    """Return (candidate_signal, typed_effect, abstract atom)."""

    if kind == "markup":
        marker = _safe_marker(payload.get("marker"))
        candidate = str(payload.get("mode", "control")) == "canary" and bool(marker)
        return candidate, candidate, "render_context"
    if kind == "parser":
        shape = str(payload.get("shape", "plain"))
        candidate = shape == "operator"
        return candidate, candidate, "interpreter_context"
    if kind == "boundary":
        transition = str(payload.get("transition", "normal"))
        candidate = transition == "candidate"
        return candidate, candidate, "state_transition"
    if kind == "opaque":
        # The anonymous signal is intentional.  No evaluator is made
        # available for this surface, so the collector must abstain.
        marker = _safe_marker(payload.get("marker"))
        return str(payload.get("mode", "control")) == "canary" and bool(marker), False, "withheld_context"
    raise HTTPException(404, "unknown PG-113 surface")


def _response(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in _ALL_SURFACES:
        raise HTTPException(404, "unknown PG-113 surface")
    candidate_signal, typed_effect, atom = _classify(kind, payload)
    typed_available = kind in _KNOWN_SURFACES
    # This is deterministic nuisance variation, not a label: it changes with
    # the hidden epoch and seed so a surface string cannot be memorized.
    noise_bucket = (state.seed * 17 + state.epoch * 7 + len(kind)) % 5
    evaluator = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "oracle_id": f"pg113-independent-{kind}-v1",
        "surface_slot": _surface_slot(kind),
        "typed_available": typed_available,
        "candidate_signal": candidate_signal,
        "typed_effect": typed_effect,
        "effect_atoms": [atom] if typed_effect else [],
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "navigation": False,
            "credentials_accessed": False,
            "state_mutated": False,
        },
    }
    # Withheld surfaces deliberately expose only the anonymous signal and a
    # contract marker; the bridge must not infer a positive oracle from it.
    if not typed_available:
        evaluator = {key: value for key, value in evaluator.items() if key in {"schema_version", "surface_slot", "typed_available", "candidate_signal", "safety"}}
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "target_instance_id": state.target_instance_id,
        "reset_epoch": state.epoch,
        "surface_slot": _surface_slot(kind),
        "candidate_signal": candidate_signal,
        "typed_evaluator_available": typed_available,
        "evaluator": evaluator,
        "response_projection": {"status_class": "2xx", "shape_class": "object", "noise_bucket": noise_bucket},
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": "pg113-independent-target",
        "surface_slot": _surface_slot(kind),
        "response_projection": evidence["response_projection"],
        "candidate_signal": candidate_signal,
        "evaluator": evaluator,
        "evidence": {**evidence, "evidence_hash": _hash_json(evidence)},
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION, "implementation": "pg113-independent-target"}


@app.post("/v1/reset")
async def reset() -> dict[str, Any]:
    state.epoch += 1
    return {"status": "reset", "reset": _reset_evidence()}


@app.get("/v1/surface/{kind}")
async def surface_get(kind: str, request: Request) -> dict[str, Any]:
    return _response(kind, await _payload(request))


@app.post("/v1/surface/{kind}")
async def surface_post(kind: str, request: Request) -> dict[str, Any]:
    return _response(kind, await _payload(request))


__all__ = ["SCHEMA_VERSION", "app", "state"]
