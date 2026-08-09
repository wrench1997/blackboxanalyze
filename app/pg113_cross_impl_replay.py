"""PG-113 replay bridge for the independent local target.

The bridge shares only the generic Trace/Rule-IR and Python BSP contracts with
PG-112.  Transport paths, request shapes, target-side typed evidence, reset
protocol, and surface implementation are independent.  Raw request values
and response bytes never leave a single action's memory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .bsp_v3_research_core import BspV3Config, BspV3State
from .generic_belief_state import GenericBeliefState
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg113-independent-cross-implementation-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg113-independent-replay-evidence-v1"
ORACLE_CONTRACT = "pg113-independent-typed-oracle-v1"
ALLOWED_METHODS = frozenset({"GET", "POST"})
_HASH_RE = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Surface:
    kind: str
    surface_slot: str
    typed_available: bool
    semantic_ref: str


SURFACES: tuple[Surface, ...] = (
    Surface("markup", "slot-2cdfe607d83ffe28", True, "render_context"),
    Surface("parser", "slot-be6bb8bf9db6539a", True, "interpreter_context"),
    Surface("boundary", "slot-d0853b34126f9fb0", True, "state_transition"),
    Surface("opaque", "slot-3136f97e68d03cf3", False, "withheld_context"),
)


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _verify_hash(value: Mapping[str, Any], *, key: str = "evidence_hash") -> str:
    declared = str(value.get(key, ""))
    if len(declared) != 64 or any(char not in _HASH_RE for char in declared):
        raise ValueError("independent target evidence hash is invalid")
    body = dict(value)
    body.pop(key, None)
    if sha256_json(body) != declared:
        raise ValueError("independent target evidence hash mismatch")
    return declared


def _payload(surface: Surface, *, candidate: bool, marker: str) -> dict[str, Any]:
    if surface.kind in {"markup", "opaque"}:
        return {"mode": "canary" if candidate else "control", "marker": marker}
    if surface.kind == "parser":
        return {"shape": "operator" if candidate else "plain"}
    if surface.kind == "boundary":
        return {"transition": "candidate" if candidate else "normal"}
    raise ValueError("unknown PG-113 surface")


def _status_class(code: int) -> str:
    return f"{code // 100}xx" if 100 <= int(code) <= 599 else "other"


def _bsp_observation(state: BspV3State, evidence_hash: str) -> dict[str, Any]:
    import numpy as np

    values = [int(evidence_hash[index : index + 2], 16) / 255.0 for index in (0, 2, 4, 6)]
    output = state.forward(np.asarray([values], dtype=np.float64), np.asarray([[0.7, 0.3]], dtype=np.float64))
    return {
        "selected_leaf_ids": [int(value) for value in output.selected_leaf_ids.reshape(-1)],
        "leaf_mass_error": float(abs(float(output.leaf_mass_sum.sum()) - 1.0)),
        "topology_version": state.topology_version,
        "state_sha256": state.state_sha256(),
    }


def _oracle_projection(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool, str]:
    evidence = response.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("independent target omitted evidence")
    evidence_hash = _verify_hash(evidence)
    evaluator = evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("independent target omitted evaluator projection")
    typed_available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate_signal = bool(evaluator.get("candidate_signal"))
    typed_effect = bool(evaluator.get("typed_effect")) if typed_available else False
    positive = bool(typed_effect and typed_available)
    atoms = [surface.semantic_ref] if positive else ["candidate_signal_observed"] if candidate_signal else ["no_effect"]
    projection = {
        "modality": "independent_typed_differential" if typed_available else "independent_untyped_signal",
        "positive": positive,
        "positive_authority": positive,
        "candidate_signal": candidate_signal,
        "observed_atoms": atoms,
        "oracle_contract_sha256": _hash_text(ORACLE_CONTRACT),
        "source_evidence_sha256": evidence_hash,
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_touched": False,
            "real_sleep_performed": False,
            "navigation": False,
            "credentials_accessed": False,
        },
    }
    return projection, positive, bool(typed_available), evidence_hash


async def _action(
    client: httpx.AsyncClient,
    surface: Surface,
    *,
    target_instance_id: str,
    method: str,
    role: str,
    marker: str,
    sequence: int,
    state: BspV3State,
) -> dict[str, Any]:
    method = str(method).upper()
    if method not in ALLOWED_METHODS:
        raise ValueError("PG-113 permits only GET and POST")
    health = await client.get("/health")
    if health.status_code != 200:
        raise RuntimeError("independent target health failed")
    reset_response = await client.post("/v1/reset")
    if reset_response.status_code != 200:
        raise RuntimeError("independent target reset failed")
    reset_body = reset_response.json()
    reset = reset_body.get("reset")
    if not isinstance(reset, Mapping):
        raise ValueError("independent target reset omitted bounded evidence")
    reset_hash = _verify_hash(reset)
    if not bool(reset.get("evaluator_state_hidden")) or not bool(reset.get("state_change_allowed") is False):
        raise ValueError("independent target reset contract failed")
    candidate = role == "candidate"
    payload = _payload(surface, candidate=candidate, marker=marker)
    route = f"/v1/surface/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"independent target returned {response.status_code}")
    body = response.json()
    oracle, positive, authority, source_hash = _oracle_projection(surface, body)
    response_projection = dict(body.get("response_projection") or {})
    response_projection.update(
        {
            "status_class": _status_class(response.status_code),
            "shape_class": "object",
            "response_projection_sha256": sha256_json(dict(body.get("response_projection") or {})),
            "candidate_signal": bool(body.get("candidate_signal")),
        }
    )
    request_hash = sha256_json({"surface_slot": surface.surface_slot, "method": method, "role": role, "payload_shape": sorted(payload.keys()), "marker_sha256": _hash_text(marker)})
    evidence_response_projection = dict(response_projection)
    evidence_without_hash = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "target_instance_id": target_instance_id,
        "surface_slot": surface.surface_slot,
        "method": method,
        "role": role,
        "request_sha256": request_hash,
        "target_evidence_sha256": source_hash,
        "reset_evidence_sha256": reset_hash,
        "response_projection": evidence_response_projection,
        "oracle_projection": oracle,
        "fresh_reset": dict(reset),
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False},
    }
    evidence = {**evidence_without_hash, "evidence_hash": sha256_json(evidence_without_hash)}
    bsp = _bsp_observation(state, evidence["evidence_hash"])
    response_projection["bsp_core_projection"] = bsp
    action_manifest = {
        "method": method,
        "route_template_id": f"pg113-route-{_hash_text(route)[:16]}",
        "placement": placement,
        "encoding_chain": ["identity"],
        "probe_ref": f"pg113-abstract-probe-{_hash_text(surface.surface_slot)[:12]}",
        "probe_sha256": request_hash,
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
        **({"form_field_names": ["abstract_probe"]} if method == "POST" else {}),
    }
    return {
        "sample_id": f"pg113-{target_instance_id[-5:]}-{surface.kind}-{method.casefold()}-{role}",
        "target_instance_id": target_instance_id,
        "surface_slot": surface.surface_slot,
        "method": method,
        "role": role,
        "positive": positive,
        "positive_authority": authority,
        "typed_oracle_called": authority or str(oracle["modality"]) == "independent_typed_differential",
        "action_manifest": action_manifest,
        "baseline_projection": {"status_class": _status_class(health.status_code), "body_length_bucket": "1-255", "body_sha256": _hash_text("pg113-health")},
        "response_projection": response_projection,
        "oracle_projection": oracle,
        "fresh_reset": {**dict(reset), "reset_evidence_sha256": reset_hash, "target_instance_id": target_instance_id},
        "evidence": evidence,
    }


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    if bool(record.get("positive_authority")):
        return {"effect": 0.84, "input_only": 0.04, "no_effect": 0.02, "unknown": 0.10}
    if bool(record.get("positive")):
        return {"effect": 0.08, "input_only": 0.20, "no_effect": 0.07, "unknown": 0.65}
    return {"effect": 0.03, "input_only": 0.08, "no_effect": 0.80, "unknown": 0.09}


def _step(record: Mapping[str, Any], *, episode_id: str, index: int, parent: str | None, decision: str, next_action: str, before: Mapping[str, float], after: Mapping[str, float], pair_id: str | None) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if pair_id is not None:
        oracle["negative_control_pair_id"] = pair_id
    body = {
        "action_manifest": record["action_manifest"],
        "baseline_projection": record["baseline_projection"],
        "response_projection": record["response_projection"],
        "oracle_projection": oracle,
        "belief_before": dict(before),
        "belief_after": dict(after),
        "decision": decision,
        "next_action": next_action,
    }
    return validate_trace_step(
        {
            "episode_id": episode_id,
            "step_id": f"{episode_id}-s{index:02d}",
            "parent_step_id": parent,
            "sampling_seed": int(record["target_instance_id"].rsplit("-", 1)[-1]),
            "target_instance_id": record["target_instance_id"],
            "hypothesis": "unknown_surface",
            **body,
            "fresh_reset": record["fresh_reset"],
            "evidence_sha256": record["evidence"]["evidence_hash"],
            "dataset_stage": "pg113_cross_implementation_evaluation_only",
            "online_weight_update": False,
            "long_term_memory_write": False,
            "echo": {"sha256": sha256_json(body)},
        }
    )


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, target_seed: int) -> dict[str, Any]:
    target_instance_id = f"pg113-target-{target_seed}"
    episode_id = f"pg113-{target_seed}-{surface.surface_slot}"
    state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=target_seed)
    parameter_before = state.parameter_sha256()
    marker = f"r113{target_seed}{surface.kind[:3]}"
    records: list[dict[str, Any]] = []
    for sequence, (role, method) in enumerate((("control", "GET"), ("candidate", "GET"), ("control", "POST"), ("candidate", "POST")), start=1):
        records.append(await _action(client, surface, target_instance_id=target_instance_id, method=method, role=role, marker=marker, sequence=sequence, state=state))
    controls = [record for record in records if record["role"] == "control"]
    candidates = [record for record in records if record["role"] == "candidate"]
    known_positive = surface.typed_available and all(bool(record["positive_authority"]) for record in candidates) and all(not bool(record["positive"]) for record in controls)
    final_decision = "confirmed_positive" if known_positive else "abstain"
    belief = GenericBeliefState()
    steps: list[dict[str, Any]] = []
    parent: str | None = None
    for index, record in enumerate(records, start=1):
        before = dict(belief.posterior)
        update = belief.observe(f"{record['sample_id']}-belief", _likelihood(record), evidence_hash=record["evidence"]["evidence_hash"])
        after = dict(update["posterior"])
        if record["role"] == "control":
            decision, next_action = "confirmed_negative", "probe_candidate_same_method"
        elif record["method"] == "GET":
            decision, next_action = ("candidate", "replay_other_method")
        else:
            decision, next_action = final_decision, "stop_episode" if known_positive else "abstain_unknown_surface"
        pair_id = next((control["sample_id"] for control in controls if control["method"] == record["method"]), None) if record["role"] == "candidate" else None
        current = _step(record, episode_id=episode_id, index=index, parent=parent, decision=decision, next_action=next_action, before=before, after=after, pair_id=pair_id)
        steps.append(current)
        parent = current["step_id"]
    return {
        "episode_id": episode_id,
        "target_instance_id": target_instance_id,
        "surface_slot": surface.surface_slot,
        "surface_kind": surface.kind,
        "oracle_available": surface.typed_available,
        "steps": steps,
        "evidence_records": [record["evidence"] for record in records],
        "episode_report": evaluate_episode(steps),
        "final_decision": final_decision,
        "candidate_pair_positive": known_positive,
        "negative_control_pair_clear": all(not bool(record["positive"]) for record in controls),
        "bsp": {"parameter_sha256_before": parameter_before, "parameter_sha256_after": state.parameter_sha256(), "parameter_unchanged": parameter_before == state.parameter_sha256(), "topology_version": state.topology_version, "state_sha256": state.state_sha256()},
    }


async def collect_target(client: httpx.AsyncClient, *, target_seed: int) -> dict[str, Any]:
    health = await client.get("/health")
    if health.status_code != 200 or health.json().get("implementation") != "pg113-independent-target":
        raise RuntimeError("PG-113 cross-implementation health identity failed")
    episodes = [await collect_episode(client, surface, target_seed=target_seed) for surface in SURFACES]
    return {"target_seed": target_seed, "target_instance_id": f"pg113-target-{target_seed}", "target_implementation": health.json().get("implementation"), "target_schema_version": health.json().get("schema_version"), "episodes": episodes}


__all__ = ["SCHEMA_VERSION", "SURFACES", "collect_episode", "collect_target"]
