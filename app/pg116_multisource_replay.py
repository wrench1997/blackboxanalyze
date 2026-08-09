"""Local GET/POST replay collector for PG-116.

It runs two independently named FastAPI target profiles through an in-process
loopback ASGI transport.  Every action performs a fresh reset, records target
and bridge evidence hashes, and keeps typed-oracle fields outside the
model-facing projection.  The resulting rows are training candidates for the
small PG-115 decoder; they are never promoted to long-term memory here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .bsp_v3_research_core import BspV3Config, BspV3State
from .generic_belief_state import GenericBeliefState
from .pg115_small_rule_ir_decoder import canonical_model_input
from .pg116_multisource_training_target import SOURCE_SURFACES, create_target_app
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg116-multisource-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg116-multisource-bridge-evidence-v1"
ORACLE_CONTRACT = "pg116-multisource-typed-oracle-v1"
ALLOWED_METHODS = frozenset({"GET", "POST"})


@dataclass(frozen=True)
class Surface:
    source: str
    kind: str
    expected: str
    typed_available: bool


SURFACES: tuple[Surface, ...] = tuple(
    Surface(source, kind, "positive" if index == 0 else "negative" if index < 3 else "abstain", index < 3)
    for source, kinds in SOURCE_SURFACES.items()
    for index, kind in enumerate(kinds)
)


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _verify_hash(value: Mapping[str, Any]) -> str:
    declared = str(value.get("evidence_hash", ""))
    if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise ValueError("PG-116 evidence hash is invalid")
    body = dict(value)
    body.pop("evidence_hash", None)
    if sha256_json(body) != declared:
        raise ValueError("PG-116 evidence hash mismatch")
    return declared


def _payload(surface: Surface, candidate: bool, marker: str) -> dict[str, Any]:
    if surface.source == "alpha":
        if surface.kind == "policy":
            return {"policy_mode": "alternate" if candidate else "baseline"}
        if surface.kind == "decoy":
            return {"view_mode": "alternate" if candidate else "baseline"}
        if surface.kind == "neutral":
            return {"mode": "baseline"}
        return {"mode": "canary" if candidate else "control", "marker": marker}
    if surface.kind == "boundary":
        return {"phase": "candidate" if candidate else "baseline"}
    if surface.kind == "layout":
        return {"layout": "variant" if candidate else "base"}
    if surface.kind == "steady":
        return {"state": "baseline"}
    return {"probe": "canary" if candidate else "control", "marker": marker}


def _status_class(code: int) -> str:
    return f"{int(code) // 100}xx" if 100 <= int(code) <= 599 else "other"


def _bsp(state: BspV3State, evidence_hash: str) -> dict[str, Any]:
    import numpy as np

    values = [int(evidence_hash[index : index + 2], 16) / 255.0 for index in (0, 2, 4, 6)]
    output = state.forward(np.asarray([values], dtype=np.float64), np.asarray([[0.5, 0.5]], dtype=np.float64))
    return {
        "selected_leaf_ids": [int(value) for value in output.selected_leaf_ids.reshape(-1)],
        "leaf_mass_error": float(abs(float(output.leaf_mass_sum.sum()) - 1.0)),
        "topology_version": state.topology_version,
        "state_sha256": state.state_sha256(),
    }


def _oracle(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool, str]:
    evidence = response.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("PG-116 target omitted evidence")
    source_hash = _verify_hash(evidence)
    evaluator = evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("PG-116 target omitted evaluator projection")
    available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate_signal = bool(evaluator.get("candidate_signal"))
    typed_effect = bool(evaluator.get("typed_effect")) if available else False
    positive = bool(available and typed_effect)
    atoms = ["effect_present"] if positive else ["candidate_signal_observed"] if candidate_signal else ["no_effect"]
    projection = {
        "modality": "multisource_typed_differential" if available else "multisource_untyped_signal",
        "positive": positive,
        "positive_authority": positive,
        "candidate_signal": candidate_signal,
        "observed_atoms": atoms,
        "oracle_contract_sha256": _hash_text(ORACLE_CONTRACT),
        "source_evidence_sha256": source_hash,
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_touched": False,
            "real_sleep_performed": False,
            "navigation": False,
            "credentials_accessed": False,
        },
    }
    return projection, positive, available, source_hash


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    if bool(record.get("positive_authority")):
        return {"effect": 0.84, "input_only": 0.04, "no_effect": 0.03, "unknown": 0.09}
    if bool(record.get("candidate_signal")):
        return {"effect": 0.10, "input_only": 0.25, "no_effect": 0.08, "unknown": 0.57}
    return {"effect": 0.03, "input_only": 0.07, "no_effect": 0.80, "unknown": 0.10}


async def _action(
    client: httpx.AsyncClient,
    surface: Surface,
    *,
    target_seed: int,
    method: str,
    role: str,
    marker: str,
    state: BspV3State,
) -> dict[str, Any]:
    method = str(method).upper()
    if method not in ALLOWED_METHODS:
        raise ValueError("PG-116 permits only GET and POST")
    health = await client.get("/health")
    expected_impl = f"pg116-{surface.source}-target"
    if health.status_code != 200 or health.json().get("implementation") != expected_impl:
        raise RuntimeError("PG-116 target identity failed")
    reset_response = await client.post("/v4/reset")
    if reset_response.status_code != 200:
        raise RuntimeError("PG-116 fresh reset failed")
    reset = reset_response.json().get("reset")
    if not isinstance(reset, Mapping):
        raise ValueError("PG-116 reset omitted evidence")
    reset_hash = _verify_hash(reset)
    if not (bool(reset.get("fresh_target")) and bool(reset.get("completed")) and bool(reset.get("evaluator_state_hidden")) and reset.get("state_change_allowed") is False):
        raise ValueError("PG-116 reset contract failed")
    candidate = role == "candidate"
    payload = _payload(surface, candidate, marker)
    route = f"/v4/probe/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"PG-116 target returned {response.status_code}")
    body = response.json()
    oracle, positive, authority, target_hash = _oracle(surface, body)
    target_projection = dict(body.get("response_projection") or {})
    response_projection = {
        **target_projection,
        "status_class": _status_class(response.status_code),
        "shape_class": str(target_projection.get("shape_class", "object")),
        "response_projection_sha256": sha256_json(target_projection),
        "candidate_signal": bool(body.get("candidate_signal")),
    }
    request_hash = sha256_json({
        "source": surface.source,
        "surface": surface.kind,
        "method": method,
        "role": role,
        "payload_keys": sorted(payload.keys()),
        "marker_sha256": _hash_text(marker),
    })
    evidence_body = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "target_instance_id": f"pg116-{surface.source}-target-{target_seed}",
        "surface": surface.kind,
        "source": surface.source,
        "method": method,
        "role": role,
        "request_sha256": request_hash,
        "target_evidence_sha256": target_hash,
        "reset_evidence_sha256": reset_hash,
        "response_projection": dict(response_projection),
        "oracle_projection": dict(oracle),
        "fresh_reset": dict(reset),
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False},
    }
    bridge_evidence = {**evidence_body, "evidence_hash": sha256_json(evidence_body)}
    response_projection["bsp_core_projection"] = _bsp(state, bridge_evidence["evidence_hash"])
    action_manifest = {
        "method": method,
        "route_template_id": f"pg116-{surface.source}-route-{_hash_text(route)[:16]}",
        "placement": placement,
        "encoding_chain": ["identity"],
        "probe_ref": f"pg116-abstract-probe-{_hash_text(surface.source + surface.kind)[:12]}",
        "probe_sha256": request_hash,
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
    }
    if method == "POST":
        action_manifest["form_field_names"] = ["abstract_probe"]
    model_input = canonical_model_input({
        "action_manifest": action_manifest,
        "baseline_projection": {"status_class": "2xx", "body_length_bucket": "1-255" if surface.source == "alpha" else "256-4095"},
        "response_projection": response_projection,
        "belief_before": {},
    })
    return {
        "source": surface.source,
        "surface": surface.kind,
        "target_seed": target_seed,
        "target_instance_id": f"pg116-{surface.source}-target-{target_seed}",
        "method": method,
        "role": role,
        "candidate_signal": bool(oracle["candidate_signal"]),
        "positive": positive,
        "positive_authority": authority,
        "action_manifest": action_manifest,
        "baseline_projection": model_input["baseline_projection"],
        "response_projection": response_projection,
        "model_input": model_input,
        "oracle_projection": oracle,
        "fresh_reset": {**dict(reset), "reset_evidence_sha256": reset_hash},
        "bridge_evidence": bridge_evidence,
    }


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
    return validate_trace_step({
        "episode_id": episode_id,
        "step_id": f"{episode_id}-s{index:02d}",
        "parent_step_id": parent,
        "sampling_seed": int(record["target_seed"]),
        "target_instance_id": record["target_instance_id"],
        "hypothesis": "unknown_surface",
        **body,
        "fresh_reset": record["fresh_reset"],
        "evidence_sha256": record["bridge_evidence"]["evidence_hash"],
        "dataset_stage": "pg116_multisource_training_candidate",
        "online_weight_update": False,
        "long_term_memory_write": False,
        "echo": {"sha256": sha256_json(body)},
    })


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, target_seed: int) -> dict[str, Any]:
    episode_id = f"pg116-{surface.source}-{target_seed}-{surface.kind}"
    bsp_state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=target_seed)
    parameter_before = bsp_state.parameter_sha256()
    marker = f"pg116{target_seed}{surface.source}{surface.kind[:3]}"
    records = [
        await _action(client, surface, target_seed=target_seed, method=method, role=role, marker=marker, state=bsp_state)
        for role, method in (("control", "GET"), ("candidate", "GET"), ("control", "POST"), ("candidate", "POST"))
    ]
    controls = [record for record in records if record["role"] == "control"]
    candidates = [record for record in records if record["role"] == "candidate"]
    if surface.expected == "positive":
        final_decision = "confirmed_positive" if all(record["positive_authority"] for record in candidates) and all(not record["positive"] for record in controls) else "abstain"
    elif surface.expected == "negative":
        final_decision = "confirmed_negative" if all(not record["positive"] for record in records) else "abstain"
    else:
        final_decision = "abstain"
    belief = GenericBeliefState()
    steps: list[dict[str, Any]] = []
    parent: str | None = None
    for index, record in enumerate(records, start=1):
        before = dict(belief.posterior)
        after = dict(belief.observe(f"{episode_id}-belief-{index}", _likelihood(record), evidence_hash=record["bridge_evidence"]["evidence_hash"])["posterior"])
        if record["role"] == "control":
            decision, next_action = "confirmed_negative", "probe_candidate_same_method"
        elif record["method"] == "GET":
            decision, next_action = ("candidate", "replay_other_method") if record["candidate_signal"] else ("confirmed_negative", "replay_other_method")
        else:
            decision, next_action = final_decision, "stop_episode" if final_decision != "abstain" else "abstain_unknown_oracle"
        pair_id = next((control["bridge_evidence"]["evidence_hash"] for control in controls if control["method"] == record["method"]), None) if record["role"] == "candidate" else None
        current = _step(record, episode_id=episode_id, index=index, parent=parent, decision=decision, next_action=next_action, before=before, after=after, pair_id=pair_id)
        model_input = dict(record["model_input"])
        model_input["belief_before"] = dict(before)
        current["model_input"] = model_input
        steps.append(current)
        parent = current["step_id"]
    return {
        "episode_id": episode_id,
        "source": surface.source,
        "target_seed": target_seed,
        "target_instance_id": f"pg116-{surface.source}-target-{target_seed}",
        "surface_kind": surface.kind,
        "oracle_available": surface.typed_available,
        "expected_outcome": surface.expected,
        "steps": steps,
        "evidence_records": [record["bridge_evidence"] for record in records],
        "episode_report": evaluate_episode(steps),
        "final_decision": final_decision,
        "candidate_pair_positive": surface.expected == "positive" and final_decision == "confirmed_positive",
        "negative_control_pair_clear": all(not record["positive"] for record in controls),
        "bsp": {
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": bsp_state.parameter_sha256(),
            "parameter_unchanged": parameter_before == bsp_state.parameter_sha256(),
            "topology_version": bsp_state.topology_version,
            "state_sha256": bsp_state.state_sha256(),
        },
    }


async def collect_source(source: str, seeds: list[int]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        # Recreate the target so the target instance identity and state are
        # fresh for every seed; each individual action still resets as well.
        app = create_target_app(source, seed)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            health = await client.get("/health")
            if health.status_code != 200 or health.json().get("source") != source:
                raise RuntimeError("PG-116 source identity failed")
            for surface in SURFACES:
                if surface.source == source:
                    episodes.append(await collect_episode(client, surface, target_seed=seed))
    return {"source": source, "target_seeds": seeds, "episodes": episodes}


__all__ = ["SCHEMA_VERSION", "SURFACES", "collect_episode", "collect_source"]
