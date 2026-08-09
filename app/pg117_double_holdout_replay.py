"""PG-117 implementation + double-encoding holdout replay bridge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .bsp_v3_research_core import BspV3Config, BspV3State
from .generic_belief_state import GenericBeliefState
from .pg115_small_rule_ir_decoder import canonical_model_input
from .pg117_gamma_target import SURFACES as TARGET_SURFACES
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg117-double-holdout-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg117-double-holdout-bridge-evidence-v1"
ORACLE_CONTRACT = "pg117-gamma-route-typed-oracle-v1"
ENCODING_CHAIN = ["url_percent", "html_entity"]


@dataclass(frozen=True)
class Surface:
    kind: str
    expected: str
    typed_available: bool


SURFACES = tuple(Surface(kind, "positive" if index == 0 else "negative" if index < 3 else "abstain", index < 3) for index, kind in enumerate(TARGET_SURFACES))


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _verify_hash(value: Mapping[str, Any]) -> str:
    declared = str(value.get("evidence_hash", ""))
    if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise ValueError("PG-117 evidence hash is invalid")
    body = dict(value)
    body.pop("evidence_hash", None)
    if sha256_json(body) != declared:
        raise ValueError("PG-117 evidence hash mismatch")
    return declared


def _payload(surface: Surface, candidate: bool, marker: str) -> dict[str, Any]:
    if surface.kind == "route":
        return {"route_mode": "alternate" if candidate else "baseline"}
    if surface.kind == "decoy":
        return {"shape_mode": "alternate" if candidate else "baseline"}
    if surface.kind == "steady":
        return {"state": "baseline"}
    return {"mode": "canary" if candidate else "control", "marker": marker}


def _bsp(state: BspV3State, evidence_hash: str) -> dict[str, Any]:
    import numpy as np

    values = [int(evidence_hash[index : index + 2], 16) / 255.0 for index in (0, 2, 4, 6)]
    output = state.forward(np.asarray([values], dtype=np.float64), np.asarray([[0.5, 0.5]], dtype=np.float64))
    return {"selected_leaf_ids": [int(value) for value in output.selected_leaf_ids.reshape(-1)], "leaf_mass_error": float(abs(float(output.leaf_mass_sum.sum()) - 1.0)), "topology_version": state.topology_version, "state_sha256": state.state_sha256()}


def _oracle(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool, str]:
    target_evidence = response.get("evidence")
    if not isinstance(target_evidence, Mapping):
        raise ValueError("PG-117 target omitted evidence")
    target_hash = _verify_hash(target_evidence)
    evaluator = target_evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("PG-117 target omitted evaluator")
    available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate = bool(evaluator.get("candidate_signal"))
    effect = bool(evaluator.get("typed_effect")) if available else False
    positive = bool(available and effect)
    projection = {"modality": "route_typed_differential" if available else "route_untyped_signal", "positive": positive, "positive_authority": positive, "candidate_signal": candidate, "observed_atoms": ["route_transition"] if positive else ["candidate_signal_observed"] if candidate else ["no_effect"], "oracle_contract_sha256": _hash_text(ORACLE_CONTRACT), "source_evidence_sha256": target_hash, "safety": {"external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "navigation": False, "credentials_accessed": False}}
    return projection, positive, available, target_hash


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    if record["positive_authority"]:
        return {"effect": 0.84, "input_only": 0.04, "no_effect": 0.03, "unknown": 0.09}
    if record["candidate_signal"]:
        return {"effect": 0.10, "input_only": 0.25, "no_effect": 0.08, "unknown": 0.57}
    return {"effect": 0.03, "input_only": 0.07, "no_effect": 0.80, "unknown": 0.10}


async def _action(client: httpx.AsyncClient, surface: Surface, *, seed: int, method: str, role: str, marker: str, state: BspV3State) -> dict[str, Any]:
    health = await client.get("/health")
    if health.status_code != 200 or health.json().get("implementation") != "pg117-gamma-independent-target":
        raise RuntimeError("PG-117 target identity failed")
    reset_response = await client.post("/v5/reset")
    reset = reset_response.json().get("reset")
    if reset_response.status_code != 200 or not isinstance(reset, Mapping):
        raise RuntimeError("PG-117 fresh reset failed")
    reset_hash = _verify_hash(reset)
    if not (reset["fresh_target"] and reset["completed"] and reset["evaluator_state_hidden"] and reset["state_change_allowed"] is False):
        raise ValueError("PG-117 reset contract failed")
    payload = _payload(surface, role == "candidate", marker)
    route = f"/v5/inspect/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"PG-117 target returned {response.status_code}")
    body = response.json()
    oracle, positive, authority, target_hash = _oracle(surface, body)
    target_projection = dict(body.get("response_projection") or {})
    response_projection = {**target_projection, "response_projection_sha256": sha256_json(target_projection), "candidate_signal": bool(body.get("candidate_signal"))}
    request_hash = sha256_json({"surface": surface.kind, "method": method, "role": role, "encoding_chain": ENCODING_CHAIN, "payload_keys": sorted(payload.keys()), "marker_sha256": _hash_text(marker)})
    evidence_body = {"schema_version": EVIDENCE_SCHEMA_VERSION, "target_instance_id": f"pg117-gamma-target-{seed}", "surface": surface.kind, "method": method, "role": role, "encoding_chain": ENCODING_CHAIN, "request_sha256": request_hash, "target_evidence_sha256": target_hash, "reset_evidence_sha256": reset_hash, "response_projection": dict(response_projection), "oracle_projection": dict(oracle), "fresh_reset": dict(reset), "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False}}
    bridge_evidence = {**evidence_body, "evidence_hash": sha256_json(evidence_body)}
    response_projection["bsp_core_projection"] = _bsp(state, bridge_evidence["evidence_hash"])
    action_manifest = {"method": method, "route_template_id": f"pg117-route-{_hash_text(route)[:16]}", "placement": placement, "encoding_chain": ENCODING_CHAIN, "probe_ref": f"pg117-abstract-double-probe-{_hash_text(surface.kind)[:12]}", "probe_sha256": request_hash, "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action_manifest["form_field_names"] = ["abstract_probe"]
    model_input = canonical_model_input({"action_manifest": action_manifest, "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095"}, "response_projection": response_projection, "belief_before": {}})
    return {"seed": seed, "target_instance_id": f"pg117-gamma-target-{seed}", "surface": surface.kind, "method": method, "role": role, "candidate_signal": bool(oracle["candidate_signal"]), "positive": positive, "positive_authority": authority, "action_manifest": action_manifest, "baseline_projection": model_input["baseline_projection"], "response_projection": response_projection, "model_input": model_input, "oracle_projection": oracle, "fresh_reset": {**dict(reset), "reset_evidence_sha256": reset_hash}, "bridge_evidence": bridge_evidence}


def _step(record: Mapping[str, Any], *, episode_id: str, index: int, parent: str | None, decision: str, next_action: str, before: Mapping[str, float], after: Mapping[str, float], pair_id: str | None) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if pair_id:
        oracle["negative_control_pair_id"] = pair_id
    body = {"action_manifest": record["action_manifest"], "baseline_projection": record["baseline_projection"], "response_projection": record["response_projection"], "oracle_projection": oracle, "belief_before": dict(before), "belief_after": dict(after), "decision": decision, "next_action": next_action}
    step = validate_trace_step({"episode_id": episode_id, "step_id": f"{episode_id}-s{index:02d}", "parent_step_id": parent, "sampling_seed": record["seed"], "target_instance_id": record["target_instance_id"], "hypothesis": "unknown_surface", **body, "fresh_reset": record["fresh_reset"], "evidence_sha256": record["bridge_evidence"]["evidence_hash"], "dataset_stage": "pg117_double_implementation_encoding_holdout", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": sha256_json(body)}})
    model_input = dict(record["model_input"])
    model_input["belief_before"] = dict(before)
    step["model_input"] = model_input
    return step


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, seed: int) -> dict[str, Any]:
    episode_id = f"pg117-gamma-{seed}-{surface.kind}"
    state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=seed)
    before_parameter = state.parameter_sha256()
    marker = f"pg117{seed}{surface.kind[:3]}"
    records = [await _action(client, surface, seed=seed, method=method, role=role, marker=marker, state=state) for role, method in (("control", "GET"), ("candidate", "GET"), ("control", "POST"), ("candidate", "POST"))]
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
        prior = dict(belief.posterior)
        posterior = dict(belief.observe(f"{episode_id}-belief-{index}", _likelihood(record), evidence_hash=record["bridge_evidence"]["evidence_hash"])["posterior"])
        if record["role"] == "control":
            decision, next_action = "confirmed_negative", "probe_candidate_same_method"
        elif record["method"] == "GET":
            decision, next_action = ("candidate", "replay_other_method") if record["candidate_signal"] else ("confirmed_negative", "replay_other_method")
        else:
            decision, next_action = final_decision, "stop_episode" if final_decision != "abstain" else "abstain_unknown_oracle"
        pair_id = next((control["bridge_evidence"]["evidence_hash"] for control in controls if control["method"] == record["method"]), None) if record["role"] == "candidate" else None
        current = _step(record, episode_id=episode_id, index=index, parent=parent, decision=decision, next_action=next_action, before=prior, after=posterior, pair_id=pair_id)
        steps.append(current)
        parent = current["step_id"]
    return {"episode_id": episode_id, "target_instance_id": f"pg117-gamma-target-{seed}", "target_seed": seed, "surface_kind": surface.kind, "oracle_available": surface.typed_available, "expected_outcome": surface.expected, "encoding_chain": ENCODING_CHAIN, "steps": steps, "evidence_records": [record["bridge_evidence"] for record in records], "episode_report": evaluate_episode(steps), "final_decision": final_decision, "candidate_pair_positive": surface.expected == "positive" and final_decision == "confirmed_positive", "negative_control_pair_clear": all(not record["positive"] for record in controls), "bsp": {"parameter_sha256_before": before_parameter, "parameter_sha256_after": state.parameter_sha256(), "parameter_unchanged": before_parameter == state.parameter_sha256(), "topology_version": state.topology_version, "state_sha256": state.state_sha256()}}


async def collect_target(seed: int) -> dict[str, Any]:
    from .pg117_gamma_target import create_app

    app = create_app(seed)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        health = await client.get("/health")
        if health.status_code != 200 or health.json().get("implementation") != "pg117-gamma-independent-target":
            raise RuntimeError("PG-117 health identity failed")
        episodes = [await collect_episode(client, surface, seed=seed) for surface in SURFACES]
    return {"target_seed": seed, "target_instance_id": f"pg117-gamma-target-{seed}", "target_implementation": "pg117-gamma-independent-target", "target_schema_version": health.json().get("schema_version"), "episodes": episodes}


__all__ = ["ENCODING_CHAIN", "SCHEMA_VERSION", "SURFACES", "collect_target"]

