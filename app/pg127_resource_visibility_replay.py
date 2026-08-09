"""PG-127 six-step GET/POST shadow-to-confirm replay bridge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .bsp_v3_research_core import BspV3Config, BspV3State
from .failure_guided_scheduler import failure_signature
from .generic_belief_state import GenericBeliefState
from .pg123_authorization_rule_ir_decoder import canonical_model_input
from .pg127_resource_visibility_target import SURFACES as TARGET_SURFACES
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg127-visibility-long-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg127-visibility-long-bridge-evidence-v1"
ORACLE_CONTRACT = "pg127-visibility-transition-typed-oracle-v1"
ENCODING_CHAIN = ["html_entity", "unicode_escape", "url_percent"]
MAX_STEPS = 6
ACTION_PLAN = (("control", "GET", "shadow", 1), ("candidate", "GET", "shadow", 1), ("control", "POST", "shadow", 1), ("candidate", "POST", "shadow", 1), ("control", "GET", "confirm", 2), ("candidate", "POST", "confirm", 2))


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
        raise ValueError("PG-127 evidence hash is invalid")
    body = dict(value)
    body.pop("evidence_hash", None)
    if sha256_json(body) != declared:
        raise ValueError("PG-127 evidence hash mismatch")
    return declared


def _payload(surface: Surface, role: str, phase: str, marker: str) -> dict[str, Any]:
    candidate = role == "candidate"
    if surface.kind == "resource":
        return {"visibility": "shared" if candidate else "private", "probe_phase": phase}
    if surface.kind == "decoy":
        return {"preview": "shared" if candidate else "private", "probe_phase": phase}
    if surface.kind == "steady":
        return {"state": "private", "probe_phase": phase}
    return {"phase": "canary" if candidate else "normal", "marker": marker, "probe_phase": phase}


def _bsp(state: BspV3State, evidence_hash: str) -> dict[str, Any]:
    import numpy as np

    values = [int(evidence_hash[index : index + 2], 16) / 255.0 for index in (0, 2, 4, 6)]
    output = state.forward(np.asarray([values], dtype=np.float64), np.asarray([[0.5, 0.5]], dtype=np.float64))
    return {"selected_leaf_ids": [int(value) for value in output.selected_leaf_ids.reshape(-1)], "leaf_mass_error": float(abs(float(output.leaf_mass_sum.sum()) - 1.0)), "topology_version": state.topology_version, "state_sha256": state.state_sha256()}


def _oracle(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool, str]:
    evidence = response.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("PG-127 target omitted evidence")
    target_hash = _verify_hash(evidence)
    evaluator = evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("PG-127 target omitted evaluator")
    available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate = bool(evaluator.get("candidate_signal"))
    positive = bool(available and evaluator.get("typed_effect"))
    projection = {"modality": "visibility_typed_differential" if available else "visibility_untyped_signal", "positive": positive, "positive_authority": positive, "candidate_signal": candidate, "observed_atoms": ["visibility_transition"] if positive else ["candidate_signal_observed"] if candidate else ["no_effect"], "oracle_contract_sha256": _hash_text(ORACLE_CONTRACT), "source_evidence_sha256": target_hash, "safety": {"external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "navigation": False, "credentials_accessed": False}}
    return projection, positive, positive, target_hash


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    if record["positive_authority"]:
        return {"effect": 0.86, "input_only": 0.03, "no_effect": 0.03, "unknown": 0.08}
    if record["candidate_signal"]:
        return {"effect": 0.08, "input_only": 0.28, "no_effect": 0.08, "unknown": 0.56}
    return {"effect": 0.03, "input_only": 0.07, "no_effect": 0.81, "unknown": 0.09}


async def _action(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int, method: str, role: str, phase: str, probe_round: int, marker: str, state: BspV3State) -> dict[str, Any]:
    health = await client.get("/healthz")
    if health.status_code != 200 or health.json().get("implementation") != "pg127-omega-visibility-target":
        raise RuntimeError("PG-127 target identity failed")
    reset_response = await client.post("/v11/reset")
    reset = reset_response.json().get("reset")
    if reset_response.status_code != 200 or not isinstance(reset, Mapping):
        raise RuntimeError("PG-127 fresh reset failed")
    reset_hash = _verify_hash(reset)
    if not (reset["fresh_target"] and reset["completed"] and reset["evaluator_state_hidden"] and reset["state_change_allowed"] is False):
        raise ValueError("PG-127 reset contract failed")
    payload = _payload(surface, role, phase, marker)
    route = f"/v11/observe/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"PG-127 target returned {response.status_code}")
    body = response.json()
    oracle, positive, authority, target_hash = _oracle(surface, body)
    available = bool((body.get("evidence") or {}).get("evaluator", {}).get("typed_available")) and surface.typed_available
    target_projection = dict(body.get("response_projection") or {})
    response_projection = {**target_projection, "response_projection_sha256": sha256_json(target_projection), "candidate_signal": bool(body.get("candidate_signal"))}
    request_hash = sha256_json({"surface": surface.kind, "method": method, "role": role, "phase": phase, "probe_round": probe_round, "decoy_strength": decoy_strength, "encoding_chain": ENCODING_CHAIN, "payload_keys": sorted(payload.keys()), "marker_sha256": _hash_text(marker)})
    target_id = f"pg127-omega-target-{seed}-d{decoy_strength}"
    evidence_body = {"schema_version": EVIDENCE_SCHEMA_VERSION, "target_instance_id": target_id, "surface": surface.kind, "decoy_strength": decoy_strength, "method": method, "role": role, "phase": phase, "probe_round": probe_round, "encoding_chain": ENCODING_CHAIN, "request_sha256": request_hash, "target_evidence_sha256": target_hash, "reset_evidence_sha256": reset_hash, "response_projection": dict(response_projection), "oracle_projection": dict(oracle), "fresh_reset": dict(reset), "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False}}
    bridge_evidence = {**evidence_body, "evidence_hash": sha256_json(evidence_body)}
    response_projection["bsp_core_projection"] = _bsp(state, bridge_evidence["evidence_hash"])
    action_manifest = {"method": method, "route_template_id": f"pg127-omega-route-{_hash_text(route)[:16]}", "placement": placement, "encoding_chain": ENCODING_CHAIN, "probe_ref": f"pg127-abstract-probe-{_hash_text(surface.kind + phase)[:12]}", "probe_sha256": request_hash, "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action_manifest["form_field_names"] = ["abstract_probe"]
    model_input = canonical_model_input({"action_manifest": action_manifest, "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095"}, "response_projection": response_projection, "belief_before": {}})
    return {"seed": seed, "decoy_strength": decoy_strength, "target_instance_id": target_id, "surface": surface.kind, "method": method, "role": role, "phase": phase, "probe_round": probe_round, "max_probe_rounds": 2, "candidate_signal": bool(oracle["candidate_signal"]), "positive": positive, "positive_authority": authority, "typed_available": available, "oracle_available": surface.typed_available, "action_manifest": action_manifest, "baseline_projection": model_input["baseline_projection"], "response_projection": response_projection, "model_input": model_input, "oracle_projection": oracle, "fresh_reset": {**dict(reset), "reset_evidence_sha256": reset_hash}, "bridge_evidence": bridge_evidence}


def _step(record: Mapping[str, Any], *, episode_id: str, index: int, parent: str | None, decision: str, before: Mapping[str, float], after: Mapping[str, float], pair_id: str | None, prior_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if pair_id:
        oracle["negative_control_pair_id"] = pair_id
    failure = failure_signature(record, prior_records=prior_records, max_steps=MAX_STEPS, step_count=index)
    next_action = failure["next_action"]
    body = {"action_manifest": record["action_manifest"], "baseline_projection": record["baseline_projection"], "response_projection": record["response_projection"], "oracle_projection": oracle, "belief_before": dict(before), "belief_after": dict(after), "decision": decision, "next_action": next_action, "failure_signature": failure}
    step = validate_trace_step({"episode_id": episode_id, "step_id": f"{episode_id}-s{index:02d}", "parent_step_id": parent, "sampling_seed": record["seed"], "target_instance_id": record["target_instance_id"], "hypothesis": "unknown_visibility_surface", **body, "fresh_reset": record["fresh_reset"], "evidence_sha256": record["bridge_evidence"]["evidence_hash"], "dataset_stage": "pg127_long_horizon_visibility_holdout", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": sha256_json(body)}})
    step["model_input"] = dict(record["model_input"])
    step["model_input"]["belief_before"] = dict(before)
    step["probe_round"] = record["probe_round"]
    step["remaining_probe_budget"] = failure["remaining_probe_budget"]
    return step


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int) -> dict[str, Any]:
    episode_id = f"pg127-omega-{seed}-d{decoy_strength}-{surface.kind}"
    state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=seed * 10 + decoy_strength)
    before_parameter = state.parameter_sha256()
    marker = f"pg127{seed}{decoy_strength}{surface.kind[:2]}"
    records = [await _action(client, surface, seed=seed, decoy_strength=decoy_strength, method=method, role=role, phase=phase, probe_round=probe_round, marker=marker, state=state) for role, method, phase, probe_round in ACTION_PLAN]
    controls = [record for record in records if record["role"] == "control"]
    candidates = [record for record in records if record["role"] == "candidate"]
    positive_candidate = [record for record in candidates if record["positive_authority"]]
    if surface.expected == "positive":
        final_decision = "confirmed_positive" if positive_candidate and all(not record["positive"] for record in controls) else "abstain"
    elif surface.expected == "negative":
        final_decision = "confirmed_negative" if all(not record["positive"] for record in records) else "abstain"
    else:
        final_decision = "abstain"
    belief = GenericBeliefState()
    steps: list[dict[str, Any]] = []
    parent: str | None = None
    prior_records: list[Mapping[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prior = dict(belief.posterior)
        posterior = dict(belief.observe(f"{episode_id}-belief-{index}", _likelihood(record), evidence_hash=record["bridge_evidence"]["evidence_hash"])["posterior"])
        if record["role"] == "control":
            decision = "confirmed_negative"
        elif record["positive_authority"]:
            decision = final_decision
        elif record["candidate_signal"]:
            decision = "candidate"
        else:
            decision = "confirmed_negative"
        pair_id = next((control["bridge_evidence"]["evidence_hash"] for control in controls if control["method"] == record["method"]), None) if record["role"] == "candidate" else None
        current = _step(record, episode_id=episode_id, index=index, parent=parent, decision=decision, before=prior, after=posterior, pair_id=pair_id, prior_records=prior_records)
        steps.append(current)
        parent = current["step_id"]
        prior_records.append(record)
    binding = {"slot_id": f"pg127-omega-visibility-slot-{_hash_text(surface.kind)[:16]}", "binding_stage": "after_long_horizon_failure_guided_probe_and_typed_oracle", "decision": final_decision, "evidence_sha256": steps[-1]["evidence_sha256"], "shadow_probe_evidence_sha256": [step["evidence_sha256"] for step in steps], "typed_oracle_available": surface.typed_available, "positive_authority": final_decision == "confirmed_positive", "failure_signatures_recorded": True, "long_horizon_step_count": len(steps), "long_term_memory_write": False}
    return {"episode_id": episode_id, "target_instance_id": f"pg127-omega-target-{seed}-d{decoy_strength}", "target_seed": seed, "decoy_strength": decoy_strength, "surface_kind": surface.kind, "oracle_available": surface.typed_available, "expected_outcome": surface.expected, "steps": steps, "evidence_records": [record["bridge_evidence"] for record in records], "episode_report": evaluate_episode(steps), "final_decision": final_decision, "candidate_pair_positive": surface.expected == "positive" and final_decision == "confirmed_positive", "negative_control_pair_clear": all(not record["positive"] for record in controls), "rule_ir_slot_binding": binding, "bsp": {"parameter_sha256_before": before_parameter, "parameter_sha256_after": state.parameter_sha256(), "parameter_unchanged": before_parameter == state.parameter_sha256(), "topology_version": state.topology_version, "state_sha256": state.state_sha256()}}


async def collect_target(seed: int, *, decoy_strength: int = 1) -> dict[str, Any]:
    from .pg127_resource_visibility_target import create_app

    app = create_app(seed, decoy_strength=decoy_strength)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        health = await client.get("/healthz")
        if health.status_code != 200 or health.json().get("implementation") != "pg127-omega-visibility-target":
            raise RuntimeError("PG-127 health identity failed")
        episodes = [await collect_episode(client, surface, seed=seed, decoy_strength=decoy_strength) for surface in SURFACES]
    return {"target_seed": seed, "decoy_strength": decoy_strength, "target_instance_id": f"pg127-omega-target-{seed}-d{decoy_strength}", "target_implementation": "pg127-omega-visibility-target", "target_schema_version": health.json().get("schema_version"), "episodes": episodes}


__all__ = ["ACTION_PLAN", "ENCODING_CHAIN", "MAX_STEPS", "SCHEMA_VERSION", "SURFACES", "collect_target"]
