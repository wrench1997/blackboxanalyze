"""PG-125 GET/POST replay bridge with failure-conditioned next actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .bsp_v3_research_core import BspV3Config, BspV3State
from .failure_guided_scheduler import failure_signature
from .generic_belief_state import GenericBeliefState
from .pg123_authorization_rule_ir_decoder import canonical_model_input
from .pg125_scope_logic_target import SURFACES as TARGET_SURFACES
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg125-scope-logic-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg125-scope-logic-bridge-evidence-v1"
ORACLE_CONTRACT = "pg125-scope-transition-typed-oracle-v1"
ENCODING_CHAIN = ["unicode_escape", "url_percent", "html_entity"]


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
        raise ValueError("PG-125 evidence hash is invalid")
    body = dict(value)
    body.pop("evidence_hash", None)
    if sha256_json(body) != declared:
        raise ValueError("PG-125 evidence hash mismatch")
    return declared


def _payload(surface: Surface, candidate: bool, marker: str) -> dict[str, Any]:
    if surface.kind == "scope":
        return {"scope_level": "cross_tenant" if candidate else "local"}
    if surface.kind == "decoy":
        return {"view_mode": "cross_tenant" if candidate else "local"}
    if surface.kind == "steady":
        return {"state": "local"}
    return {"phase": "canary" if candidate else "normal", "marker": marker}


def _bsp(state: BspV3State, evidence_hash: str) -> dict[str, Any]:
    import numpy as np

    values = [int(evidence_hash[index : index + 2], 16) / 255.0 for index in (0, 2, 4, 6)]
    output = state.forward(np.asarray([values], dtype=np.float64), np.asarray([[0.5, 0.5]], dtype=np.float64))
    return {"selected_leaf_ids": [int(value) for value in output.selected_leaf_ids.reshape(-1)], "leaf_mass_error": float(abs(float(output.leaf_mass_sum.sum()) - 1.0)), "topology_version": state.topology_version, "state_sha256": state.state_sha256()}


def _oracle(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool, str]:
    target_evidence = response.get("evidence")
    if not isinstance(target_evidence, Mapping):
        raise ValueError("PG-125 target omitted evidence")
    target_hash = _verify_hash(target_evidence)
    evaluator = target_evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("PG-125 target omitted evaluator")
    available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate = bool(evaluator.get("candidate_signal"))
    positive = bool(available and evaluator.get("typed_effect"))
    projection = {"modality": "scope_typed_differential" if available else "scope_untyped_signal", "positive": positive, "positive_authority": positive, "candidate_signal": candidate, "observed_atoms": ["scope_transition"] if positive else ["candidate_signal_observed"] if candidate else ["no_effect"], "oracle_contract_sha256": _hash_text(ORACLE_CONTRACT), "source_evidence_sha256": target_hash, "safety": {"external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "navigation": False, "credentials_accessed": False}}
    return projection, positive, positive, target_hash


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    if record["positive_authority"]:
        return {"effect": 0.86, "input_only": 0.03, "no_effect": 0.03, "unknown": 0.08}
    if record["candidate_signal"]:
        return {"effect": 0.08, "input_only": 0.28, "no_effect": 0.08, "unknown": 0.56}
    return {"effect": 0.03, "input_only": 0.07, "no_effect": 0.81, "unknown": 0.09}


async def _action(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int, method: str, role: str, marker: str, state: BspV3State) -> dict[str, Any]:
    health = await client.get("/healthz")
    if health.status_code != 200 or health.json().get("implementation") != "pg125-sigma-scope-target":
        raise RuntimeError("PG-125 target identity failed")
    reset_response = await client.post("/v10/reset")
    reset = reset_response.json().get("reset")
    if reset_response.status_code != 200 or not isinstance(reset, Mapping):
        raise RuntimeError("PG-125 fresh reset failed")
    reset_hash = _verify_hash(reset)
    if not (reset["fresh_target"] and reset["completed"] and reset["evaluator_state_hidden"] and reset["state_change_allowed"] is False):
        raise ValueError("PG-125 reset contract failed")
    payload = _payload(surface, role == "candidate", marker)
    route = f"/v10/inspect/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"PG-125 target returned {response.status_code}")
    body = response.json()
    oracle, positive, authority, target_hash = _oracle(surface, body)
    available = bool((body.get("evidence") or {}).get("evaluator", {}).get("typed_available")) and surface.typed_available
    target_projection = dict(body.get("response_projection") or {})
    response_projection = {**target_projection, "response_projection_sha256": sha256_json(target_projection), "candidate_signal": bool(body.get("candidate_signal"))}
    request_hash = sha256_json({"surface": surface.kind, "method": method, "role": role, "decoy_strength": decoy_strength, "encoding_chain": ENCODING_CHAIN, "payload_keys": sorted(payload.keys()), "marker_sha256": _hash_text(marker)})
    target_id = f"pg125-sigma-target-{seed}-d{decoy_strength}"
    evidence_body = {"schema_version": EVIDENCE_SCHEMA_VERSION, "target_instance_id": target_id, "surface": surface.kind, "decoy_strength": decoy_strength, "method": method, "role": role, "encoding_chain": ENCODING_CHAIN, "request_sha256": request_hash, "target_evidence_sha256": target_hash, "reset_evidence_sha256": reset_hash, "response_projection": dict(response_projection), "oracle_projection": dict(oracle), "fresh_reset": dict(reset), "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False}}
    bridge_evidence = {**evidence_body, "evidence_hash": sha256_json(evidence_body)}
    response_projection["bsp_core_projection"] = _bsp(state, bridge_evidence["evidence_hash"])
    action_manifest = {"method": method, "route_template_id": f"pg125-sigma-route-{_hash_text(route)[:16]}", "placement": placement, "encoding_chain": ENCODING_CHAIN, "probe_ref": f"pg125-abstract-triple-probe-{_hash_text(surface.kind)[:12]}", "probe_sha256": request_hash, "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action_manifest["form_field_names"] = ["abstract_probe"]
    model_input = canonical_model_input({"action_manifest": action_manifest, "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095"}, "response_projection": response_projection, "belief_before": {}})
    return {"seed": seed, "decoy_strength": decoy_strength, "target_instance_id": target_id, "surface": surface.kind, "method": method, "role": role, "candidate_signal": bool(oracle["candidate_signal"]), "positive": positive, "positive_authority": authority, "typed_available": available, "oracle_available": surface.typed_available, "action_manifest": action_manifest, "baseline_projection": model_input["baseline_projection"], "response_projection": response_projection, "model_input": model_input, "oracle_projection": oracle, "fresh_reset": {**dict(reset), "reset_evidence_sha256": reset_hash}, "bridge_evidence": bridge_evidence}


def _step(record: Mapping[str, Any], *, episode_id: str, index: int, parent: str | None, decision: str, next_action: str, before: Mapping[str, float], after: Mapping[str, float], pair_id: str | None, prior_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if pair_id:
        oracle["negative_control_pair_id"] = pair_id
    failure = failure_signature(record, prior_records=prior_records, step_count=index)
    body = {"action_manifest": record["action_manifest"], "baseline_projection": record["baseline_projection"], "response_projection": record["response_projection"], "oracle_projection": oracle, "belief_before": dict(before), "belief_after": dict(after), "decision": decision, "next_action": next_action, "failure_signature": failure}
    step = validate_trace_step({"episode_id": episode_id, "step_id": f"{episode_id}-s{index:02d}", "parent_step_id": parent, "sampling_seed": record["seed"], "target_instance_id": record["target_instance_id"], "hypothesis": "unknown_scope_surface", **body, "fresh_reset": record["fresh_reset"], "evidence_sha256": record["bridge_evidence"]["evidence_hash"], "dataset_stage": "pg125_cross_family_failure_policy_holdout", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": sha256_json(body)}})
    step["model_input"] = dict(record["model_input"])
    step["model_input"]["belief_before"] = dict(before)
    return step


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int) -> dict[str, Any]:
    episode_id = f"pg125-sigma-{seed}-d{decoy_strength}-{surface.kind}"
    state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=seed * 10 + decoy_strength)
    before_parameter = state.parameter_sha256()
    marker = f"pg125{seed}{decoy_strength}{surface.kind[:2]}"
    records = [await _action(client, surface, seed=seed, decoy_strength=decoy_strength, method=method, role=role, marker=marker, state=state) for role, method in (("control", "GET"), ("candidate", "GET"), ("control", "POST"), ("candidate", "POST"))]
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
    prior_records: list[Mapping[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prior = dict(belief.posterior)
        posterior = dict(belief.observe(f"{episode_id}-belief-{index}", _likelihood(record), evidence_hash=record["bridge_evidence"]["evidence_hash"])["posterior"])
        if record["role"] == "control":
            decision, next_action = "confirmed_negative", "repeat_matched_negative_pair"
        elif record["method"] == "GET":
            decision, next_action = ("candidate", "probe_candidate_other_method") if record["candidate_signal"] else ("confirmed_negative", "probe_candidate_other_method")
        else:
            decision, next_action = final_decision, "stop_confirmed_positive" if final_decision == "confirmed_positive" else "abstain_unknown_oracle" if final_decision == "abstain" else "stop_confirmed_negative"
        pair_id = next((control["bridge_evidence"]["evidence_hash"] for control in controls if control["method"] == record["method"]), None) if record["role"] == "candidate" else None
        current = _step(record, episode_id=episode_id, index=index, parent=parent, decision=decision, next_action=next_action, before=prior, after=posterior, pair_id=pair_id, prior_records=prior_records)
        steps.append(current)
        parent = current["step_id"]
        prior_records.append(record)
    binding = {"slot_id": f"pg125-sigma-scope-slot-{_hash_text(surface.kind)[:16]}", "binding_stage": "after_failure_guided_probe_and_typed_oracle", "decision": final_decision, "evidence_sha256": steps[-1]["evidence_sha256"], "shadow_probe_evidence_sha256": [step["evidence_sha256"] for step in steps], "typed_oracle_available": surface.typed_available, "positive_authority": final_decision == "confirmed_positive", "failure_signatures_recorded": True, "long_term_memory_write": False}
    return {"episode_id": episode_id, "target_instance_id": f"pg125-sigma-target-{seed}-d{decoy_strength}", "target_seed": seed, "decoy_strength": decoy_strength, "surface_kind": surface.kind, "oracle_available": surface.typed_available, "expected_outcome": surface.expected, "encoding_chain": ENCODING_CHAIN, "steps": steps, "evidence_records": [record["bridge_evidence"] for record in records], "episode_report": evaluate_episode(steps), "final_decision": final_decision, "candidate_pair_positive": surface.expected == "positive" and final_decision == "confirmed_positive", "negative_control_pair_clear": all(not record["positive"] for record in controls), "rule_ir_slot_binding": binding, "bsp": {"parameter_sha256_before": before_parameter, "parameter_sha256_after": state.parameter_sha256(), "parameter_unchanged": before_parameter == state.parameter_sha256(), "topology_version": state.topology_version, "state_sha256": state.state_sha256()}}


async def collect_target(seed: int, *, decoy_strength: int = 1) -> dict[str, Any]:
    from .pg125_scope_logic_target import create_app

    app = create_app(seed, decoy_strength=decoy_strength)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        health = await client.get("/healthz")
        if health.status_code != 200 or health.json().get("implementation") != "pg125-sigma-scope-target":
            raise RuntimeError("PG-125 health identity failed")
        episodes = [await collect_episode(client, surface, seed=seed, decoy_strength=decoy_strength) for surface in SURFACES]
    return {"target_seed": seed, "decoy_strength": decoy_strength, "target_instance_id": f"pg125-sigma-target-{seed}-d{decoy_strength}", "target_implementation": "pg125-sigma-scope-target", "target_schema_version": health.json().get("schema_version"), "episodes": episodes}


__all__ = ["ENCODING_CHAIN", "SCHEMA_VERSION", "SURFACES", "collect_target"]
