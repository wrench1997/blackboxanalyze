"""PG-135 balanced 2GET/2POST replay bridge.

This bridge reuses only the local PG-133 ASGI fixture as a target surface, but
uses a new four-step action plan. Every episode has exactly two GET and two
POST requests; the current counterfactual step is the same candidate POST in
both history variants. No external network, code execution, database write,
or real vulnerability payload is involved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .failure_guided_scheduler import failure_signature
from .generic_belief_state import GenericBeliefState
from .pg133_history_latch_target import SURFACES as TARGET_SURFACES
from .pg133_history_latch_target import create_app
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg135-balanced-history-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg135-balanced-history-bridge-evidence-v1"
ORACLE_CONTRACT = "pg135-balanced-history-workflow-oracle-v1"
MAX_STEPS = 4
ACTION_PLANS = {
    "control_first": (("control", "GET"), ("candidate", "POST"), ("control", "GET"), ("control", "POST")),
    "candidate_first": (("candidate", "GET"), ("candidate", "POST"), ("control", "GET"), ("control", "POST")),
}


@dataclass(frozen=True)
class Surface:
    kind: str
    typed_available: bool


SURFACES = tuple(Surface(kind, kind != "blind") for kind in TARGET_SURFACES)


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _verify_hash(value: Mapping[str, Any]) -> str:
    declared = str(value.get("evidence_hash", ""))
    if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise ValueError("PG-135 evidence hash is invalid")
    body = dict(value)
    body.pop("evidence_hash", None)
    if sha256_json(body) != declared:
        raise ValueError("PG-135 evidence hash mismatch")
    return declared


def _payload(role: str, phase: str, marker: str) -> dict[str, Any]:
    return {"role": role, "probe_phase": phase, "marker": marker}


def _oracle(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool, str]:
    evidence = response.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("PG-135 target omitted evidence")
    target_hash = _verify_hash(evidence)
    evaluator = evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("PG-135 target omitted evaluator")
    available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate = bool(evaluator.get("candidate_signal"))
    action = str(evaluator.get("workflow_action", ""))
    if action not in {"repeat_matched_negative_pair", "probe_candidate_other_method", "abstain_unknown_oracle"}:
        raise ValueError("PG-135 evaluator emitted unsafe action")
    oracle = {
        "modality": "history_workflow_typed" if available else "history_workflow_unknown",
        "positive": False,
        "positive_authority": False,
        "candidate_signal": candidate,
        "observed_atoms": ["candidate_signal_observed"] if candidate else ["no_effect"],
        "oracle_contract_sha256": _hash_text(ORACLE_CONTRACT),
        "source_evidence_sha256": target_hash,
        "safety": {"external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "navigation": False, "credentials_accessed": False},
    }
    authority = {"history_stage": str(evaluator.get("history_stage", "unknown")), "workflow_action": action, "methods_seen": [str(item).upper() for item in evaluator.get("methods_seen", []) if str(item).upper() in {"GET", "POST"}], "typed_available": available, "source_evidence_sha256": target_hash}
    return oracle, authority, available, target_hash


async def _action(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int, method: str, role: str, variant: str, index: int, marker: str) -> dict[str, Any]:
    payload = _payload(role, "confirm" if role == "candidate" else "shadow", marker)
    route = f"/v13/observe/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"PG-135 target returned {response.status_code}")
    body = response.json()
    oracle, authority, available, target_hash = _oracle(surface, body)
    projection = dict(body.get("response_projection") or {})
    response_projection = {**projection, "response_projection_sha256": sha256_json(projection), "candidate_signal": bool(body.get("candidate_signal"))}
    request_hash = sha256_json({"surface": surface.kind, "method": method, "role": role, "variant": variant, "step_index": index, "decoy_strength": decoy_strength, "payload_keys": sorted(payload.keys()), "marker_sha256": _hash_text(marker)})
    route_id = f"pg135-balanced-route-{_hash_text(route)[:16]}"
    manifest = {"method": method, "route_template_id": route_id, "placement": placement, "encoding_chain": ["url_percent", "unicode_escape"], "probe_ref": f"pg135-safe-workflow-probe-{_hash_text(surface.kind + role)[:12]}", "probe_sha256": request_hash, "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        manifest["form_field_names"] = ["role", "probe_phase", "marker"]
    target_id = f"pg133-latch-target-{seed}-d{decoy_strength}"
    bridge = {"schema_version": EVIDENCE_SCHEMA_VERSION, "target_instance_id": target_id, "surface": surface.kind, "method": method, "role": role, "episode_variant": variant, "step_index": index, "request_sha256": request_hash, "target_evidence_sha256": target_hash, "response_projection": response_projection, "oracle_projection": oracle, "workflow_oracle": authority, "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False}}
    bridge["evidence_hash"] = sha256_json(bridge)
    return {"seed": seed, "target_instance_id": target_id, "surface": surface.kind, "method": method, "role": role, "variant": variant, "step_index": index, "candidate_signal": bool(oracle["candidate_signal"]), "positive": False, "positive_authority": False, "typed_available": available, "oracle_available": surface.typed_available, "workflow_authority": authority, "action_manifest": manifest, "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095", "shape_class": "decision-v8"}, "response_projection": response_projection, "oracle_projection": oracle, "bridge_evidence": bridge}


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    return {"effect": 0.02, "input_only": 0.12 if record["candidate_signal"] else 0.04, "no_effect": 0.78 if not record["candidate_signal"] else 0.18, "unknown": 0.68 if record["candidate_signal"] and not record["typed_available"] else 0.08}


def _step(record: Mapping[str, Any], *, episode_id: str, index: int, parent: str | None, before: Mapping[str, float], after: Mapping[str, float], pair_id: str | None, prior: list[Mapping[str, Any]], fresh_reset: Mapping[str, Any]) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if pair_id:
        oracle["negative_control_pair_id"] = pair_id
    failure = failure_signature(record, prior_records=prior, max_steps=MAX_STEPS, step_count=index)
    failure["next_action"] = str(record["workflow_authority"]["workflow_action"])
    body = {"action_manifest": record["action_manifest"], "baseline_projection": record["baseline_projection"], "response_projection": record["response_projection"], "oracle_projection": oracle, "belief_before": dict(before), "belief_after": dict(after), "decision": "confirmed_negative" if record["role"] == "control" else "candidate", "next_action": failure["next_action"], "failure_signature": failure}
    return validate_trace_step({"episode_id": episode_id, "step_id": f"{episode_id}-s{index:02d}", "parent_step_id": parent, "sampling_seed": record["seed"], "target_instance_id": record["target_instance_id"], "hypothesis": "balanced_history_sensitive_safe_workflow", **body, "fresh_reset": fresh_reset, "evidence_sha256": record["bridge_evidence"]["evidence_hash"], "dataset_stage": "pg135_balanced_history_replay", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": sha256_json(body)}})


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int, variant: str) -> dict[str, Any]:
    episode_id = f"pg135-balanced-{seed}-d{decoy_strength}-{surface.kind}-{variant}"
    reset_response = await client.post("/v13/reset")
    reset = reset_response.json().get("reset")
    if reset_response.status_code != 200 or not isinstance(reset, Mapping):
        raise RuntimeError("PG-135 fresh reset failed")
    reset_hash = _verify_hash(reset)
    if not (reset["fresh_target"] and reset["completed"] and reset["evaluator_state_hidden"] and reset["state_change_allowed"] is False):
        raise ValueError("PG-135 reset contract failed")
    fresh_reset = {**dict(reset), "reset_evidence_sha256": reset_hash}
    marker = f"pg135{seed}{decoy_strength}{surface.kind[:2]}{variant[:2]}"
    records = [await _action(client, surface, seed=seed, decoy_strength=decoy_strength, method=method, role=role, variant=variant, index=index, marker=marker) for index, (role, method) in enumerate(ACTION_PLANS[variant], start=1)]
    controls_by_method = {record["method"]: record for record in records if record["role"] == "control"}
    if sum(record["method"] == "GET" for record in records) != 2 or sum(record["method"] == "POST" for record in records) != 2:
        raise AssertionError("PG-135 action plan is not exactly 2GET/2POST")
    belief = GenericBeliefState()
    steps: list[dict[str, Any]] = []
    parent: str | None = None
    prior: list[Mapping[str, Any]] = []
    for index, record in enumerate(records, start=1):
        before = dict(belief.posterior)
        after = dict(belief.observe(f"{episode_id}-belief-{index}", _likelihood(record), evidence_hash=record["bridge_evidence"]["evidence_hash"])["posterior"])
        pair = controls_by_method.get(record["method"]) if record["role"] == "candidate" else None
        record["bridge_evidence"]["reset_evidence_sha256"] = reset_hash
        record["bridge_evidence"]["fresh_reset"] = fresh_reset
        record["bridge_evidence"]["evidence_hash"] = sha256_json({key: value for key, value in record["bridge_evidence"].items() if key != "evidence_hash"})
        steps.append(_step(record, episode_id=episode_id, index=index, parent=parent, before=before, after=after, pair_id=pair["bridge_evidence"]["evidence_hash"] if pair else None, prior=prior, fresh_reset=fresh_reset))
        parent = steps[-1]["step_id"]
        prior.append(record)
    current = records[1]
    pair_id = sha256_json({"surface": surface.kind, "method": current["method"], "response_projection": current["response_projection"], "route_template_id": current["action_manifest"]["route_template_id"], "placement": current["action_manifest"]["placement"]})
    binding = {"slot_id": f"pg135-history-slot-{_hash_text(surface.kind)[:16]}", "binding_stage": "after_evaluator_only_history_workflow_oracle", "decision": "abstain", "evidence_sha256": steps[-1]["evidence_sha256"], "shadow_probe_evidence_sha256": [step["evidence_sha256"] for step in steps], "typed_oracle_available": surface.typed_available, "positive_authority": False, "failure_signatures_recorded": True, "counterfactual_pair_id": pair_id, "long_term_memory_write": False}
    return {"episode_id": episode_id, "target_instance_id": f"pg133-latch-target-{seed}-d{decoy_strength}", "target_seed": seed, "decoy_strength": decoy_strength, "surface_kind": surface.kind, "episode_variant": variant, "oracle_available": surface.typed_available, "steps": steps, "episode_report": evaluate_episode(steps), "final_decision": "abstain", "negative_control_pair_clear": bool(controls_by_method), "counterfactual_pair_id": pair_id, "history_authority": {"current_step": f"{episode_id}-s02", "history_stage": current["workflow_authority"]["history_stage"], "workflow_action": current["workflow_authority"]["workflow_action"], "source_evidence_sha256": current["workflow_authority"]["source_evidence_sha256"]}, "rule_ir_slot_binding": binding, "get_count": sum(record["method"] == "GET" for record in records), "post_count": sum(record["method"] == "POST" for record in records)}


async def collect_target(seed: int, *, decoy_strength: int = 1) -> dict[str, Any]:
    app = create_app(seed, decoy_strength=decoy_strength)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        health = await client.get("/healthz")
        if health.status_code != 200 or health.json().get("implementation") != "pg133-history-latch-target":
            raise RuntimeError("PG-135 target identity failed")
        episodes = [await collect_episode(client, surface, seed=seed, decoy_strength=decoy_strength, variant=variant) for surface in SURFACES for variant in ACTION_PLANS]
    return {"target_seed": seed, "decoy_strength": decoy_strength, "target_instance_id": f"pg133-latch-target-{seed}-d{decoy_strength}", "target_implementation": "pg133-history-latch-target-pg135-balanced-replay", "target_schema_version": health.json().get("schema_version"), "episodes": episodes, "get_count": sum(episode["get_count"] for episode in episodes), "post_count": sum(episode["post_count"] for episode in episodes)}


__all__ = ["ACTION_PLANS", "MAX_STEPS", "SCHEMA_VERSION", "SURFACES", "collect_episode", "collect_target"]
