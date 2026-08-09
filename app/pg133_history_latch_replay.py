"""PG-133 replay bridge for history-sensitive safe workflow actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .failure_guided_scheduler import failure_signature
from .generic_belief_state import GenericBeliefState
from .pg133_history_latch_target import SURFACES as TARGET_SURFACES
from .trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


SCHEMA_VERSION = "pg133-history-latch-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg133-history-latch-bridge-evidence-v1"
ORACLE_CONTRACT = "pg133-history-latch-workflow-oracle-v1"
MAX_STEPS = 3
ACTION_PLANS = {
    "control_first": (("control", "GET"), ("candidate", "POST"), ("control", "POST")),
    "candidate_first": (("candidate", "GET"), ("candidate", "POST"), ("control", "POST")),
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
        raise ValueError("PG-133 evidence hash is invalid")
    body = dict(value)
    body.pop("evidence_hash", None)
    if sha256_json(body) != declared:
        raise ValueError("PG-133 evidence hash mismatch")
    return declared


def _payload(role: str, phase: str, marker: str) -> dict[str, Any]:
    return {"role": role, "probe_phase": phase, "marker": marker}


def _oracle(surface: Surface, response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool, str]:
    target_evidence = response.get("evidence")
    if not isinstance(target_evidence, Mapping):
        raise ValueError("PG-133 target omitted evidence")
    target_hash = _verify_hash(target_evidence)
    evaluator = target_evidence.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("PG-133 target omitted evaluator")
    available = bool(evaluator.get("typed_available")) and surface.typed_available
    candidate = bool(evaluator.get("candidate_signal"))
    workflow_action = str(evaluator.get("workflow_action", ""))
    if workflow_action not in {"repeat_matched_negative_pair", "probe_candidate_other_method", "abstain_unknown_oracle"}:
        raise ValueError("PG-133 evaluator emitted an unsafe workflow action")
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
    authority = {"history_stage": str(evaluator.get("history_stage", "unknown")), "workflow_action": workflow_action, "methods_seen": [str(item).upper() for item in evaluator.get("methods_seen", []) if str(item).upper() in {"GET", "POST"}], "typed_available": available, "source_evidence_sha256": target_hash}
    return oracle, authority, available, target_hash


async def _action(
    client: httpx.AsyncClient,
    surface: Surface,
    *,
    seed: int,
    decoy_strength: int,
    method: str,
    role: str,
    episode_variant: str,
    step_index: int,
    marker: str,
) -> dict[str, Any]:
    payload = _payload(role, "confirm" if role == "candidate" else "shadow", marker)
    route = f"/v13/observe/{surface.kind}"
    if method == "GET":
        response = await client.get(route, params=payload)
        placement = "query"
    else:
        response = await client.post(route, json=payload)
        placement = "json"
    if response.status_code != 200:
        raise RuntimeError(f"PG-133 target returned {response.status_code}")
    body = response.json()
    oracle, authority, available, target_hash = _oracle(surface, body)
    target_projection = dict(body.get("response_projection") or {})
    # The hidden history stage is not copied into this projection.
    response_projection = {**target_projection, "response_projection_sha256": sha256_json(target_projection), "candidate_signal": bool(body.get("candidate_signal"))}
    request_hash = sha256_json({"surface": surface.kind, "method": method, "role": role, "episode_variant": episode_variant, "step_index": step_index, "decoy_strength": decoy_strength, "payload_keys": sorted(payload.keys()), "marker_sha256": _hash_text(marker)})
    target_id = f"pg133-latch-target-{seed}-d{decoy_strength}"
    reset = client  # reset evidence is attached by collect_episode below
    action_manifest = {"method": method, "route_template_id": f"pg133-latch-route-{_hash_text(route)[:16]}", "placement": placement, "encoding_chain": ["url_percent", "unicode_escape"], "probe_ref": f"pg133-safe-workflow-probe-{_hash_text(surface.kind + role)[:12]}", "probe_sha256": request_hash, "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action_manifest["form_field_names"] = ["role", "probe_phase", "marker"]
    bridge_evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "target_instance_id": target_id,
        "surface": surface.kind,
        "method": method,
        "role": role,
        "episode_variant": episode_variant,
        "step_index": step_index,
        "request_sha256": request_hash,
        "target_evidence_sha256": target_hash,
        "response_projection": dict(response_projection),
        "oracle_projection": dict(oracle),
        "workflow_oracle": authority,
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False},
    }
    bridge_evidence["evidence_hash"] = sha256_json(bridge_evidence)
    return {"seed": seed, "decoy_strength": decoy_strength, "target_instance_id": target_id, "surface": surface.kind, "method": method, "role": role, "episode_variant": episode_variant, "step_index": step_index, "candidate_signal": bool(oracle["candidate_signal"]), "positive": False, "positive_authority": False, "typed_available": available, "oracle_available": surface.typed_available, "workflow_authority": authority, "action_manifest": action_manifest, "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095", "shape_class": "decision-v8"}, "response_projection": response_projection, "oracle_projection": oracle, "bridge_evidence": bridge_evidence, "reset_context": reset}


def _likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    return {"effect": 0.02, "input_only": 0.12 if record["candidate_signal"] else 0.04, "no_effect": 0.78 if not record["candidate_signal"] else 0.18, "unknown": 0.68 if record["candidate_signal"] and not record["typed_available"] else 0.08}


def _step(record: Mapping[str, Any], *, episode_id: str, index: int, parent: str | None, before: Mapping[str, float], after: Mapping[str, float], pair_id: str | None, prior_records: list[Mapping[str, Any]], fresh_reset: Mapping[str, Any]) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if pair_id:
        oracle["negative_control_pair_id"] = pair_id
    failure = failure_signature(record, prior_records=prior_records, max_steps=MAX_STEPS, step_count=index)
    # The evaluator-only workflow oracle is the label authority for this
    # counterfactual task; no history field is copied into the failure input.
    failure["next_action"] = str(record["workflow_authority"]["workflow_action"])
    decision = "confirmed_negative" if record["role"] == "control" else "candidate"
    body = {"action_manifest": record["action_manifest"], "baseline_projection": record["baseline_projection"], "response_projection": record["response_projection"], "oracle_projection": oracle, "belief_before": dict(before), "belief_after": dict(after), "decision": decision, "next_action": failure["next_action"], "failure_signature": failure}
    return validate_trace_step({"episode_id": episode_id, "step_id": f"{episode_id}-s{index:02d}", "parent_step_id": parent, "sampling_seed": record["seed"], "target_instance_id": record["target_instance_id"], "hypothesis": "history_sensitive_safe_workflow", **body, "fresh_reset": fresh_reset, "evidence_sha256": record["bridge_evidence"]["evidence_hash"], "dataset_stage": "pg133_history_counterfactual_replay", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": sha256_json(body)}})


async def collect_episode(client: httpx.AsyncClient, surface: Surface, *, seed: int, decoy_strength: int, episode_variant: str) -> dict[str, Any]:
    episode_id = f"pg133-latch-{seed}-d{decoy_strength}-{surface.kind}-{episode_variant}"
    reset_response = await client.post("/v13/reset")
    reset = reset_response.json().get("reset")
    if reset_response.status_code != 200 or not isinstance(reset, Mapping):
        raise RuntimeError("PG-133 fresh reset failed")
    reset_hash = _verify_hash(reset)
    if not (reset["fresh_target"] and reset["completed"] and reset["evaluator_state_hidden"] and reset["state_change_allowed"] is False):
        raise ValueError("PG-133 reset contract failed")
    fresh_reset = {**dict(reset), "reset_evidence_sha256": reset_hash}
    marker = f"pg133{seed}{decoy_strength}{surface.kind[:2]}{episode_variant[:2]}"
    records: list[dict[str, Any]] = []
    for index, (role, method) in enumerate(ACTION_PLANS[episode_variant], start=1):
        record = await _action(client, surface, seed=seed, decoy_strength=decoy_strength, method=method, role=role, episode_variant=episode_variant, step_index=index, marker=marker)
        record["bridge_evidence"]["reset_evidence_sha256"] = reset_hash
        record["bridge_evidence"]["fresh_reset"] = fresh_reset
        record["bridge_evidence"]["evidence_hash"] = sha256_json({key: value for key, value in record["bridge_evidence"].items() if key != "evidence_hash"})
        records.append(record)
    controls_by_method = {record["method"]: record for record in records if record["role"] == "control"}
    belief = GenericBeliefState()
    steps: list[dict[str, Any]] = []
    parent: str | None = None
    prior_records: list[Mapping[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prior = dict(belief.posterior)
        posterior = dict(belief.observe(f"{episode_id}-belief-{index}", _likelihood(record), evidence_hash=record["bridge_evidence"]["evidence_hash"])["posterior"])
        pair = controls_by_method.get(record["method"]) if record["role"] == "candidate" else None
        current = _step(record, episode_id=episode_id, index=index, parent=parent, before=prior, after=posterior, pair_id=pair["bridge_evidence"]["evidence_hash"] if pair else None, prior_records=prior_records, fresh_reset=fresh_reset)
        steps.append(current)
        parent = current["step_id"]
        prior_records.append(record)
    current = records[1]
    # Pair identity intentionally excludes probe hashes/variant labels.  The
    # two episodes must share the same current observation while differing
    # only in their prior history.
    pair_id = sha256_json({"surface": surface.kind, "method": current["method"], "response_projection": current["response_projection"], "route_template_id": current["action_manifest"]["route_template_id"], "placement": current["action_manifest"]["placement"]})
    binding = {"slot_id": f"pg133-history-latch-slot-{_hash_text(surface.kind)[:16]}", "binding_stage": "after_evaluator_only_history_workflow_oracle", "decision": "abstain", "evidence_sha256": steps[-1]["evidence_sha256"], "shadow_probe_evidence_sha256": [step["evidence_sha256"] for step in steps], "typed_oracle_available": surface.typed_available, "positive_authority": False, "failure_signatures_recorded": True, "counterfactual_pair_id": pair_id, "long_term_memory_write": False}
    return {"episode_id": episode_id, "target_instance_id": f"pg133-latch-target-{seed}-d{decoy_strength}", "target_seed": seed, "decoy_strength": decoy_strength, "surface_kind": surface.kind, "episode_variant": episode_variant, "oracle_available": surface.typed_available, "steps": steps, "evidence_records": [record["bridge_evidence"] for record in records], "episode_report": evaluate_episode(steps), "final_decision": "abstain", "negative_control_pair_clear": bool(controls_by_method), "counterfactual_pair_id": pair_id, "history_authority": {"current_step": f"{episode_id}-s02", "history_stage": current["workflow_authority"]["history_stage"], "workflow_action": current["workflow_authority"]["workflow_action"], "source_evidence_sha256": current["workflow_authority"]["source_evidence_sha256"]}, "rule_ir_slot_binding": binding}


async def collect_target(seed: int, *, decoy_strength: int = 1) -> dict[str, Any]:
    from .pg133_history_latch_target import create_app

    app = create_app(seed, decoy_strength=decoy_strength)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        health = await client.get("/healthz")
        if health.status_code != 200 or health.json().get("implementation") != "pg133-history-latch-target":
            raise RuntimeError("PG-133 target identity failed")
        episodes = [await collect_episode(client, surface, seed=seed, decoy_strength=decoy_strength, episode_variant=variant) for surface in SURFACES for variant in ACTION_PLANS]
    return {"target_seed": seed, "decoy_strength": decoy_strength, "target_instance_id": f"pg133-latch-target-{seed}-d{decoy_strength}", "target_implementation": "pg133-history-latch-target", "target_schema_version": health.json().get("schema_version"), "episodes": episodes}


__all__ = ["ACTION_PLANS", "MAX_STEPS", "SCHEMA_VERSION", "SURFACES", "collect_episode", "collect_target"]
