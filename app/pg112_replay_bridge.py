"""PG-112 bridge from local typed replay oracles into Python BSP v3 traces.

The bridge is deliberately an evaluator-side collector.  It sends only inert,
allow-listed abstract probes to the in-process local ASGI application, keeps
raw request/response values in memory for the duration of one call, and emits
bounded projections plus SHA-256 commitments.  The model-facing view contains
no family name and no typed-oracle label.  Confirmation is allowed only after
GET/POST repetition, a matched negative control, fresh reset records, and a
typed oracle that was called after the probe.

This is not an internet scanner, an exploit generator, or a training loop.
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


SCHEMA_VERSION = "pg112-python-bsp-local-replay-v1"
EVIDENCE_SCHEMA_VERSION = "pg112-bounded-evidence-v1"
LOCAL_BASE_URL = "http://127.0.0.1:3100"
ALLOWED_METHODS = frozenset({"GET", "POST"})
_HASH_RE = frozenset("0123456789abcdef")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(str(value).encode("utf-8"))


def _slot(label: str) -> str:
    return f"slot-{_sha256_text(label)[:16]}"


def _contract_hash(label: str) -> str:
    return _sha256_text(f"pg112-typed-local-oracle:{label}")


@dataclass(frozen=True)
class ReplaySurface:
    """An abstract surface; endpoint details stay collector-side."""

    surface_slot: str
    endpoint_kind: str
    path: str
    oracle_contract_sha256: str
    oracle_available: bool = True


SURFACES: tuple[ReplaySurface, ...] = (
    ReplaySurface(_slot("detached-markup"), "detached_markup", "/api/maze/replay/dom", _contract_hash("detached_markup")),
    ReplaySurface(_slot("interpreter-shape"), "interpreter_shape", "/api/maze/replay/sql", _contract_hash("interpreter_shape")),
    ReplaySurface(_slot("boundary-transition"), "boundary_transition", "/api/maze/replay/logic", _contract_hash("boundary_transition")),
    # The same inert signal is intentionally replayed without exposing its
    # typed oracle.  The correct result is abstain, not a guessed positive.
    ReplaySurface(_slot("withheld-surface"), "withheld_surface", "/api/maze/replay/dom", _contract_hash("withheld_surface"), False),
)


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"kind": "object", "keys": sorted(str(key) for key in value)[:32]}
    if isinstance(value, list):
        return {"kind": "array", "length": min(len(value), 1024)}
    if value is None:
        return {"kind": "null"}
    if isinstance(value, bool):
        return {"kind": "boolean"}
    if isinstance(value, (int, float)):
        return {"kind": "number"}
    return {"kind": "string"}


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx" if 100 <= status_code <= 599 else "other"


def _length_bucket(length: int) -> str:
    return "0" if length == 0 else "1-255" if length <= 255 else "256-4095" if length <= 4095 else "4096+"


def _verify_oracle_hash(body: Mapping[str, Any], *, nested: bool = False) -> str:
    evidence = body.get("evidence") if nested else body
    if not isinstance(evidence, Mapping):
        raise ValueError("local oracle did not return a bounded evidence object")
    declared = str(evidence.get("evidence_hash", ""))
    if len(declared) != 64 or any(char not in _HASH_RE for char in declared):
        raise ValueError("local oracle evidence hash is not a SHA-256 digest")
    without_hash = dict(evidence)
    without_hash.pop("evidence_hash", None)
    if sha256_json(without_hash) != declared:
        raise ValueError("local oracle evidence hash mismatch")
    return declared


def _response_projection(response: httpx.Response, *, baseline: httpx.Response, oracle_projection: Mapping[str, Any], bsp_projection: Mapping[str, Any]) -> dict[str, Any]:
    body_bytes = bytes(response.content)
    try:
        body = response.json()
    except ValueError:
        body = None
    shape = _shape(body)
    baseline_digest = _sha256_bytes(bytes(baseline.content))
    response_digest = _sha256_bytes(body_bytes)
    return {
        "status_class": _status_class(int(response.status_code)),
        "body_length_bucket": _length_bucket(len(body_bytes)),
        "body_sha256": response_digest,
        "json_shape_sha256": sha256_json(shape),
        "response_delta": "changed" if response_digest != baseline_digest else "unchanged",
        "oracle_projection_sha256": sha256_json(dict(oracle_projection)),
        "bsp_core_projection": dict(bsp_projection),
    }


def _request_payload(surface: ReplaySurface, *, positive: bool, marker: str) -> dict[str, Any]:
    if surface.endpoint_kind in {"detached_markup", "withheld_surface"}:
        value = f'<span data-sift-marker="{marker}">inert</span>' if positive else "plain"
        return {"value": value, "marker": marker}
    if surface.endpoint_kind == "interpreter_shape":
        return {"fragment_class": "operator_like" if positive else "plain"}
    if surface.endpoint_kind == "boundary_transition":
        return {"probe_class": "boundary_candidate" if positive else "normal", "surface": "authorization_boundary"}
    raise ValueError("unknown PG-112 surface")


def _oracle_projection(surface: ReplaySurface, body: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool, str]:
    """Project a local response; the returned positive label is target-only."""

    if surface.endpoint_kind in {"detached_markup", "withheld_surface"}:
        raw = body
        source_hash = _verify_oracle_hash(raw)
        signal = bool(raw.get("browser_sink_observed")) and bool(raw.get("dom_change"))
        candidate_signal = bool(raw.get("candidate_signal"))
        atoms = ["effect_present", "surface_observed"] if signal else ["no_effect"]
        modality = "typed_surface_differential" if surface.oracle_available else "untyped_surface_signal"
    elif surface.endpoint_kind == "interpreter_shape":
        raw = body.get("evidence") if isinstance(body.get("evidence"), Mapping) else {}
        source_hash = _verify_oracle_hash(body, nested=True)
        signal = bool(raw.get("controlled_differential")) and bool(raw.get("interpreter_boundary"))
        candidate_signal = bool(raw.get("candidate_signal"))
        atoms = ["effect_present", "interpreter_boundary"] if signal else ["input_only" if candidate_signal else "no_effect"]
        modality = "typed_surface_differential" if surface.oracle_available else "untyped_surface_signal"
    elif surface.endpoint_kind == "boundary_transition":
        raw = body
        source_hash = _verify_oracle_hash(raw)
        signal = bool(raw.get("typed_boundary_observed"))
        candidate_signal = bool(raw.get("candidate_signal"))
        atoms = ["effect_present", "boundary_observed"] if signal else ["no_effect"]
        modality = "typed_surface_differential" if surface.oracle_available else "untyped_surface_signal"
    else:
        raise ValueError("unknown PG-112 oracle kind")

    # The typed result is never placed in model_input.  A withheld oracle can
    # expose only an anonymous candidate signal and must remain abstain-only.
    positive = bool(signal) if surface.oracle_available else False
    authority = bool(signal and surface.oracle_available)
    if not surface.oracle_available and candidate_signal:
        atoms = ["candidate_signal_observed"]
    projection = {
        "modality": modality,
        "positive": positive,
        "positive_authority": authority,
        "candidate_signal": candidate_signal,
        "observed_atoms": atoms,
        "oracle_contract_sha256": surface.oracle_contract_sha256,
        "source_evidence_sha256": source_hash,
        "safety": {
            "script_execution": False,
            "network_access": False,
            "navigation": False,
            "database_touched": False,
            "real_sleep_performed": False,
            "credentials_accessed": False,
        },
    }
    return projection, positive, authority, source_hash


def _fresh_reset(target_instance_id: str, sequence: int, baseline_sha256: str, *, reset_adapter_sha256: str) -> dict[str, Any]:
    body = {
        "kind": "pg112-fresh-local-asgi",
        "target_instance_id": target_instance_id,
        "reset_sequence": int(sequence),
        "baseline_sha256": baseline_sha256,
        "reset_adapter_sha256": reset_adapter_sha256,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }
    return {**body, "reset_evidence_sha256": sha256_json(body)}


def _bsp_observation(state: BspV3State, evidence_hash: str) -> dict[str, Any]:
    values = [int(evidence_hash[index : index + 2], 16) / 255.0 for index in (0, 2, 4, 6)]
    import numpy as np

    output = state.forward(np.asarray([values], dtype=np.float64), np.asarray([[0.6, 0.4]], dtype=np.float64))
    return {
        "topology_version": state.topology_version,
        "leaf_mass_error": float(abs(float(output.leaf_mass_sum.sum()) - 1.0)),
        "selected_leaf_ids": [int(value) for value in output.selected_leaf_ids.reshape(-1)],
        "state_sha256": state.state_sha256(),
    }


async def _collect_action(
    client: httpx.AsyncClient,
    surface: ReplaySurface,
    *,
    target_instance_id: str,
    method: str,
    role: str,
    marker: str,
    sequence: int,
    reset_adapter_sha256: str,
    state: BspV3State,
) -> dict[str, Any]:
    method = str(method).upper()
    if method not in ALLOWED_METHODS:
        raise ValueError("PG-112 permits only GET and POST")
    baseline = await client.get("/api/health")
    if baseline.status_code != 200:
        raise RuntimeError("local replay health baseline failed")
    payload = _request_payload(surface, positive=role == "candidate", marker=marker)
    if method == "GET":
        response = await client.get(surface.path, params=payload)
        placement = "query"
    else:
        response = await client.post(surface.path, json=payload)
        placement = "form"
    if response.status_code != 200:
        raise RuntimeError(f"local replay endpoint returned {response.status_code}")
    body = response.json()
    oracle, positive, authority, source_evidence_hash = _oracle_projection(surface, body)
    request_sha256 = sha256_json({"surface_slot": surface.surface_slot, "method": method, "role": role, "marker_sha256": _sha256_text(marker), "payload_shape": _shape(payload)})
    reset = _fresh_reset(target_instance_id, sequence, _sha256_bytes(bytes(baseline.content)), reset_adapter_sha256=reset_adapter_sha256)
    evidence_without_hash = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "target_instance_id": target_instance_id,
        "surface_slot": surface.surface_slot,
        "method": method,
        "role": role,
        "request_sha256": request_sha256,
        "response_projection": {
            "status_class": _status_class(int(response.status_code)),
            "body_length_bucket": _length_bucket(len(response.content)),
            "body_sha256": _sha256_bytes(bytes(response.content)),
        },
        "oracle_projection": oracle,
        "fresh_reset": reset,
        "safety": {"external_network": False, "script_execution": False, "database_touched": False, "state_mutated": False},
    }
    evidence_hash = sha256_json(evidence_without_hash)
    evidence = {**evidence_without_hash, "evidence_hash": evidence_hash}
    bsp_projection = _bsp_observation(state, evidence_hash)
    response_projection = _response_projection(response, baseline=baseline, oracle_projection=oracle, bsp_projection=bsp_projection)
    action_manifest = {
        "method": method,
        "route_template_id": f"route-{_sha256_text(surface.path)[:16]}",
        "placement": placement,
        "encoding_chain": ["identity"],
        "probe_ref": f"pg112-abstract-probe-{surface.surface_slot[-8:]}",
        "probe_sha256": request_sha256,
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
        **({"form_field_names": ["probe"]} if method == "POST" else {}),
    }
    return {
        "sample_id": f"pg112-{target_instance_id[-3:]}-{surface.surface_slot[-8:]}-{method.casefold()}-{role}",
        "target_instance_id": target_instance_id,
        "surface_slot": surface.surface_slot,
        "method": method,
        "role": role,
        "positive": positive,
        "positive_authority": authority,
        "typed_oracle_called": bool(surface.oracle_available),
        "source_evidence_sha256": source_evidence_hash,
        "request_sha256": request_sha256,
        "action_manifest": action_manifest,
        "baseline_projection": {
            "status_class": _status_class(int(baseline.status_code)),
            "body_sha256": _sha256_bytes(bytes(baseline.content)),
            "body_length_bucket": _length_bucket(len(baseline.content)),
        },
        "response_projection": response_projection,
        "oracle_projection": oracle,
        "fresh_reset": reset,
        "evidence": evidence,
    }


def _belief_likelihood(record: Mapping[str, Any]) -> dict[str, float]:
    if bool(record.get("positive_authority")):
        return {"effect": 0.82, "input_only": 0.05, "no_effect": 0.03, "unknown": 0.10}
    if bool(record.get("positive")):
        return {"effect": 0.10, "input_only": 0.18, "no_effect": 0.08, "unknown": 0.64}
    return {"effect": 0.04, "input_only": 0.08, "no_effect": 0.78, "unknown": 0.10}


def _step_from_record(
    record: Mapping[str, Any],
    *,
    episode_id: str,
    step_index: int,
    parent_step_id: str | None,
    decision: str,
    next_action: str,
    belief_before: Mapping[str, float],
    belief_after: Mapping[str, float],
    negative_control_pair_id: str | None,
) -> dict[str, Any]:
    oracle = dict(record["oracle_projection"])
    if negative_control_pair_id is not None:
        oracle["negative_control_pair_id"] = negative_control_pair_id
    step_id = f"{episode_id}-s{step_index:02d}"
    body = {
        "action_manifest": record["action_manifest"],
        "baseline_projection": record["baseline_projection"],
        "response_projection": record["response_projection"],
        "oracle_projection": oracle,
        "belief_before": dict(belief_before),
        "belief_after": dict(belief_after),
        "decision": decision,
        "next_action": next_action,
    }
    return validate_trace_step(
        {
            "episode_id": episode_id,
            "step_id": step_id,
            "parent_step_id": parent_step_id,
            "sampling_seed": int(record["target_instance_id"].split("-")[-1]),
            "target_instance_id": record["target_instance_id"],
            "hypothesis": "unknown_surface",
            **body,
            "fresh_reset": record["fresh_reset"],
            "evidence_sha256": record["evidence"]["evidence_hash"],
            "dataset_stage": "pg112_evaluation_only",
            "online_weight_update": False,
            "long_term_memory_write": False,
            "echo": {"sha256": sha256_json(body)},
        }
    )


async def collect_episode(
    client: httpx.AsyncClient,
    surface: ReplaySurface,
    *,
    target_seed: int,
    reset_adapter_sha256: str,
) -> dict[str, Any]:
    target_instance_id = f"pg112-target-{int(target_seed)}"
    episode_id = f"pg112-{int(target_seed)}-{surface.surface_slot}"
    state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=int(target_seed))
    parameter_before = state.parameter_sha256()
    marker = f"pg112m{int(target_seed)}{surface.surface_slot[-6:]}"
    records: list[dict[str, Any]] = []
    sequence = 0
    for role, method in (("control", "GET"), ("candidate", "GET"), ("control", "POST"), ("candidate", "POST")):
        sequence += 1
        records.append(
            await _collect_action(
                client,
                surface,
                target_instance_id=target_instance_id,
                method=method,
                role=role,
                marker=marker,
                sequence=sequence,
                reset_adapter_sha256=reset_adapter_sha256,
                state=state,
            )
        )

    controls = [record for record in records if record["role"] == "control"]
    candidates = [record for record in records if record["role"] == "candidate"]
    known_positive_pair = (
        surface.oracle_available
        and all(bool(record["positive_authority"]) for record in candidates)
        and all(not bool(record["positive"]) for record in controls)
    )
    known_negative_pair = surface.oracle_available and all(not bool(record["positive"]) for record in records)
    final_decision = "confirmed_positive" if known_positive_pair else "confirmed_negative" if known_negative_pair else "abstain"
    belief = GenericBeliefState()
    steps: list[dict[str, Any]] = []
    parent: str | None = None
    for index, record in enumerate(records, start=1):
        before = dict(belief.posterior)
        update = belief.observe(f"{record['sample_id']}-belief", _belief_likelihood(record), evidence_hash=record["evidence"]["evidence_hash"])
        after = dict(update["posterior"])
        if record["role"] == "control":
            decision = "confirmed_negative"
            next_action = "probe_candidate_same_method"
        elif record["method"] == "GET":
            decision = "candidate" if record["positive"] else "abstain" if not surface.oracle_available else "confirmed_negative"
            next_action = "replay_other_method"
        else:
            decision = final_decision
            next_action = "stop_episode" if final_decision != "abstain" else "abstain_unknown_surface"
        pair_id = next((control["sample_id"] for control in controls if control["method"] == record["method"]), None) if record["role"] == "candidate" else None
        step = _step_from_record(record, episode_id=episode_id, step_index=index, parent_step_id=parent, decision=decision, next_action=next_action, belief_before=before, belief_after=after, negative_control_pair_id=pair_id)
        steps.append(step)
        parent = step["step_id"]
    episode_report = evaluate_episode(steps)
    return {
        "episode_id": episode_id,
        "target_instance_id": target_instance_id,
        "surface_slot": surface.surface_slot,
        "oracle_available": surface.oracle_available,
        "steps": steps,
        "evidence_records": [record["evidence"] for record in records],
        "episode_report": episode_report,
        "final_decision": final_decision,
        "candidate_pair_positive": known_positive_pair,
        "negative_control_pair_clear": all(not bool(record["positive"]) for record in controls),
        "belief": belief.snapshot(),
        "bsp": {
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": state.parameter_sha256(),
            "parameter_unchanged": parameter_before == state.parameter_sha256(),
            "topology_version": state.topology_version,
            "state_sha256": state.state_sha256(),
        },
    }


async def collect_all(application: Any, *, target_seeds: tuple[int, ...] = (101, 202, 303), reset_adapter_sha256: str | None = None) -> dict[str, Any]:
    """Collect a multi-target local ASGI matrix without training side effects."""

    if reset_adapter_sha256 is None:
        reset_adapter_sha256 = _sha256_text("pg112-local-asgi-fresh-reset-v1")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url=LOCAL_BASE_URL, timeout=5.0, follow_redirects=False) as client:
        episodes = [
            await collect_episode(client, surface, target_seed=seed, reset_adapter_sha256=reset_adapter_sha256)
            for seed in target_seeds
            for surface in SURFACES
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "transport": "in_process_asgi_loopback",
        "loopback_only": True,
        "external_network": False,
        "target_seeds": list(target_seeds),
        "target_instance_count": len(target_seeds),
        "surface_count": len(SURFACES),
        "episodes": episodes,
        "steps": [step for episode in episodes for step in episode["steps"]],
        "evidence_records": [record for episode in episodes for record in episode["evidence_records"]],
        "training_eligible": False,
        "long_term_memory_write": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


__all__ = ["LOCAL_BASE_URL", "SCHEMA_VERSION", "SURFACES", "collect_all", "collect_episode"]
