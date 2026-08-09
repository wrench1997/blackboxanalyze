"""PG-331 Juice Shop source-row planning and evaluator binding contract.

This module is intentionally a *planning-only* adapter.  It does not import
Docker, HTTP clients, browser libraries, torch, or training code, and it never
contacts a target.  A later operator-approved live runner can consume the
plan and use the PG-324 loopback transport, but this file only describes the
allowlisted lanes and binds already-observed abstract projections.

The plan covers three frozen seeds, the six PG-324 Juice Shop GET/POST lanes,
and four fresh identities per route (candidate/reference/negative/replay).
Each episode carries an explicit seven-axis/107-field ``not_observed``
manifest and a safe ASK projection.  Consequently the plan is
``planning_only`` and can never be mistaken for a collected or
training-eligible dataset.

Literal probes, wire requests, response bytes, route paths, family labels and
browser/oracle answers are evaluator-side concerns.  ``build_evaluator_binding``
accepts optional literal probes only to derive a role-bound SHA-256 in memory;
the returned binding contains no literal.  ``bind_source_row`` is a pure
adapter for a future loopback observation and delegates schema enforcement to
``app.pg331_source_row`` with operator review disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import (  # noqa: E402
    build_pg331_evaluator_record,
    sha256_json,
    validate_pg331_evaluator_record,
)
from app.pg331_source_row import collect_pg331_source_row  # noqa: E402


SCHEMA_VERSION = "pg331-juice-shop-source-row-plan-v1"
IMAGE = "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe"
SAFETY_MODE_CONFIG = '{"challenges":{"safetyMode":"disabled"}}'
SAFETY_MODE_CONFIG_SHA256 = hashlib.sha256(SAFETY_MODE_CONFIG.encode("utf-8")).hexdigest()
SEEDS = (31901, 31902, 31903)
ROLES = ("candidate", "reference", "negative", "replay")
EVIDENCE_ROLES = ("candidate", "reference", "negative")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_FRAGMENTS = ("payload", "response_body", "raw_", "oracle_answer", "evaluator_answer", "wire")
SAFE_KEYS = frozenset(
    {
        "raw_payload_stored",
        "raw_response_stored",
        "raw_response_body_stored",
        "oracle_answer_in_context",
        "payload_catalog_promotion_allowed",
        "raw_wire_off_context",
        "raw_response_off_context",
        "oracle_answer_off_context",
        "raw_fields_omitted",
    }
)

# Paths and families remain in this module's evaluator memory.  Plans expose
# route IDs and one-way path attestations only; no route path enters model
# context, source rows or the plan JSON.
ROUTES: tuple[dict[str, Any], ...] = (
    {
        "id": "juice-track-order-xss-get",
        "path": "/rest/track-order/{id}",
        "method": "GET",
        "value_field": "id",
        "style": "juice_track",
        "family": "xss",
        "expected_lane": "positive",
        "post_supported": False,
    },
    {
        "id": "juice-track-order-safe-get",
        "path": "/rest/track-order/{id}",
        "method": "GET",
        "value_field": "id",
        "style": "juice_track",
        "family": "xss",
        "expected_lane": "negative",
        "post_supported": False,
    },
    {
        "id": "juice-products-search-get",
        "path": "/rest/products/search",
        "method": "GET",
        "value_field": "q",
        "style": "json",
        "family": "xss",
        "expected_lane": "negative",
        "post_supported": False,
    },
    {
        "id": "juice-track-order-xss-post-unsupported",
        "path": "/rest/track-order/{id}",
        "method": "POST",
        "value_field": "id",
        "style": "juice_track",
        "family": "xss",
        "expected_lane": "unsupported_post",
        "post_supported": False,
    },
    {
        "id": "juice-products-search-post-unsupported",
        "path": "/rest/products/search",
        "method": "POST",
        "value_field": "q",
        "style": "json",
        "family": "xss",
        "expected_lane": "unsupported_post",
        "post_supported": False,
    },
    {
        "id": "juice-login-post-unsupported",
        "path": "/rest/user/login",
        "method": "POST",
        "value_field": "email",
        "style": "json",
        "family": "authentication",
        "expected_lane": "unsupported_post",
        "post_supported": False,
    },
)
_ROUTE_BY_ID = {str(route["id"]): route for route in ROUTES}


def _contains_raw(value: Any, key: str = "") -> bool:
    lowered = str(key).casefold()
    if lowered in SAFE_KEYS:
        return False
    if any(fragment in lowered for fragment in RAW_FRAGMENTS):
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_raw(child, key) for child in value)
    return False


def _route(route_id: str) -> dict[str, Any]:
    try:
        return dict(_ROUTE_BY_ID[str(route_id)])
    except KeyError as error:
        raise ValueError(f"PG-331 Juice Shop route is not allowlisted: {route_id}") from error


def _route_attestation(route: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "id": str(route["id"]),
                "path": str(route["path"]),
                "method": str(route["method"]),
                "value_field": str(route["value_field"]),
                "image": IMAGE,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _container_identity(*, seed: int, route: Mapping[str, Any], role: str) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": int(seed),
            "route_attestation_sha256": _route_attestation(route),
            "role": str(role),
            "image": IMAGE,
            "network_mode": "none",
            "loopback_only": True,
            "fresh_reset": True,
        }
    )


@lru_cache(maxsize=1)
def _ontology() -> dict[str, Any]:
    path = ROOT / "research" / "pg331_web_token_ontology_v1.json"
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("PG-331 ontology must be an object")
    return value


def _field_manifest(status: str = "not_observed") -> dict[str, dict[str, str]]:
    if status not in {"observed", "absent", "not_observed", "unknown"}:
        raise ValueError(f"invalid PG-331 field status: {status}")
    result: dict[str, dict[str, str]] = {}
    for axis, spec in dict(_ontology().get("axes") or {}).items():
        fields = list(spec.get("fields") or []) if isinstance(spec, Mapping) else []
        result[str(axis)] = {str(field): status for field in fields}
    return result


def _axis_presence(status: str = "not_observed") -> dict[str, str]:
    result: dict[str, str] = {}
    for axis, spec in dict(_ontology().get("axes") or {}).items():
        presence = str(spec.get("presence_token") or f"{axis}_presence") if isinstance(spec, Mapping) else f"{axis}_presence"
        result[presence] = status
    return result


def _ontology_counts() -> dict[str, int]:
    axes = dict(_ontology().get("axes") or {})
    return {"axis_count": len(axes), "field_count": sum(len(list(spec.get("fields") or [])) for spec in axes.values() if isinstance(spec, Mapping))}


def _ask_projection(route: Mapping[str, Any]) -> dict[str, Any]:
    # All plan observations are uncollected.  POST is additionally known to
    # be unsupported by the PG-324 route contract, so it must remain ASK-only.
    return {
        "method": str(route["method"]),
        "post_supported": bool(route.get("post_supported")),
        "typed_available": "unknown",
        "evidence_present": "unknown",
        "negative_control": "unknown",
        "fresh_reset": "unknown",
        "replay_ready": "unknown",
        "next_action": "ask_typed" if str(route["method"]) == "POST" or not bool(route.get("post_supported", True)) else "ask",
        "safe_to_send": False,
    }


def _role_plan(*, seed: int, route: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": role,
        "container_identity_sha256": _container_identity(seed=seed, route=route, role=role),
        "fresh_reset_required": True,
        "fresh_reset_observed": False,
        "network_mode": "none",
        "loopback_only": True,
        "external_network": False,
        "host_port_published": False,
        "zero_bind_or_volume_mounts_required": True,
        "source_row_allowed": role in EVIDENCE_ROLES,
        "evaluator_only": role == "replay" or role in EVIDENCE_ROLES,
    }


def _episode(*, seed: int, route: Mapping[str, Any]) -> dict[str, Any]:
    counts = _ontology_counts()
    route_ref = _route_attestation(route)
    return {
        "seed": int(seed),
        "route_id": str(route["id"]),
        "route_ref_sha256": route_ref,
        "method": str(route["method"]),
        "lane": "unsupported_post_ask" if str(route["method"]) == "POST" and not bool(route.get("post_supported")) else "get_baseline_then_typed",
        "post_supported": bool(route.get("post_supported")),
        "roles": {role: _role_plan(seed=seed, route=route, role=role) for role in ROLES},
        "observation_contract": {
            "required_axis_count": counts["axis_count"],
            "required_field_count": counts["field_count"],
            "axis_presence": _axis_presence("not_observed"),
            "field_capture_manifest": _field_manifest("not_observed"),
            "manifest_status": "not_observed_until_live_adapter",
        },
        "model_context_projection": _ask_projection(route),
        "evaluator_contract": {
            "candidate_reference_negative_required": True,
            "replay_required": True,
            "role_bound_evidence_sha256_required": True,
            "raw_wire_off_context": True,
            "raw_response_off_context": True,
            "oracle_answer_off_context": True,
        },
        "source_row_status": "incomplete_ask",
        "training_eligible": False,
    }


def build_pg331_juice_shop_source_plan(*, seeds: Sequence[int] = SEEDS, route_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Build a pure, no-I/O fresh collection plan for the Juice Shop lanes."""

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("PG-331 Juice Shop plan requires at least one seed")
    requested_routes = tuple(str(route_id) for route_id in (route_ids if route_ids is not None else _ROUTE_BY_ID))
    if set(requested_routes) != set(_ROUTE_BY_ID) or len(requested_routes) != len(_ROUTE_BY_ID):
        raise ValueError("PG-331 Juice Shop plan requires exactly the six allowlisted GET/POST routes")
    episodes = [_episode(seed=seed, route=_route(route_id)) for seed in normalized_seeds for route_id in requested_routes]
    counts = _ontology_counts()
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only",
        "execution": {
            "real_execution": False,
            "docker_started": False,
            "network_contacted": False,
            "image": IMAGE,
            "network_mode": "none",
            "loopback_only": True,
            "external_network": False,
            "host_port_published": False,
            "safety_mode_override_sha256": SAFETY_MODE_CONFIG_SHA256,
        },
        "seeds": list(normalized_seeds),
        "route_count": len(requested_routes),
        "episode_count": len(episodes),
        "roles": list(ROLES),
        "source_roles": list(EVIDENCE_ROLES),
        "ontology_contract": {
            "axis_count": counts["axis_count"],
            "field_count": counts["field_count"],
            "required_statuses": ["observed", "absent", "not_observed", "unknown"],
            "missing_status_forces_ask": True,
        },
        "episodes": episodes,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": "planning_only: no target was contacted; all source-row manifests remain not_observed and must become ASK/incomplete until a fresh evaluator run supplies abstract observations.",
    }
    if _contains_raw(plan):
        raise ValueError("PG-331 Juice Shop plan contains raw material")
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def _role_with_probe(role: str, value: Mapping[str, Any], probe: Any | None, route_ref: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-331 Juice Shop {role} evaluator result must be an object")
    result = dict(value)
    # The probe is deliberately never passed to the sidecar.  Its only use is
    # a role/route-bound source digest while it remains in evaluator memory.
    if probe is not None:
        bounded = {str(key): child for key, child in result.items() if str(key) not in SAFE_KEYS and not any(fragment in str(key).casefold() for fragment in RAW_FRAGMENTS)}
        result["evidence_sha256"] = sha256_json({"role": role, "route_ref_sha256": route_ref, "probe": probe, "projection": bounded})
    return result


def build_evaluator_binding(
    *,
    seed: int,
    route_id: str,
    reset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    literal_probes: Mapping[str, Any] | None = None,
    replay_consistent: bool = False,
    reference_agreement: bool | None = None,
    negative_control_clean: bool | None = None,
    replay_evidence_sha256: str | None = None,
    evaluator_id: str = "pg331-juice-shop-evaluator-v1",
) -> dict[str, Any]:
    """Bind evaluator-side role projections without emitting literals."""

    route = _route(route_id)
    probes = dict(literal_probes or {})
    unknown = sorted(str(role) for role in probes if str(role) not in EVIDENCE_ROLES)
    if unknown:
        raise ValueError(f"PG-331 Juice Shop probes contain unsupported roles: {', '.join(unknown)}")
    route_ref = _route_attestation(route)
    roles = {
        role: _role_with_probe(role, value, probes.get(role), route_ref)
        for role, value in (("candidate", candidate), ("reference", reference), ("negative", negative))
    }
    evaluator_record = build_pg331_evaluator_record(
        record_id=f"pg331js:{int(seed)}:{route_ref[:16]}",
        reset=reset,
        candidate=roles["candidate"],
        reference=roles["reference"],
        negative=roles["negative"],
        replay_consistent=bool(replay_consistent),
        reference_agreement=reference_agreement,
        negative_control_clean=negative_control_clean,
        evaluator_id=evaluator_id,
    )
    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "binding_status": "complete" if evaluator_record["evaluator_sidecar"].get("confirmed_positive") else "incomplete_ask",
        "seed": int(seed),
        "route_projection": {
            "route_ref_sha256": route_ref,
            "method": str(route["method"]),
            "post_supported": bool(route.get("post_supported")),
        },
        "evaluator_record": evaluator_record,
        "evaluator_sidecar": evaluator_record["evaluator_sidecar"],
        "model_context": evaluator_record["model_context"],
        "replay": {
            "required": True,
            "consistent": bool(replay_consistent),
            "evidence_sha256": str(replay_evidence_sha256 or ""),
        },
        "raw_payload_stored": False,
        "raw_response_stored": False,
        "oracle_answer_in_context": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "training_eligible": False,
    }
    if binding["replay"]["evidence_sha256"] and not HASH_RE.fullmatch(binding["replay"]["evidence_sha256"]):
        raise ValueError("PG-331 Juice Shop replay evidence must be SHA-256")
    if _contains_raw(binding):
        raise ValueError("PG-331 Juice Shop evaluator binding contains raw material")
    binding["binding_sha256"] = sha256_json(binding)
    return binding


def _default_target(route: Mapping[str, Any], role: str) -> dict[str, Any]:
    if str(route["method"]) == "POST" and not bool(route.get("post_supported")):
        return {
            "question": "ask_typed",
            "next_action": "ask_typed",
            "repair_action": "observe",
            "transport_ref": "unknown",
            "field_role_ref": "unknown",
            "encoding_ref": "unknown",
            "probe_variant_ref": "none",
            "safe_to_send": False,
        }
    return {
        "question": "none",
        "next_action": "send_probe",
        "repair_action": "none",
        "transport_ref": "request_method",
        "field_role_ref": "parameter_role",
        "encoding_ref": "encoding_chain",
        "probe_variant_ref": "source_attested_candidate" if role == "candidate" else "reference" if role == "reference" else "negative_control",
        "safe_to_send": True,
    }


def _source_meta(*, seed: int, route: Mapping[str, Any], role: str, evidence_hash: str) -> dict[str, Any]:
    route_ref = _route_attestation(route)
    return {
        "source_id": "pg331-juice-shop-fixed-local",
        "implementation": "bkimminich-juice-shop",
        "collector_id": "pg331-juice-shop-source-row-plan-v1",
        "authorization_id": "operator-authorized-local-docker-loopback",
        "image_digest": IMAGE.split("@sha256:", 1)[1],
        "source_digest": sha256_json({"seed": int(seed), "route_ref_sha256": route_ref, "role": role, "evidence_sha256": evidence_hash}),
        "surface_id": f"surface_{route_ref[:16]}",
    }


def _normalize_reset_for_source_row(reset: Mapping[str, Any]) -> dict[str, Any]:
    """Project PG-324's richer reset attestation to the strict PG-331 keys.

    The live PG-324 ``_start`` contract uses ``fresh_target``,
    ``container_id_sha256`` and ``relay_loopback_only`` aliases, while the
    source-row collector deliberately accepts only its smaller schema.  This
    adapter drops no safety fact silently: nonzero bind/volume mounts or an
    explicitly restarted container fail closed before collection.
    """

    value = dict(reset)
    bind_count = int(value.get("volume_mount_count", value.get("bind_or_volume_mount_count", 0)) or 0)
    tmpfs_count = int(value.get("tmpfs_mount_count", 0) or 0)
    if bind_count != 0:
        raise ValueError("PG-331 Juice Shop reset has a forbidden bind/volume mount")
    if value.get("container_restart_used") is True:
        raise ValueError("PG-331 Juice Shop reset used a container restart")
    fresh = value.get("fresh_reset")
    if fresh is None:
        fresh = value.get("fresh_target")
    if fresh is None:
        fresh = value.get("completed")
    external = value.get("external_network")
    loopback = value.get("loopback_only")
    if loopback is None:
        loopback = value.get("relay_loopback_only")
    state_clean = value.get("state_clean")
    if state_clean is None:
        state_clean = value.get("domain_data_write_allowed") is False
    target_digest = value.get("target_instance_digest")
    if target_digest is None:
        target_digest = value.get("container_id_sha256")
    normalized: dict[str, Any] = {
        "fresh_reset": fresh,
        "reset_id": value.get("reset_id"),
        "target_instance_digest": target_digest,
        "network_mode": value.get("network_mode"),
        "external_network": external,
        "loopback_only": loopback,
        "state_clean": state_clean,
    }
    if "database_health_gate" in value:
        normalized["database_health_gate"] = value["database_health_gate"]
    # tmpfs is allowed and intentionally not copied into the strict row; the
    # live runner keeps it in its evaluator attestation instead.
    _ = tmpfs_count
    return normalized


def bind_source_row(
    *,
    seed: int,
    route_id: str,
    role: str,
    observation: Mapping[str, Any],
    reset: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    field_capture_manifest: Mapping[str, Any],
    target_projection: Mapping[str, Any] | None = None,
    split: str = "implementation_holdout",
) -> dict[str, Any]:
    """Bind a future live abstract observation to the strict source-row schema.

    This function accepts no URL, payload, response body or route literal.  A
    live adapter must provide an already-redacted seven-axis observation and a
    verified fresh reset/evaluator sidecar.  Operator review is intentionally
    false.  ``replay`` is evaluator-only and cannot become a source row.
    """

    route = _route(route_id)
    role = str(role)
    if role not in EVIDENCE_ROLES:
        raise ValueError("PG-331 Juice Shop source rows allow only candidate/reference/negative roles; replay is evaluator-only")
    if not isinstance(observation, Mapping) or not isinstance(reset, Mapping) or not isinstance(evaluator, Mapping) or not isinstance(field_capture_manifest, Mapping):
        raise ValueError("PG-331 Juice Shop source-row binding requires abstract observation, reset, evaluator and field manifest objects")
    target = dict(target_projection) if target_projection is not None else _default_target(route, role)
    if str(route["method"]) == "POST" and not bool(route.get("post_supported")) and target.get("safe_to_send") is True:
        raise ValueError("unsupported PG-331 Juice Shop POST lane must remain ASK-only")
    evidence_hash = str(evaluator.get("evidence_hash", evaluator.get("evidence_sha256", "")))
    row = collect_pg331_source_row(
        record_id=f"pg331js:{int(seed)}:{_route_attestation(route)[:16]}:{role}",
        observation=observation,
        source_meta=_source_meta(seed=int(seed), route=route, role=role, evidence_hash=evidence_hash),
        reset=_normalize_reset_for_source_row(reset),
        evaluator=evaluator,
        field_capture_manifest=field_capture_manifest,
        target_projection=target,
        split=split,
        operator_reviewed=False,
        hard_negative=False,
    )
    if _contains_raw(row):
        raise ValueError("PG-331 Juice Shop source row contains raw material")
    return row


def validate_pg331_juice_shop_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(plan, Mapping):
        return {"valid": False, "failures": ["plan_not_mapping"]}
    if str(plan.get("schema_version", "")) != SCHEMA_VERSION:
        failures.append("schema_version")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or execution.get("image") != IMAGE or execution.get("network_mode") != "none" or execution.get("loopback_only") is not True or execution.get("external_network") is not False or execution.get("real_execution") is not False:
        failures.append("execution_contract")
    if plan.get("status") != "planning_only":
        failures.append("planning_status")
    if plan.get("promotion") != {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }:
        failures.append("promotion")
    episodes = plan.get("episodes")
    expected_count = len(list(plan.get("seeds") or [])) * len(_ROUTE_BY_ID)
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        failures.append("episode_count")
    else:
        for episode in episodes:
            if not isinstance(episode, Mapping):
                failures.append("episode_shape")
                continue
            if set(str(key) for key in episode.get("roles", {})) != set(ROLES):
                failures.append(f"role_allowlist:{episode.get('route_id')}")
            contract = episode.get("observation_contract")
            if not isinstance(contract, Mapping) or contract.get("required_axis_count") != 7 or contract.get("required_field_count") != 107:
                failures.append(f"ontology_contract:{episode.get('route_id')}")
            if episode.get("training_eligible") is not False or episode.get("source_row_status") != "incomplete_ask":
                failures.append(f"training_gate:{episode.get('route_id')}")
    if _contains_raw(plan):
        failures.append("raw_material")
    expected_hash = str(plan.get("plan_sha256", "")).casefold()
    if not HASH_RE.fullmatch(expected_hash):
        failures.append("plan_sha256")
    else:
        unsigned = dict(plan)
        unsigned.pop("plan_sha256", None)
        if sha256_json(unsigned) != expected_hash:
            failures.append("plan_hash_mismatch")
    return {"valid": not failures, "failures": sorted(set(failures))}


def validate_evaluator_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(binding, Mapping):
        return {"valid": False, "failures": ["binding_not_mapping"]}
    if str(binding.get("schema_version", "")) != SCHEMA_VERSION:
        failures.append("schema_version")
    route = binding.get("route_projection")
    if not isinstance(route, Mapping) or not HASH_RE.fullmatch(str(route.get("route_ref_sha256", "")).casefold()):
        failures.append("route_projection")
    evaluator_record = binding.get("evaluator_record")
    if not isinstance(evaluator_record, Mapping):
        failures.append("evaluator_record")
    else:
        result = validate_pg331_evaluator_record(evaluator_record)
        if not result.get("valid"):
            failures.extend(f"evaluator:{item}" for item in result.get("failures") or [])
    for key in ("raw_payload_stored", "raw_response_stored", "oracle_answer_in_context", "training_eligible"):
        if binding.get(key) is not False:
            failures.append(key)
    promotion = binding.get("promotion")
    if not isinstance(promotion, Mapping) or any(promotion.get(key) is not False for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")):
        failures.append("promotion")
    if _contains_raw(binding):
        failures.append("raw_material")
    expected_hash = str(binding.get("binding_sha256", "")).casefold()
    if not HASH_RE.fullmatch(expected_hash):
        failures.append("binding_sha256")
    else:
        unsigned = dict(binding)
        unsigned.pop("binding_sha256", None)
        if sha256_json(unsigned) != expected_hash:
            failures.append("binding_hash_mismatch")
    return {"valid": not failures, "failures": sorted(set(failures)), "confirmed_positive": bool((binding.get("evaluator_sidecar") or {}).get("confirmed_positive"))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a pure PG-331 Juice Shop source-row plan; never starts Docker")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    plan = build_pg331_juice_shop_source_plan()
    print(json.dumps(plan, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMAGE",
    "ROUTES",
    "ROLES",
    "SEEDS",
    "bind_source_row",
    "build_evaluator_binding",
    "build_pg331_juice_shop_source_plan",
    "validate_evaluator_binding",
    "validate_pg331_juice_shop_plan",
]
