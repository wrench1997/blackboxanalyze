"""Pure PG-331 typed-replay planning and evaluator binding.

This module is intentionally a *pre-runtime* adapter.  It does not import
Docker, sockets, HTTP clients, model checkpoints, or training code.  A later
operator-approved runner may consume the plan and execute the allowlisted
lanes.  Here we only bind evaluator-side evidence to the fixed Pikachu SQL
surface and pass the abstract result through :mod:`app.pg331_evaluator_sidecar`.

Literal candidate/reference/negative probes may be supplied to
``build_pg331_typed_replay_record`` as in-memory evaluator arguments.  They
are used only when deriving a digest and are never returned in the model
context, record, trace, or catalog projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
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


SCHEMA_VERSION = "pg331-typed-replay-plan-v1"
RECORD_SCHEMA_VERSION = "pg331-typed-replay-record-v1"
IMAGE = "sift/pikachu-fixed@sha256:cca4288b6b701725e7a771f47ce7fcafd6cea9bd7622fa34ef2ed0b440f472c6"
NETWORK_MODE = "none"
ROLES = ("candidate", "reference", "negative")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_FRAGMENTS = ("payload", "response_body", "raw_", "oracle_answer", "evaluator_answer")
SAFE_METADATA_KEYS = frozenset(
    {
        "raw_payload_stored",
        "raw_response_stored",
        "oracle_answer_in_context",
        # This is a promotion decision field, not a payload value.  Keep it
        # outside the lexical ``payload`` firewall so the contract can carry
        # an explicit false promotion gate without a false positive.
        "payload_catalog_promotion_allowed",
    }
)

# This is the only route set accepted by the planning contract.  Paths stay
# in evaluator memory in this module; plan/record projections carry only a
# non-reversible path attestation.
ROUTES: tuple[dict[str, Any], ...] = (
    {
        "id": "pg331-sql-string-get",
        "method": "GET",
        "channel": "query",
        "path": "/vul/sqli/sqli_str.php",
        "field_count": 2,
        "oracle": "row_shape",
    },
    {
        "id": "pg331-sql-search-get",
        "method": "GET",
        "channel": "query",
        "path": "/vul/sqli/sqli_search.php",
        "field_count": 2,
        "oracle": "row_shape",
    },
    {
        "id": "pg331-sql-id-post",
        "method": "POST",
        "channel": "form",
        "path": "/vul/sqli/sqli_id.php",
        "field_count": 2,
        "oracle": "row_shape",
    },
)
_ROUTE_BY_ID = {str(route["id"]): route for route in ROUTES}


def _contains_raw(value: Any, key: str = "") -> bool:
    lowered = str(key).casefold()
    if lowered in SAFE_METADATA_KEYS:
        return False
    if any(fragment in lowered for fragment in RAW_FRAGMENTS):
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_raw(child, key) for child in value)
    return False


def _path_attestation(path: str) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _route(route_id: str) -> dict[str, Any]:
    try:
        route = _ROUTE_BY_ID[str(route_id)]
    except KeyError as error:
        raise ValueError(f"PG-331 route is not allowlisted: {route_id}") from error
    return dict(route)


def _role_container_digest(*, seed: int, route_id: str, role: str) -> str:
    return sha256_json({"schema": SCHEMA_VERSION, "image": IMAGE, "seed": int(seed), "route_id": route_id, "role": role})


def _route_plan(route: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    route_id = str(route["id"])
    return {
        "route_id": route_id,
        "method": str(route["method"]),
        "channel": str(route["channel"]),
        "field_count": int(route["field_count"]),
        "oracle_kind": "row_shape",
        "route_path_attestation_sha256": _path_attestation(str(route["path"])),
        "fresh_container_per_role": True,
        "network_mode": NETWORK_MODE,
        "host_port_published": False,
        "external_network": False,
        "role_containers": {
            role: {
                "role": role,
                "container_identity_sha256": _role_container_digest(seed=seed, route_id=route_id, role=role),
                "fresh_reset_required": True,
                "zero_volume_mounts_required": True,
            }
            for role in ROLES
        },
    }


def build_pg331_typed_replay_plan(*, seed: int = 33101, route_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Create a no-I/O plan for the fixed SQL row-shape replay matrix."""

    requested = tuple(str(value) for value in (route_ids if route_ids is not None else _ROUTE_BY_ID))
    if set(requested) != set(_ROUTE_BY_ID) or len(requested) != len(_ROUTE_BY_ID):
        raise ValueError("PG-331 typed replay requires exactly the three allowlisted SQL routes")
    routes = [_route_plan(_route(route_id), seed=int(seed)) for route_id in requested]
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned_incomplete",
        "execution": {
            "image": IMAGE,
            "network_mode": NETWORK_MODE,
            "external_network": False,
            "host_port_published": False,
            "fresh_disposable_container_per_route_role": True,
            "real_execution": False,
        },
        "seed": int(seed),
        "roles": list(ROLES),
        "routes": routes,
        "model_context": {
            "surface_method": "GET_POST_PAIR",
            "surface_field_role": "parameter_role",
            "surface_encoding": "url_percent_or_form_urlencoded",
            "typed_available": "unknown",
            "evidence_present": "unknown",
            "negative_control": "unknown",
            "fresh_reset": "unknown",
            "replay_ready": "unknown",
            "candidate_present": "unknown",
            "reference_present": "unknown",
        },
        "trace_projection": [],
        "catalog_projection": [],
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    if _contains_raw(plan["model_context"]) or _contains_raw(plan["trace_projection"]) or _contains_raw(plan["catalog_projection"]):
        raise ValueError("PG-331 typed replay plan leaked raw material")
    plan["plan_sha256"] = sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    return plan


def _role_with_probe(role: str, value: Mapping[str, Any], probe: Any | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-331 {role} evaluator result must be an object")
    result = dict(value)
    # The probe may be a literal evaluator value.  It is hashed in memory and
    # then discarded; no ``probe`` key is passed to the sidecar.
    if probe is not None:
        unsigned = {str(key): child for key, child in result.items() if not any(fragment in str(key).casefold() for fragment in RAW_FRAGMENTS)}
        result["evidence_sha256"] = sha256_json({"role": role, "probe": probe, "evaluator_projection": unsigned})
    return result


def build_pg331_typed_replay_record(
    *,
    route_id: str,
    seed: int,
    reset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    literal_probes: Mapping[str, Any] | None = None,
    replay_consistent: bool = False,
    reference_agreement: bool | None = None,
    negative_control_clean: bool | None = None,
    evaluator_id: str = "pg331-typed-replay-evaluator",
) -> dict[str, Any]:
    """Bind one route's evaluator evidence while keeping literal probes off-record."""

    route = _route(route_id)
    probes = dict(literal_probes or {})
    unknown_probe_roles = sorted(str(key) for key in probes if str(key) not in ROLES)
    if unknown_probe_roles:
        raise ValueError(f"PG-331 literal probes contain unsupported roles: {', '.join(unknown_probe_roles)}")
    roles = {
        role: _role_with_probe(role, value, probes.get(role))
        for role, value in (("candidate", candidate), ("reference", reference), ("negative", negative))
    }
    sidecar_record = build_pg331_evaluator_record(
        record_id=f"pg331:{int(seed)}:{route_id}",
        reset=reset,
        candidate=roles["candidate"],
        reference=roles["reference"],
        negative=roles["negative"],
        replay_consistent=replay_consistent,
        reference_agreement=reference_agreement,
        negative_control_clean=negative_control_clean,
        evaluator_id=evaluator_id,
    )
    route_projection = _route_plan(route, seed=int(seed))
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_id": sidecar_record["record_id"],
        "route_projection": route_projection,
        "evaluator_sidecar": sidecar_record["evaluator_sidecar"],
        "model_context": sidecar_record["model_context"],
        "trace_projection": [],
        "catalog_projection": [],
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
    if _contains_raw(record):
        raise ValueError("PG-331 typed replay record contains raw material")
    record["record_sha256"] = sha256_json(record)
    return record


def validate_pg331_typed_replay_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(plan, Mapping):
        return {"valid": False, "failures": ["plan_not_mapping"]}
    if str(plan.get("schema_version", "")) != SCHEMA_VERSION:
        failures.append("schema_version")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or execution.get("image") != IMAGE or execution.get("network_mode") != NETWORK_MODE or execution.get("external_network") is not False or execution.get("host_port_published") is not False or execution.get("fresh_disposable_container_per_route_role") is not True:
        failures.append("execution_contract")
    routes = plan.get("routes")
    if not isinstance(routes, list) or len(routes) != 3 or {str(item.get("route_id")) for item in routes if isinstance(item, Mapping)} != set(_ROUTE_BY_ID):
        failures.append("route_allowlist")
    else:
        for route in routes:
            if not isinstance(route, Mapping):
                failures.append("route_shape")
                continue
            if set(str(key) for key in route.get("role_containers", {})) != set(ROLES):
                failures.append(f"role_allowlist:{route.get('route_id')}")
            if route.get("network_mode") != NETWORK_MODE or route.get("fresh_container_per_role") is not True or route.get("host_port_published") is not False:
                failures.append(f"route_execution:{route.get('route_id')}")
    if _contains_raw(plan.get("model_context")) or _contains_raw(plan.get("trace_projection")) or _contains_raw(plan.get("catalog_projection")):
        failures.append("context_firewall")
    expected_hash = str(plan.get("plan_sha256", "")).casefold()
    if not HASH_RE.fullmatch(expected_hash):
        failures.append("plan_sha256")
    else:
        unsigned = dict(plan)
        unsigned.pop("plan_sha256", None)
        if sha256_json(unsigned) != expected_hash:
            failures.append("plan_hash_mismatch")
    return {"valid": not failures, "failures": sorted(set(failures))}


def validate_pg331_typed_replay_record(record: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(record, Mapping):
        return {"valid": False, "failures": ["record_not_mapping"]}
    if str(record.get("schema_version", "")) != RECORD_SCHEMA_VERSION:
        failures.append("schema_version")
    route = record.get("route_projection")
    if not isinstance(route, Mapping) or str(route.get("route_id")) not in _ROUTE_BY_ID:
        failures.append("route_allowlist")
    elif route.get("network_mode") != NETWORK_MODE or route.get("fresh_container_per_role") is not True or set(str(key) for key in route.get("role_containers", {})) != set(ROLES):
        failures.append("route_execution")
    sidecar = record.get("evaluator_sidecar")
    sidecar_record = {
        "schema_version": "pg331-evaluator-record-v1",
        "record_id": record.get("record_id"),
        "model_context": record.get("model_context"),
        "evaluator_sidecar": sidecar,
        "raw_payload_stored": record.get("raw_payload_stored"),
        "raw_response_stored": record.get("raw_response_stored"),
        "oracle_answer_in_context": record.get("oracle_answer_in_context"),
        "context_firewall": {"sidecars_off_context": True, "forbidden_token_count": 0},
        "training_eligible": record.get("training_eligible"),
        "promotion": record.get("promotion"),
    }
    # The sidecar module validates its own evidence hash; this adapter adds
    # the route/network/role planning gates without re-contacting anything.
    sidecar_result = validate_pg331_evaluator_record({**sidecar_record, "record_sha256": sha256_json(sidecar_record)})
    if not sidecar_result.get("valid"):
        failures.extend(f"sidecar:{item}" for item in sidecar_result.get("failures") or [])
    if _contains_raw(record):
        failures.append("raw_material")
    expected_hash = str(record.get("record_sha256", "")).casefold()
    if not HASH_RE.fullmatch(expected_hash):
        failures.append("record_sha256")
    else:
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        if sha256_json(unsigned) != expected_hash:
            failures.append("record_hash_mismatch")
    return {"valid": not failures, "failures": sorted(set(failures)), "confirmed_positive": bool(sidecar_result.get("confirmed_positive"))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a pure PG-331 typed replay plan; never starts Docker")
    parser.add_argument("--seed", type=int, default=33101)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    plan = build_pg331_typed_replay_plan(seed=args.seed)
    print(json.dumps(plan, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
