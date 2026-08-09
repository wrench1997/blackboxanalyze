"""Planning-only PG-332 DVWA strict whole-page source-row contract.

This module has no Docker, HTTP, browser, model, or evaluator dependency.
Routes remain internal allowlist entries and are exposed only as one-way
attestations.  It cannot transform older coarse DVWA data into PG-331 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pg332-dvwa-source-row-plan-v1"
IMAGE = "vulnerables/web-dvwa@sha256:dae203fe11646a86937bf04db0079adef295f426da68a92b40e3b181f337daa7"
SEEDS = (33201, 33202, 33203)
ROLES = ("candidate", "reference", "negative", "replay")
SOURCE_ROLES = ("candidate", "reference", "negative")

# Never serialize these path/field literals into a plan artifact.
_ROUTES = (
    {"id": "dvwa-xss-reflected-get", "path": "/vulnerabilities/xss_r/", "field": "name", "method": "GET"},
    {"id": "dvwa-sqli-get", "path": "/vulnerabilities/sqli/", "field": "id", "method": "GET"},
    # User-authorized evaluator-only stateful lane.  It may run only in a
    # per-role disposable target with reset before/after and teardown.
    {"id": "dvwa-xss-stored-post", "path": "/vulnerabilities/xss_s/", "field": "txtName", "method": "POST", "stateful_disposable": True},
)
_BY_ID = {str(route["id"]): route for route in _ROUTES}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _ontology() -> dict[str, Any]:
    value = json.loads((ROOT / "research" / "pg331_web_token_ontology_v1.json").read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict): raise ValueError("PG-331 ontology must be an object")
    return value


def _route_ref(route: Mapping[str, Any]) -> str:
    return _hash({"id": route["id"], "path": route["path"], "field": route["field"], "method": route["method"], "image": IMAGE})


def _manifest() -> dict[str, dict[str, str]]:
    return {str(axis): {str(field): "not_observed" for field in list(spec.get("fields") or [])} for axis, spec in dict(_ontology().get("axes") or {}).items() if isinstance(spec, Mapping)}


def _identity(seed: int, route: Mapping[str, Any], role: str) -> str:
    return _hash({"seed": seed, "route_ref": _route_ref(route), "role": role, "image": IMAGE, "network": "none", "fresh_reset": True})


def build_pg332_dvwa_source_plan(*, seeds: Sequence[int] = SEEDS, route_ids: Sequence[str] | None = None) -> dict[str, Any]:
    requested = tuple(str(value) for value in (route_ids if route_ids is not None else _BY_ID))
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized: raise ValueError("PG-332 requires seeds")
    if set(requested) != set(_BY_ID) or len(requested) != len(_BY_ID): raise ValueError("PG-332 requires exactly the DVWA allowlist")
    manifest = _manifest()
    episodes = []
    for seed in normalized:
        for route_id in requested:
            route = _BY_ID[route_id]
            method = str(route["method"])
            stateful = bool(route.get("stateful_disposable"))
            episodes.append({"seed": seed, "route_ref_sha256": _route_ref(route), "method": method, "roles": {role: {"target_identity_sha256": _identity(seed, route, role), "fresh_reset_required": True, "fresh_reset_observed": False, "state_reset_before_required": stateful, "state_reset_after_required": stateful, "database_clean_attestation_required": stateful, "teardown_required": stateful, "source_row_allowed": role in SOURCE_ROLES} for role in ROLES}, "stateful_disposable": stateful, "state_contract": {"evaluator_side_state_delta_only": stateful, "model_context_state_or_payload_allowed": False, "fresh_container_per_role": stateful, "external_network": False, "bind_or_volume_mounts": False} if stateful else {"evaluator_side_state_delta_only": False, "model_context_state_or_payload_allowed": False}, "observation_contract": {"required_axis_count": len(manifest), "required_field_count": sum(len(fields) for fields in manifest.values()), "field_capture_manifest": manifest, "manifest_status": "not_observed_until_live_adapter"}, "typed_contract": {"candidate_reference_negative_required": True, "replay_required": True, "role_bound_evidence_sha256_required": True, "post_typed_available": "unknown_until_evaluator" if method == "POST" else "unknown_until_evaluator"}, "model_projection": {"next_action": "ask_typed", "safe_to_send": False}, "source_row_status": "incomplete_ask", "training_eligible": False})
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "planning_only", "execution": {"docker_started": False, "network_contacted": False, "image": IMAGE, "network_mode": "none", "loopback_relay_required": True, "published_ports_allowed": False, "bind_or_volume_mounts_allowed": False, "stateful_disposable_evaluator_only": True}, "seeds": list(normalized), "episodes": episodes, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}, "interpretation": "planning_only: all 107 ontology fields are not_observed and must remain ASK/incomplete; a user-authorized stateful lane is evaluator-only/disposable, while legacy PG-146/207 coarse data is excluded."}
    result["plan_sha256"] = _hash(result)
    return result


def validate_pg332_dvwa_source_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(plan, Mapping) or plan.get("schema_version") != SCHEMA_VERSION or plan.get("status") != "planning_only": failures.append("schema")
    execution = plan.get("execution") if isinstance(plan, Mapping) else {}
    if not isinstance(execution, Mapping) or execution.get("image") != IMAGE or execution.get("network_mode") != "none" or execution.get("loopback_relay_required") is not True or execution.get("published_ports_allowed") is not False or execution.get("bind_or_volume_mounts_allowed") is not False: failures.append("execution")
    episodes = plan.get("episodes") if isinstance(plan, Mapping) else []
    if not isinstance(episodes, list) or len(episodes) != len(list(plan.get("seeds") or [])) * len(_BY_ID): failures.append("episodes")
    else:
        for episode in episodes:
            roles = episode.get("roles") if isinstance(episode, Mapping) else {}
            contract = episode.get("observation_contract") if isinstance(episode, Mapping) else {}
            if not isinstance(roles, Mapping) or set(roles) != set(ROLES) or len({item.get("target_identity_sha256") for item in roles.values() if isinstance(item, Mapping)}) != 4: failures.append("fresh_roles")
            if not isinstance(contract, Mapping) or contract.get("required_axis_count") != 7 or contract.get("required_field_count") != 107 or episode.get("training_eligible") is not False: failures.append("ontology_or_training")
            if isinstance(episode, Mapping) and episode.get("method") == "POST":
                if dict(episode.get("typed_contract") or {}).get("post_typed_available") != "unknown_until_evaluator": failures.append("post_typed_contract")
                state = dict(episode.get("state_contract") or {})
                if episode.get("stateful_disposable") is not True or not all(bool(dict(item).get(key)) for item in roles.values() if isinstance(item, Mapping) for key in ("state_reset_before_required", "state_reset_after_required", "database_clean_attestation_required", "teardown_required")) or state.get("evaluator_side_state_delta_only") is not True or state.get("model_context_state_or_payload_allowed") is not False: failures.append("stateful_disposable")
    unsigned = dict(plan) if isinstance(plan, Mapping) else {}; actual = unsigned.pop("plan_sha256", "")
    if _hash(unsigned) != actual: failures.append("hash")
    return {"valid": not failures, "failures": sorted(set(failures))}


def write_plan(path: Path, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write a stable UTF-8 planning artifact only inside this workspace."""
    target = path.resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("PG-332 output must remain inside the workspace") from error
    document = dict(plan) if isinstance(plan, Mapping) else build_pg332_dvwa_source_plan()
    if not validate_pg332_dvwa_source_plan(document)["valid"]:
        raise ValueError("refusing to write an invalid PG-332 plan")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="build planning-only PG-332 DVWA source-row contract")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="optional UTF-8 plan artifact path inside the workspace")
    args = parser.parse_args()
    plan = build_pg332_dvwa_source_plan()
    if args.output is not None: write_plan(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2 if args.json else None)); return 0


if __name__ == "__main__": raise SystemExit(main())
