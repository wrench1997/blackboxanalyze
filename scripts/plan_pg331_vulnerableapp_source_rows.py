"""Static PG-331 collection contract for the independent PG-246 VulnerableApp.

This is deliberately a plan, not a replay runner.  It neither imports Docker,
Playwright, nor HTTP code.  It records the six already-authorised PG-246 lane
*identities* as one-way attestations, requires separate fresh targets for the
candidate/reference/negative/replay roles, and describes the complete seven
axis / 107 field capture contract.  Until a later, explicitly authorised
loopback collector supplies structural observations and typed sidecars, every
planned row is incomplete ASK and cannot train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import sha256_json  # noqa: E402
from app.pg331_source_row import collect_pg331_source_row  # noqa: E402


SCHEMA_VERSION = "pg331-vulnerableapp-source-row-plan-v1"
IMAGE_DIGEST = "7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"
SOURCE_COMMIT = "sasanlabs-vulnerableapp-2.1.44"
SEEDS = (24601, 24602, 24603)
ROLES = ("candidate", "reference", "negative", "replay")
SOURCE_ROLES = ("candidate", "reference", "negative")
# ``path_shape`` is a declared structural ontology field, so route literals
# are excluded by construction rather than treating every ``path`` field as
# raw material.
RAW_FRAGMENTS = ("payload", "response_body", "raw_", "oracle_answer", "evaluator_answer", "wire")
SAFE_KEYS = frozenset({"raw_payload_stored", "raw_response_body_stored", "raw_wire_off_context", "raw_response_off_context", "oracle_answer_off_context", "payload_catalog_promotion_allowed"})
PROMOTION = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}

# Paths and field names remain evaluator-only.  The emitted plan carries only
# case IDs, method, lane class, and a one-way attestation.
_ROUTES: tuple[dict[str, Any], ...] = (
    {"id": "vapp_html_level1_get", "path": "/VulnerableApp/XSSWithHtmlTagInjection/LEVEL_1", "field": "probe", "method": "GET", "lane": "positive", "post_supported": False},
    {"id": "vapp_html_level4_secure_get", "path": "/VulnerableApp/XSSWithHtmlTagInjection/LEVEL_4", "field": "probe", "method": "GET", "lane": "negative", "post_supported": False},
    {"id": "vapp_img_level1_get", "path": "/VulnerableApp/XSSInImgTagAttribute/LEVEL_1", "field": "src", "method": "GET", "lane": "positive", "post_supported": False},
    {"id": "vapp_img_level6_secure_get", "path": "/VulnerableApp/XSSInImgTagAttribute/LEVEL_6", "field": "src", "method": "GET", "lane": "negative", "post_supported": False},
    {"id": "vapp_html_level1_post_405", "path": "/VulnerableApp/XSSWithHtmlTagInjection/LEVEL_1", "field": "probe", "method": "POST", "lane": "unsupported_post", "post_supported": False},
    {"id": "vapp_img_level1_post_405", "path": "/VulnerableApp/XSSInImgTagAttribute/LEVEL_1", "field": "src", "method": "POST", "lane": "unsupported_post", "post_supported": False},
)
_BY_ID = {str(route["id"]): route for route in _ROUTES}


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


@lru_cache(maxsize=1)
def _ontology() -> dict[str, Any]:
    value = json.loads((ROOT / "research" / "pg331_web_token_ontology_v1.json").read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("PG-331 ontology must be an object")
    return value


def _route_attestation(route: Mapping[str, Any]) -> str:
    return sha256_json({"id": route["id"], "path": route["path"], "field": route["field"], "method": route["method"], "image_digest": IMAGE_DIGEST, "source_commit": SOURCE_COMMIT})


def _manifest() -> dict[str, dict[str, str]]:
    return {str(axis): {str(field): "not_observed" for field in list(spec.get("fields") or [])} for axis, spec in dict(_ontology().get("axes") or {}).items() if isinstance(spec, Mapping)}


def _presence() -> dict[str, str]:
    return {str(spec.get("presence_token")): "not_observed" for spec in dict(_ontology().get("axes") or {}).values() if isinstance(spec, Mapping)}


def _identity(seed: int, route: Mapping[str, Any], role: str) -> str:
    return sha256_json({"schema": SCHEMA_VERSION, "seed": seed, "route_ref": _route_attestation(route), "role": role, "image_digest": IMAGE_DIGEST, "network_mode": "loopback", "fresh_reset": True})


def _episode(seed: int, route: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _manifest()
    field_count = sum(len(values) for values in manifest.values())
    method = str(route["method"])
    return {
        "seed": seed,
        "case_id": str(route["id"]),
        "route_ref_sha256": _route_attestation(route),
        "method": method,
        "lane": "unsupported_post_ask" if method == "POST" else "get_baseline_then_typed",
        "roles": {
            role: {"target_identity_sha256": _identity(seed, route, role), "fresh_reset_required": True, "fresh_reset_observed": False, "network_mode": "loopback", "loopback_only": True, "external_network": False, "bind_or_volume_mount_count_required": 0, "source_row_allowed": role in SOURCE_ROLES}
            for role in ROLES
        },
        "observation_contract": {"required_axis_count": len(manifest), "required_field_count": field_count, "axis_presence": _presence(), "field_capture_manifest": manifest, "manifest_status": "not_observed_until_live_adapter"},
        "typed_evidence_contract": {"candidate_reference_negative_required": True, "replay_required": True, "role_bound_evidence_sha256_required": True, "typed_oracle_required": True, "raw_wire_off_context": True, "raw_response_off_context": True, "oracle_answer_off_context": True},
        "model_context_projection": {"next_action": "ask_typed", "safe_to_send": False},
        "source_row_status": "incomplete_ask",
        "training_eligible": False,
    }


def build_pg331_vulnerableapp_source_plan(*, seeds: Sequence[int] = SEEDS, case_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Return a no-I/O plan; it does not collect, contact, or promote anything."""
    normalized = tuple(int(seed) for seed in seeds)
    requested = tuple(str(item) for item in (case_ids if case_ids is not None else _BY_ID))
    if not normalized:
        raise ValueError("plan requires at least one seed")
    if set(requested) != set(_BY_ID) or len(requested) != len(_BY_ID):
        raise ValueError("plan requires exactly the six PG-246 allowlisted cases")
    episodes = [_episode(seed, _BY_ID[case_id]) for seed in normalized for case_id in requested]
    plan: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "planning_only", "execution": {"real_execution": False, "docker_started": False, "network_contacted": False, "image_digest": IMAGE_DIGEST, "source_commit": SOURCE_COMMIT, "network_mode": "loopback", "loopback_only": True, "external_network": False}, "seeds": list(normalized), "case_count": len(requested), "episode_count": len(episodes), "roles": list(ROLES), "source_roles": list(SOURCE_ROLES), "ontology_contract": {"axis_count": 7, "field_count": 107, "missing_status_forces_ask": True}, "episodes": episodes, "promotion": dict(PROMOTION), "interpretation": "planning_only: no target contacted; all fields are not_observed, source rows remain ASK/incomplete, and no old PG-246 record is made training eligible."}
    if _contains_raw(plan):
        raise ValueError("plan contains raw material")
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def bind_source_row(*, seed: int, case_id: str, role: str, observation: Mapping[str, Any], reset: Mapping[str, Any], evaluator: Mapping[str, Any], field_capture_manifest: Mapping[str, Any], target_projection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Pure future collector binding; review stays false so it cannot promote a row."""
    if case_id not in _BY_ID or role not in SOURCE_ROLES:
        raise ValueError("source binding requires an allowlisted case and candidate/reference/negative role")
    route = _BY_ID[case_id]
    if str(route["method"]) == "POST" and target_projection and target_projection.get("safe_to_send") is True:
        raise ValueError("unsupported PG-246 POST lane must remain ASK-only")
    evidence = str(evaluator.get("evidence_hash", ""))
    return collect_pg331_source_row(record_id=f"pg331vapp:{seed}:{_route_attestation(route)[:16]}:{role}", observation=observation, source_meta={"source_id": "pg331-vulnerableapp-local", "implementation": "sasanlabs-vulnerableapp-java-spring", "collector_id": SCHEMA_VERSION, "authorization_id": "operator-authorized-local-loopback", "image_digest": IMAGE_DIGEST, "source_digest": sha256_json({"seed": seed, "route_ref": _route_attestation(route), "role": role, "evidence": evidence})}, reset=reset, evaluator=evaluator, field_capture_manifest=field_capture_manifest, target_projection=dict(target_projection or {"question": "ask_typed", "next_action": "ask_typed", "repair_action": "observe", "transport_ref": "unknown", "field_role_ref": "unknown", "encoding_ref": "unknown", "probe_variant_ref": "none", "safe_to_send": False}), split="implementation_holdout", operator_reviewed=False, hard_negative=False)


def validate_pg331_vulnerableapp_source_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if str(plan.get("schema_version", "")) != SCHEMA_VERSION or plan.get("status") != "planning_only": failures.append("schema")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or execution.get("real_execution") is not False or execution.get("docker_started") is not False or execution.get("network_contacted") is not False or execution.get("external_network") is not False or execution.get("loopback_only") is not True: failures.append("execution")
    episodes = plan.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(plan.get("seeds") or []) * len(_BY_ID): failures.append("episodes")
    else:
        for episode in episodes:
            if not isinstance(episode, Mapping) or set((episode.get("roles") or {}).keys()) != set(ROLES): failures.append("roles"); continue
            contract = episode.get("observation_contract") or {}
            if contract.get("required_axis_count") != 7 or contract.get("required_field_count") != 107 or episode.get("training_eligible") is not False: failures.append("ontology_or_gate")
            if len({entry.get("target_identity_sha256") for entry in (episode.get("roles") or {}).values()}) != 4: failures.append("fresh_role_identity")
    if plan.get("promotion") != PROMOTION: failures.append("promotion")
    if _contains_raw(plan): failures.append("raw_material")
    unsigned = dict(plan); actual = unsigned.pop("plan_sha256", "")
    if sha256_json(unsigned) != actual: failures.append("plan_hash")
    return {"valid": not failures, "failures": sorted(set(failures))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static PG-331 VulnerableApp source-row plan")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_pg331_vulnerableapp_source_plan(), ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
