"""Planning-only PG-342 full-axis failure/repair source-row contract.

The module deliberately does not import Docker, HTTP, a browser, torch, or an
evaluator.  It describes the live collector that must be implemented later.
No route, payload, response, oracle, or vulnerability-family literal is put in
the model-facing projection; the plan keeps only bounded lane references and
hashes.  A live adapter must fill the seven-axis/107-field manifest before a
row can be considered for training.
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
SCHEMA_VERSION = "pg342-full-axis-failure-repair-plan-v1"
SEEDS = (34201, 34202, 34203)
ROLES = ("candidate", "reference", "negative", "replay")
SOURCE_ROLES = ("candidate", "reference", "negative")

# These are fixed, authorized disposable images already present in the
# research ledger.  They are attestations only; the plan never starts them.
IMPLEMENTATIONS = (
    {
        "id": "impl_a",
        "image_digest": "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe",
        "source_role": "train",
    },
    {
        "id": "impl_b",
        "image_digest": "vulnerables/web-dvwa@sha256:dae203fe11646a86937bf04db0079adef295f426da68a92b40e3b181f337daa7",
        "source_role": "train",
    },
    {
        "id": "impl_c",
        "image_digest": "webgoat/webgoat@sha256:3101bd9e7bcfe122d7ef91e690ef3720de36cc4e86b3d06763a1ddf2e2751a4b",
        "source_role": "implementation_holdout",
    },
)
LANES = (
    {"id": "get_shape_a", "method": "GET", "stateful": False},
    {"id": "post_shape_a", "method": "POST", "stateful": False},
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _manifest() -> dict[str, dict[str, str]]:
    path = ROOT / "research" / "pg331_web_token_ontology_v1.json"
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    axes = value.get("axes") if isinstance(value, Mapping) else None
    if not isinstance(axes, Mapping):
        raise ValueError("PG-331 ontology axes are required")
    return {
        str(axis): {str(field): "not_observed" for field in list(spec.get("fields") or [])}
        for axis, spec in axes.items()
        if isinstance(spec, Mapping)
    }


def _implementation_ref(implementation: Mapping[str, Any]) -> str:
    return _hash({"id": implementation["id"], "image_digest": implementation["image_digest"]})


def _lane_ref(lane: Mapping[str, Any]) -> str:
    return _hash({"id": lane["id"], "method": lane["method"], "stateful": bool(lane.get("stateful"))})


def _identity(seed: int, implementation: Mapping[str, Any], lane: Mapping[str, Any], role: str) -> str:
    return _hash(
        {
            "seed": seed,
            "implementation_ref": _implementation_ref(implementation),
            "lane_ref": _lane_ref(lane),
            "role": role,
            "network": "none",
            "fresh_reset": True,
        }
    )


def _role_contract(seed: int, implementation: Mapping[str, Any], lane: Mapping[str, Any], role: str) -> dict[str, Any]:
    stateful = bool(lane.get("stateful"))
    return {
        "target_identity_sha256": _identity(seed, implementation, lane, role),
        "fresh_reset_required": True,
        "fresh_reset_observed": False,
        "state_reset_before_required": stateful,
        "state_reset_after_required": stateful,
        "database_clean_attestation_required": stateful,
        "teardown_required": True,
        "source_row_allowed": role in SOURCE_ROLES,
    }


def build_pg342_plan(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("PG-342 requires at least one seed")
    manifest = _manifest()
    episodes: list[dict[str, Any]] = []
    for implementation in IMPLEMENTATIONS:
        for seed in normalized:
            for lane in LANES:
                roles = {role: _role_contract(seed, implementation, lane, role) for role in ROLES}
                episodes.append(
                    {
                        "seed": seed,
                        "implementation_ref_sha256": _implementation_ref(implementation),
                        "implementation_split": implementation["source_role"],
                        "lane_ref_sha256": _lane_ref(lane),
                        "method": lane["method"],
                        "roles": roles,
                        "observation_contract": {
                            "required_axis_count": len(manifest),
                            "required_field_count": sum(len(fields) for fields in manifest.values()),
                            "field_capture_manifest": manifest,
                            "manifest_status": "not_observed_until_live_adapter",
                        },
                        "typed_contract": {
                            "candidate_reference_negative_required": True,
                            "replay_required": True,
                            "role_bound_evidence_sha256_required": True,
                            "post_capability": "unknown_until_live_adapter" if lane["method"] == "POST" else "required",
                        },
                        "failure_repair_contract": {
                            "baseline_required": True,
                            "failure_observation_required": True,
                            "failure_action_change_required": True,
                            "repair_observation_required": True,
                            "previous_action_must_differ_from_next": True,
                            "candidate_and_reference_repair": True,
                            "negative_action": "abstain_after_failure",
                            "variant_refs_are_hash_only": True,
                        },
                        "model_projection": {
                            "missing_observation_next_action": "ask_typed",
                            "missing_observation_safe_to_send": False,
                            "failure_without_repair_next_action": "ask_failure",
                            "negative_next_action": "abstain",
                        },
                        "source_row_status": "incomplete_ask",
                        "training_eligible": False,
                    }
                )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only",
        "execution": {
            "docker_started": False,
            "network_contacted": False,
            "network_mode": "none",
            "loopback_relay_required": True,
            "published_ports_allowed": False,
            "bind_or_volume_mounts_allowed": False,
            "fresh_container_per_seed_lane_role": True,
            "evaluator_side_state_only": True,
        },
        "implementations": [dict(item) for item in IMPLEMENTATIONS],
        "lanes": [dict(item) for item in LANES],
        "seeds": list(normalized),
        "episodes": episodes,
        "required_split_rule": {
            "train_implementations_min": 2,
            "holdout_implementations_min": 1,
            "ask_repair_negative_required_in_each_split": True,
            "no_row_relabeling": True,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": "planning_only: live adapter must fill all seven axes and 107 fields; missing typed/reset/evidence remains ASK and cannot become a training row.",
    }
    plan["plan_sha256"] = _hash(plan)
    return plan


def validate_pg342_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(plan, Mapping) or plan.get("schema_version") != SCHEMA_VERSION or plan.get("status") != "planning_only":
        failures.append("schema")
    execution = plan.get("execution") if isinstance(plan, Mapping) else None
    if (
        not isinstance(execution, Mapping)
        or execution.get("network_mode") != "none"
        or execution.get("docker_started") is not False
        or execution.get("network_contacted") is not False
        or execution.get("loopback_relay_required") is not True
        or execution.get("published_ports_allowed") is not False
        or execution.get("bind_or_volume_mounts_allowed") is not False
    ):
        failures.append("execution")
    implementations = plan.get("implementations") if isinstance(plan, Mapping) else None
    if not isinstance(implementations, list) or len(implementations) != 3:
        failures.append("implementations")
    else:
        refs = {_implementation_ref(item) for item in implementations if isinstance(item, Mapping)}
        if len(refs) != 3:
            failures.append("implementation_refs")
        splits = {str(item.get("source_role")) for item in implementations if isinstance(item, Mapping)}
        if splits != {"train", "implementation_holdout"}:
            failures.append("splits")
    lanes = plan.get("lanes") if isinstance(plan, Mapping) else None
    if not isinstance(lanes, list) or {str(item.get("method")) for item in lanes if isinstance(item, Mapping)} != {"GET", "POST"}:
        failures.append("get_post")
    episodes = plan.get("episodes") if isinstance(plan, Mapping) else None
    seeds = list(plan.get("seeds") or []) if isinstance(plan, Mapping) else []
    if not isinstance(episodes, list) or len(episodes) != len(seeds) * 3 * 2:
        failures.append("episodes")
    else:
        for episode in episodes:
            if not isinstance(episode, Mapping):
                failures.append("episode_shape")
                continue
            roles = episode.get("roles")
            observation = episode.get("observation_contract")
            typed = episode.get("typed_contract")
            repair = episode.get("failure_repair_contract")
            if not isinstance(roles, Mapping) or set(roles) != set(ROLES) or len({item.get("target_identity_sha256") for item in roles.values() if isinstance(item, Mapping)}) != 4:
                failures.append("fresh_roles")
            if not isinstance(observation, Mapping) or observation.get("required_axis_count") != 7 or observation.get("required_field_count") != 107:
                failures.append("ontology")
            if not isinstance(typed, Mapping) or typed.get("candidate_reference_negative_required") is not True or typed.get("replay_required") is not True or typed.get("role_bound_evidence_sha256_required") is not True:
                failures.append("typed")
            if not isinstance(repair, Mapping) or repair.get("failure_action_change_required") is not True or repair.get("previous_action_must_differ_from_next") is not True or repair.get("negative_action") != "abstain_after_failure":
                failures.append("failure_repair")
            if episode.get("training_eligible") is not False:
                failures.append("training")
    promotion = plan.get("promotion") if isinstance(plan, Mapping) else None
    if not isinstance(promotion, Mapping) or any(promotion.get(key) is not False for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")):
        failures.append("promotion")
    unsigned = dict(plan) if isinstance(plan, Mapping) else {}
    actual = unsigned.pop("plan_sha256", "")
    if _hash(unsigned) != actual:
        failures.append("hash")
    return {"valid": not failures, "failures": sorted(set(failures))}


def write_plan(path: Path, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    target = path.resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("PG-342 output must remain inside the workspace") from error
    document = dict(plan) if isinstance(plan, Mapping) else build_pg342_plan()
    if not validate_pg342_plan(document)["valid"]:
        raise ValueError("refusing to write an invalid PG-342 plan")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="build planning-only PG-342 full-axis failure-repair contract")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_pg342_plan()
    if args.output is not None:
        write_plan(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
