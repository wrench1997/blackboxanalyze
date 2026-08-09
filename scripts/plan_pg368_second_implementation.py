"""Fail-closed PG-368 second-implementation plan.

PG-367 exercises a small in-process WAF staircase.  This plan selects the
reviewed WebGoat image/relay as a *different implementation* and specifies the
fresh GET/POST method-shape replay that would be needed to compare the two.
The module is planning-only: it never invokes Docker, opens a socket, starts a
browser, or trains a model.  Until a future live run supplies typed evidence,
every model-facing action remains ``ASK``/``safe_to_send=False``.

Only route identifiers, method/shape classes and SHA-256 references are
serialized.  The WebGoat route path, form values, headers, response bytes and
evaluator answer are deliberately not part of the plan or model context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg333_webgoat_docker_relay import (  # noqa: E402
    IMAGE as RELAY_IMAGE,
    IMAGE_DIGEST as RELAY_IMAGE_DIGEST,
    ROUTE_PATH as RELAY_ROUTE_PATH,
)

SCHEMA_VERSION = "pg368-second-implementation-plan-v1"
IMAGE = "webgoat/webgoat@sha256:3101bd9e7bcfe122d7ef91e690ef3720de36cc4e86b3d06763a1ddf2e2751a4b"
IMAGE_DIGEST = IMAGE.split("@sha256:", 1)[1]
SEEDS = (36801, 36802, 36803)
ROLES = ("candidate", "reference", "negative", "replay")
SOURCE_ROLES = ("candidate", "reference", "negative")
RUNTIME_PATH = ROOT / "app" / "pg333_webgoat_docker_relay.py"
SOURCE_AUDIT_PATH = ROOT / "research" / "pg333_webgoat_typed_method_shape_source_audit_v1.json"
SOURCE_REPORT_PATH = ROOT / "research" / "pg333_webgoat_typed_method_shape_report_v1.json"

# PG-367 is intentionally named here only as an abstract implementation
# identity.  Comparing source/module hashes prevents relabelling a shared
# synthetic runtime as an independent target.
PG367_RUNTIME_PATH = ROOT / "app" / "pg367_waf_runtime.py"
PG367_IMPLEMENTATION_ID = "pg367_inprocess_waf_staircase"
PG368_IMPLEMENTATION_ID = "pg368_webgoat_docker_relay"

ROUTES: tuple[dict[str, str], ...] = (
    {"route_id": "webgoat_method_shape_get", "method": "GET", "response_shape": "html_page"},
    {"route_id": "webgoat_method_shape_post", "method": "POST", "response_shape": "loopback_redirect"},
)

_FORBIDDEN_KEYS = {
    "payload", "raw_payload", "probe_value", "raw_value", "request_body",
    "response_body", "raw_response", "url", "uri", "evaluator_answer",
}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_ref_sha256(route: Mapping[str, str]) -> str:
    """Hash the allowlisted route identity without serializing its path."""

    return sha256_json(
        {
            "schema": SCHEMA_VERSION,
            "route_id": str(route["route_id"]),
            "method": str(route["method"]),
            # The path is already fixed by the reviewed PG-333 relay.  The
            # literal is intentionally represented only by a digest here.
            "path_digest_ref": sha256_json(RELAY_ROUTE_PATH),
            "image_digest": IMAGE_DIGEST,
        }
    )


def target_identity_sha256(*, seed: int, route_ref: str, role: str) -> str:
    return sha256_json(
        {
            "schema": SCHEMA_VERSION,
            "seed": int(seed),
            "route_ref_sha256": str(route_ref),
            "role": str(role),
            "implementation": PG368_IMPLEMENTATION_ID,
            "image_digest": IMAGE_DIGEST,
            "network_mode": "none",
            "fresh_reset": True,
        }
    )


def _source_metadata() -> dict[str, Any]:
    """Read only existing attestations; never contact the target."""

    missing = [str(path.relative_to(ROOT)) for path in (RUNTIME_PATH, SOURCE_AUDIT_PATH, SOURCE_REPORT_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError("PG-368 source attestation missing: " + ", ".join(missing))
    audit = json.loads(SOURCE_AUDIT_PATH.read_text(encoding="utf-8-sig"))
    report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(audit, Mapping) or not isinstance(report, Mapping):
        raise ValueError("PG-333 attestation reports must be objects")
    if audit.get("status") != "passed":
        raise ValueError("PG-333 source audit is not passed")
    if report.get("status") != "completed_typed_method_shape_diagnostic_only":
        raise ValueError("PG-333 report status is not the reviewed method-shape canary")
    if RELAY_IMAGE != IMAGE or RELAY_IMAGE_DIGEST != IMAGE_DIGEST:
        raise ValueError("PG-333 relay image does not match the fixed PG-368 digest")
    return {
        "runtime_module_sha256": sha256_file(RUNTIME_PATH),
        "source_audit_sha256": sha256_file(SOURCE_AUDIT_PATH),
        "source_report_sha256": sha256_file(SOURCE_REPORT_PATH),
        "relay_image_digest_attested": True,
        "relay_route_path_sha256": sha256_json(RELAY_ROUTE_PATH),
        "source_audit_status": str(audit.get("status")),
        "source_report_status": str(report.get("status")),
        "source_report_hard_gate_status": str(dict(report.get("hard_gate") or {}).get("status", "unknown")),
        "source_report_counts": {
            "seed_count": int(dict(report.get("counts") or {}).get("seed_count", 0)),
            "route_count": int(dict(report.get("counts") or {}).get("route_count", 0)),
            "negative_violation_count": int(dict(report.get("counts") or {}).get("negative_violation_count", 0)),
        },
    }


def _role_contract(*, seed: int, route_ref: str, role: str, method: str) -> dict[str, Any]:
    stateful = False
    return {
        "target_identity_sha256": target_identity_sha256(seed=seed, route_ref=route_ref, role=role),
        "fresh_container_required": True,
        "fresh_reset_before_required": True,
        "fresh_reset_after_required": True,
        "state_reset_before_required": stateful,
        "state_reset_after_required": stateful,
        "database_clean_attestation_required": True,
        "teardown_required": True,
        "candidate_reference_negative_replay_required": True,
        "role_bound_evidence_sha256_required": True,
        "typed_oracle": {
            "status": "planned_unobserved",
            "method": method,
            "candidate_reference_required": role in SOURCE_ROLES,
            "negative_must_be_clean": role == "negative",
            "replay_must_match_candidate_shape": role == "replay",
            "effect_class": "html_page" if method == "GET" else "loopback_redirect",
        },
        "source_row_allowed": role in SOURCE_ROLES,
        "model_projection": {
            "question": "ask_typed",
            "next_action": "ask_typed",
            "repair_action": "observe",
            "transport_ref": "request_method",
            "parameter_role_ref": "credential_pair" if method == "POST" else "none",
            "encoding_ref": "form_urlencoded" if method == "POST" else "none",
            "response_shape_ref": "html_page" if method == "GET" else "loopback_redirect",
            "safe_to_send": False,
        },
        "training_eligible": False,
    }


def build_pg368_second_implementation_plan(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("PG-368 requires at least one seed")
    source = _source_metadata()
    if PG367_RUNTIME_PATH.exists() and sha256_file(RUNTIME_PATH) == sha256_file(PG367_RUNTIME_PATH):
        raise ValueError("PG-368 runtime hash matches PG-367; shared runtime cannot be independent")
    episodes: list[dict[str, Any]] = []
    for seed in normalized:
        for route in ROUTES:
            route_ref = route_ref_sha256(route)
            episodes.append(
                {
                    "seed": seed,
                    "route_ref_sha256": route_ref,
                    "method": route["method"],
                    "response_shape": route["response_shape"],
                    "roles": {role: _role_contract(seed=seed, route_ref=route_ref, role=role, method=route["method"]) for role in ROLES},
                    "typed_contract": {
                        "candidate_reference_negative_replay": True,
                        "fresh_reset_per_role": True,
                        "evidence_sha256": True,
                        "method_shape_only": True,
                        "vulnerability_oracle": False,
                    },
                }
            )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only",
        "implementation": {
            "implementation_id": PG368_IMPLEMENTATION_ID,
            "family": "WebGoat",
            "image": IMAGE,
            "image_digest": IMAGE_DIGEST,
            "runtime_module": "app/pg333_webgoat_docker_relay.py",
            "independent_from": PG367_IMPLEMENTATION_ID,
            "independence_contract": {
                "different_image_digest": True,
                "different_runtime_module": True,
                "different_process_boundary": True,
                "shared_runtime_forbidden": True,
                "shared_fixture_or_route_answer_forbidden": True,
            },
        },
        "source_attestation": source,
        "execution": {
            "docker_started": False,
            "network_contacted": False,
            "network_mode": "none",
            "loopback_relay_required": True,
            "published_ports_allowed": False,
            "bind_or_volume_mounts_allowed": False,
            "external_network": False,
            "fresh_disposable_container_per_seed_route_role": True,
            "execution_requires_explicit_operator_flag": "PG368_LOCAL_DOCKER_EVAL=1",
        },
        "seeds": list(normalized),
        "routes": [
            {
                "route_ref_sha256": route_ref_sha256(route),
                "method": route["method"],
                "response_shape": route["response_shape"],
                "path_literal_stored": False,
            }
            for route in ROUTES
        ],
        "episodes": episodes,
        "model_context_policy": {
            "abstract_tokens_only": True,
            "raw_payload_or_probe": False,
            "raw_request_or_response": False,
            "url_or_route_literal": False,
            "evaluator_answer_literal": False,
            "missing_observation_action": "ASK",
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "reason": "planning_only_until_live_typed_replay_and_cross_implementation_audit",
        },
        "interpretation": (
            "WebGoat is a source-attested second implementation candidate, not a vulnerability result. "
            "Existing PG-333/PG-342 method-shape evidence is diagnostic only; this plan must remain ASK/incomplete "
            "until a fresh candidate/reference/negative/replay run supplies role-bound typed evidence."
        ),
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def _find_forbidden_keys(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_find_forbidden_keys(item, path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, path=f"{path}[{index}]"))
    return found


def validate_pg368_second_implementation_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if plan.get("status") != "planning_only":
        failures.append("status")
    implementation = dict(plan.get("implementation") or {})
    if implementation.get("image") != IMAGE or implementation.get("image_digest") != IMAGE_DIGEST:
        failures.append("fixed_image_digest")
    if implementation.get("independent_from") != PG367_IMPLEMENTATION_ID:
        failures.append("independent_implementation_id")
    independent = dict(implementation.get("independence_contract") or {})
    for key in ("different_image_digest", "different_runtime_module", "different_process_boundary", "shared_runtime_forbidden", "shared_fixture_or_route_answer_forbidden"):
        if independent.get(key) is not True:
            failures.append(f"independence:{key}")
    execution = dict(plan.get("execution") or {})
    for key in ("docker_started", "network_contacted", "external_network", "published_ports_allowed", "bind_or_volume_mounts_allowed"):
        if execution.get(key) is not False:
            failures.append(f"execution:{key}")
    if execution.get("network_mode") != "none" or execution.get("fresh_disposable_container_per_seed_route_role") is not True:
        failures.append("execution_contract")
    routes = list(plan.get("routes") or [])
    if len(routes) != 2 or {str(route.get("method")) for route in routes} != {"GET", "POST"}:
        failures.append("get_post_routes")
    episodes = list(plan.get("episodes") or [])
    seeds = list(plan.get("seeds") or [])
    if len(episodes) != len(seeds) * 2:
        failures.append("episode_count")
    for episode in episodes:
        role_map = dict(episode.get("roles") or {})
        if set(role_map) != set(ROLES):
            failures.append("role_set")
            continue
        typed = dict(episode.get("typed_contract") or {})
        for key in ("candidate_reference_negative_replay", "fresh_reset_per_role", "evidence_sha256", "method_shape_only", "vulnerability_oracle"):
            expected = False if key == "vulnerability_oracle" else True
            if typed.get(key) is not expected:
                failures.append(f"typed_contract:{key}")
        for role, contract in role_map.items():
            if contract.get("fresh_container_required") is not True or contract.get("fresh_reset_before_required") is not True or contract.get("fresh_reset_after_required") is not True:
                failures.append(f"fresh_reset:{role}")
            if contract.get("role_bound_evidence_sha256_required") is not True or contract.get("training_eligible") is not False:
                failures.append(f"evidence_or_training:{role}")
            model_projection = dict(contract.get("model_projection") or {})
            if model_projection.get("safe_to_send") is not False or model_projection.get("question") != "ask_typed":
                failures.append(f"model_projection:{role}")
    promotion = dict(plan.get("promotion") or {})
    for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")
    forbidden = _find_forbidden_keys(plan)
    if forbidden:
        failures.append("forbidden_keys:" + ",".join(forbidden))
    return {"status": "passed" if not failures else "blocked", "failures": failures, "episode_count": len(episodes), "route_count": len(routes)}


def build_pg368_second_implementation_audit(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a read-only audit envelope for the planning artifact."""

    validation = validate_pg368_second_implementation_plan(plan)
    audit: dict[str, Any] = {
        "schema_version": "pg368-second-implementation-audit-v1",
        "status": "passed" if validation["status"] == "passed" else "blocked",
        "plan_sha256": str(plan.get("plan_sha256", "")),
        "validation": validation,
        "target_contacted": False,
        "docker_started": False,
        "network_contacted": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": "planning_only: no live typed evaluator evidence exists in this artifact",
    }
    audit["audit_sha256"] = sha256_json(audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "research" / "pg368_second_implementation_plan_v1.json"))
    parser.add_argument("--json", action="store_true", help="print the plan and validation result")
    args = parser.parse_args()
    plan = build_pg368_second_implementation_plan()
    audit = build_pg368_second_implementation_audit(plan)
    validation = dict(audit["validation"])
    Path(args.output).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = Path(args.output).with_name("pg368_second_implementation_audit_v1.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({"plan": plan, "audit": audit}, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
