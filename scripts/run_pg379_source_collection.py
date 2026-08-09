"""PG-379 planning-only/live-gated source collection contract.

This module is deliberately a contract and not a collector.  It describes the
future source/implementation-disjoint lane, validates de-identified
projections in memory through the PG-377 adapter, and keeps live execution
blocked until two independently attested local implementations are explicitly
bound.  No Docker client, socket, browser, GPU, or network operation is used
here.  In particular, the ``--live`` switch is an audit signal; it does not
override the unbound-implementation gate.

The contract keeps the distinction between a diagnostic projection and a
training source row.  A supplied HTML/projection can therefore be checked
against the seven-axis/107-field PG-331 contract without emitting or writing a
row.  Evaluator typed sidecars remain off-context and all promotion flags stay
false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg377_webgoat_source_row_adapter import (  # noqa: E402
    FIELD_COUNT,
    capture_pg377_webgoat_source_row,
    validate_pg377_webgoat_source_row,
)
from scripts.plan_pg379_source_collection import (  # noqa: E402
    METHODS,
    ROLES,
    ROUTE_SHAPES,
    SOURCE_ROLES,
    SLOTS,
    SEEDS,
    build_pg379_source_collection_plan,
    validate_pg379_source_collection_plan,
)


SCHEMA_VERSION = "pg379-source-collection-live-contract-v1"
OPERATOR_FLAG = "PG379_LOCAL_DOCKER_EVAL"
IMPLEMENTATION_KEYS = ("train", "holdout")
EXECUTION_FLAGS = (
    "docker_started",
    "network_contacted",
    "gpu_touched",
    "training_started",
    "rows_written",
    "split_relabelled",
)
PROMOTION_FLAGS = (
    "training_allowed",
    "memory_promotion_allowed",
    "payload_catalog_promotion_allowed",
    "vulnerability_claim_allowed",
)
ATTESTATION_FIELDS = (
    "image_digest",
    "runtime_module_sha256",
    "process_boundary_sha256",
    "source_digest",
    "authorization_id",
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_attestation(*, implementation_id: str, lane: str) -> dict[str, Any]:
    """Return an intentionally unbound attestation envelope.

    Empty values are important: a planned name must not be mistaken for an
    image/runtime/source identity.  A future live collector may replace this
    envelope only after an independent operator review and a separate runner
    contract; this module never performs that binding itself.
    """

    return {
        "implementation_id": implementation_id,
        "lane": lane,
        "bound": False,
        "attestation_status": "unbound",
        "image_digest": None,
        "runtime_module_sha256": None,
        "process_boundary_sha256": None,
        "source_digest": None,
        "authorization_id": None,
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "bind_or_volume_mounts_allowed": False,
        "fresh_reset_contract": False,
        "independent_source_review": False,
        "observed_fields": [],
        "side_effects_enabled": False,
    }


def _implementation_ids(plan: Mapping[str, Any]) -> dict[str, str]:
    requirements = dict(plan.get("new_implementation_requirements") or {})
    result: dict[str, str] = {}
    for lane in IMPLEMENTATION_KEYS:
        configured = dict(requirements.get(lane) or {})
        result[lane] = str(configured.get("implementation_id") or f"pg379_{lane}_unbound")
    return result


def _route_contract(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = list(plan.get("route_shape_matrix") or [])
    output: list[dict[str, Any]] = []
    for route in routes:
        output.append(
            {
                "route_class": str(route.get("route_class", "unknown")),
                "method": str(route.get("method", "unknown")).upper(),
                "parameter_role": str(route.get("parameter_role", "unknown")),
                "encoding_chain": str(route.get("encoding_chain", "unknown")),
                "response_shape": str(route.get("response_shape", "unknown")),
                "script_surface": str(route.get("script_surface", "unknown")),
                "route_ref_sha256": str(route.get("route_ref_sha256", "")),
                "target_slots_required": list(SLOTS),
                "roles_required": list(ROLES),
                "fresh_reset_before_after_each_role": True,
                "typed_candidate_reference_negative_replay": True,
                "failure_to_repair_belief_episode": True,
                "raw_route_literal_stored": False,
            }
        )
    return output


def _fresh_reset_contract() -> dict[str, Any]:
    return {
        "required": True,
        "before_each_role": True,
        "after_each_role": True,
        "network_mode": "none",
        "loopback_relay_only": True,
        "external_network": False,
        "bind_or_volume_mounts": False,
        "database_clean_attestation_when_stateful": True,
        "teardown_after_each_episode": True,
        "status": "blocked_unobserved",
    }


def _role_contract() -> dict[str, Any]:
    return {
        "candidate": {
            "source_row_required": True,
            "typed_candidate_projection_required": True,
            "role_bound_evidence_sha256_required": True,
            "negative_violation_allowed": False,
        },
        "reference": {
            "source_row_required": True,
            "typed_reference_projection_required": True,
            "role_bound_evidence_sha256_required": True,
            "negative_violation_allowed": False,
        },
        "negative": {
            "source_row_required": True,
            "typed_negative_projection_required": True,
            "role_bound_evidence_sha256_required": True,
            "negative_violation_allowed": False,
            "negative_violation_max": 0,
        },
        "replay": {
            "source_row_required": False,
            "typed_replay_projection_required": True,
            "role_bound_evidence_sha256_required": True,
            "sidecar_only": True,
            "negative_violation_allowed": False,
        },
    }


def _strict_gates() -> dict[str, Any]:
    return {
        "source_implementation_disjoint": {
            "required": True,
            "train_and_holdout_ids_differ": True,
            "image_runtime_source_attestation_required": True,
            "shared_fixture_or_route_answer_forbidden": True,
            "status": "blocked_unobserved",
        },
        "full_page_ontology": {
            "required": True,
            "axis_count": 7,
            "field_count": FIELD_COUNT,
            "unknown_or_not_observed_training_fields_max": 0,
            "status": "blocked_unobserved",
        },
        "rule_ir_target": {
            "required": True,
            "slot_count": len(SLOTS),
            "slots": list(SLOTS),
            "evaluator_answer_in_context": False,
            "status": "blocked_unobserved",
        },
        "get_post_balance": {
            "required": True,
            "methods": list(METHODS),
            "per_implementation": {"GET": 6, "POST": 6},
            "status": "blocked_unobserved",
        },
        "fresh_role_reset": _fresh_reset_contract(),
        "typed_roles": {
            "required": True,
            "roles": list(ROLES),
            "candidate_reference_negative_replay": True,
            "role_bound_evidence_sha256": True,
            "negative_violation_max": 0,
            "status": "blocked_unobserved",
        },
        "failure_repair_belief": {
            "required": True,
            "failure_feedback_axis": True,
            "failure_action_change_required": True,
            "repair_action_observed_required": True,
            "belief_prior_posterior_delta_required": True,
            "replay_state_required": True,
            "status": "blocked_unobserved",
        },
        "sidecar_context_firewall": {
            "required": True,
            "typed_sidecar_evaluator_only": True,
            "evidence_hash_evaluator_only": True,
            "oracle_answer_in_context": False,
            "raw_payload_response_wire_in_context": False,
            "status": "blocked_unobserved",
        },
    }


def _live_gate(*, live_requested: bool, environment: Mapping[str, str] | None) -> dict[str, Any]:
    env = dict(environment) if environment is not None else dict(os.environ)
    flag_present = str(env.get(OPERATOR_FLAG, "")) == "1"
    if not live_requested:
        return {
            "requested": False,
            "operator_flag": OPERATOR_FLAG,
            "operator_flag_present": flag_present,
            "ready": False,
            "status": "not_requested",
            "blocked_reasons": ["planning_only_mode"],
        }
    reasons = []
    if not flag_present:
        reasons.append(f"{OPERATOR_FLAG}=1_required")
    reasons.extend(
        [
            "independent_train_holdout_attestations_unbound",
            "collector_side_effects_disabled_in_contract",
        ]
    )
    return {
        "requested": True,
        "operator_flag": OPERATOR_FLAG,
        "operator_flag_present": flag_present,
        "ready": False,
        "status": "blocked_unbound_implementation_attestations",
        "blocked_reasons": reasons,
    }


def build_pg379_source_collection_contract(
    *,
    plan: Mapping[str, Any] | None = None,
    live_requested: bool = False,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the immutable, no-side-effect PG-379 collector contract."""

    selected_plan = dict(plan) if isinstance(plan, Mapping) else build_pg379_source_collection_plan()
    plan_validation = validate_pg379_source_collection_plan(selected_plan)
    if plan_validation.get("status") != "passed":
        raise ValueError(f"PG-379 source plan is invalid: {plan_validation.get('failures')}")
    ids = _implementation_ids(selected_plan)
    if ids["train"] == ids["holdout"]:
        raise ValueError("PG-379 train and holdout implementation IDs must differ")
    attestations = {
        lane: _empty_attestation(implementation_id=ids[lane], lane=lane) for lane in IMPLEMENTATION_KEYS
    }
    routes = _route_contract(selected_plan)
    method_counts = {method: sum(route["method"] == method for route in routes) for method in METHODS}
    configured_seeds = selected_plan.get("seeds")
    if not isinstance(configured_seeds, Sequence) or isinstance(configured_seeds, (str, bytes, bytearray)):
        planned_rows = list((selected_plan.get("planned_collections") or {}).get("train") or [])
        configured_seeds = sorted({int(row.get("seed")) for row in planned_rows if isinstance(row, Mapping) and row.get("seed") is not None})
    seeds = [int(seed) for seed in (configured_seeds or SEEDS)]
    scale = dict(selected_plan.get("expected_source_row_scale") or {})
    gate = _live_gate(live_requested=bool(live_requested), environment=environment)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only_live_blocked",
        "live_gate": gate,
        "plan": {
            "schema_version": selected_plan.get("schema_version"),
            "plan_sha256": str(selected_plan.get("plan_sha256", "")),
            "validation": plan_validation,
            "existing_split_relabelled": False,
            "existing_rows_reused_as_new_gold": False,
        },
        "independent_implementations": {
            "train": {
                "implementation_id": ids["train"],
                "attestation": attestations["train"],
                "required_attestation_fields": list(ATTESTATION_FIELDS),
                "independent_of": ids["holdout"],
            },
            "holdout": {
                "implementation_id": ids["holdout"],
                "attestation": attestations["holdout"],
                "required_attestation_fields": list(ATTESTATION_FIELDS),
                "independent_of": ids["train"],
            },
        },
        "route_contract": {
            "routes": routes,
            "route_count": len(routes),
            "method_counts": method_counts,
            "per_implementation": {lane: dict(method_counts) for lane in IMPLEMENTATION_KEYS},
            "get_routes_required": 6,
            "post_routes_required": 6,
        },
        "role_contract": _role_contract(),
        "fresh_role_reset": _fresh_reset_contract(),
        "failure_repair_belief_contract": {
            "failure_feedback_required": True,
            "failure_action_change_required": True,
            "repair_action_required": True,
            "belief_prior_posterior_delta_required": True,
            "replay_state_required": True,
            "role_bound_evidence_required": True,
        },
        "sidecar_context_firewall": {
            "typed_sidecar_evaluator_only": True,
            "evidence_sha256_evaluator_only": True,
            "oracle_answer_in_context": False,
            "raw_payload_response_wire_in_context": False,
            "context_tokens_exclude_sidecar": True,
        },
        "rule_ir_target": {"slot_count": len(SLOTS), "slots": list(SLOTS)},
        "strict_gates": _strict_gates(),
        "expected_scale": {
            "seeds": seeds,
            "seed_count": len(seeds),
            "source_roles": list(SOURCE_ROLES),
            "roles": list(ROLES),
            "source_rows_per_implementation": int(scale.get("source_rows_per_implementation", 0)),
            "planned_source_rows_total": int(scale.get("planned_source_rows_total", 0)),
            "planned_role_episode_rows_total": int(scale.get("planned_role_episode_rows_total", 0)),
            "planned_failure_repair_pairs_total": int(scale.get("planned_failure_repair_pairs_total", 0)),
        },
        # An empty list is an explicit statement that this module did not emit
        # source rows.  It is not a training dataset or a gold artifact.
        "diagnostic_projections": [],
        "rows_emitted": False,
        "rows_emitted_count": 0,
        "execution": {key: False for key in EXECUTION_FLAGS},
        "promotion": {key: False for key in PROMOTION_FLAGS},
        "blocked_reasons": [
            "train_and_holdout_image_runtime_source_attestations_unbound",
            "fresh_get_post_role_replay_unobserved",
            "full_107_field_rows_unobserved",
            "13_slot_targets_unobserved",
            "collector_has_no_live_side_effects",
        ],
    }
    report["contract_sha256"] = _sha256_json(report)
    return report


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-379 {name} must be an object")
    return value


def validate_pg379_source_collection_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the no-side-effect contract and its hard closed gates."""

    failures: list[str] = []
    if not isinstance(contract, Mapping):
        return {"status": "blocked", "failures": ["contract_not_mapping"]}
    if contract.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if contract.get("status") != "planning_only_live_blocked":
        failures.append("status")
    plan = _mapping(contract.get("plan"), "plan")
    if _mapping(plan.get("validation"), "plan.validation").get("status") != "passed":
        failures.append("plan_validation")
    implementations = _mapping(contract.get("independent_implementations"), "independent_implementations")
    if set(implementations) != set(IMPLEMENTATION_KEYS):
        failures.append("implementation_keys")
    ids: dict[str, str] = {}
    for lane in IMPLEMENTATION_KEYS:
        entry = _mapping(implementations.get(lane), f"implementation:{lane}")
        ids[lane] = str(entry.get("implementation_id", ""))
        attestation = _mapping(entry.get("attestation"), f"attestation:{lane}")
        if attestation.get("bound") is not False or attestation.get("attestation_status") != "unbound":
            failures.append(f"attestation:{lane}:bound")
        if any(attestation.get(field) is not None for field in ATTESTATION_FIELDS):
            failures.append(f"attestation:{lane}:identity_present")
        if attestation.get("side_effects_enabled") is not False:
            failures.append(f"attestation:{lane}:side_effects")
        if str(entry.get("independent_of", "")) == ids[lane]:
            failures.append(f"attestation:{lane}:independence")
    if not ids.get("train") or not ids.get("holdout") or ids.get("train") == ids.get("holdout"):
        failures.append("implementation_disjoint")
    route_contract = _mapping(contract.get("route_contract"), "route_contract")
    routes = list(route_contract.get("routes") or [])
    if len(routes) != len(ROUTE_SHAPES) or route_contract.get("route_count") != len(ROUTE_SHAPES):
        failures.append("route_count")
    method_counts = {method: sum(str(route.get("method", "")).upper() == method for route in routes) for method in METHODS}
    if method_counts != {"GET": 6, "POST": 6} or route_contract.get("method_counts") != method_counts:
        failures.append("route_balance")
    expected_route_keys = {(str(route.get("route_class")), str(route.get("method")).upper()) for route in ROUTE_SHAPES}
    actual_route_keys = {(str(route.get("route_class")), str(route.get("method")).upper()) for route in routes}
    if actual_route_keys != expected_route_keys:
        failures.append("route_shape_matrix")
    for route in routes:
        if list(route.get("target_slots_required") or []) != list(SLOTS):
            failures.append("route_slots")
        if list(route.get("roles_required") or []) != list(ROLES):
            failures.append("route_roles")
        if route.get("fresh_reset_before_after_each_role") is not True or route.get("typed_candidate_reference_negative_replay") is not True or route.get("failure_to_repair_belief_episode") is not True:
            failures.append("route_episode_contract")
        if route.get("raw_route_literal_stored") is not False:
            failures.append("route_literal")
    role_contract = _mapping(contract.get("role_contract"), "role_contract")
    if set(role_contract) != set(ROLES):
        failures.append("roles")
    for role in ROLES:
        entry = _mapping(role_contract.get(role), f"role:{role}")
        if entry.get("typed_" + role + "_projection_required") is not True:
            failures.append(f"role:{role}:typed")
        if entry.get("role_bound_evidence_sha256_required") is not True:
            failures.append(f"role:{role}:evidence")
        if entry.get("negative_violation_allowed") is not False:
            failures.append(f"role:{role}:negative")
    reset = _mapping(contract.get("fresh_role_reset"), "fresh_role_reset")
    for key, expected in (("before_each_role", True), ("after_each_role", True), ("external_network", False), ("loopback_relay_only", True), ("bind_or_volume_mounts", False), ("teardown_after_each_episode", True)):
        if reset.get(key) is not expected:
            failures.append(f"reset:{key}")
    if reset.get("network_mode") != "none":
        failures.append("reset:network_mode")
    sidecar = _mapping(contract.get("sidecar_context_firewall"), "sidecar_context_firewall")
    for key, expected in (("typed_sidecar_evaluator_only", True), ("evidence_sha256_evaluator_only", True), ("oracle_answer_in_context", False), ("raw_payload_response_wire_in_context", False), ("context_tokens_exclude_sidecar", True)):
        if sidecar.get(key) is not expected:
            failures.append(f"sidecar:{key}")
    failure_contract = _mapping(contract.get("failure_repair_belief_contract"), "failure_repair_belief_contract")
    for key in ("failure_feedback_required", "failure_action_change_required", "repair_action_required", "belief_prior_posterior_delta_required", "replay_state_required", "role_bound_evidence_required"):
        if failure_contract.get(key) is not True:
            failures.append(f"failure_repair:{key}")
    rule_ir = _mapping(contract.get("rule_ir_target"), "rule_ir_target")
    if rule_ir.get("slot_count") != len(SLOTS) or list(rule_ir.get("slots") or []) != list(SLOTS):
        failures.append("rule_ir_slots")
    execution = _mapping(contract.get("execution"), "execution")
    for key in EXECUTION_FLAGS:
        if execution.get(key) is not False:
            failures.append(f"execution:{key}")
    promotion = _mapping(contract.get("promotion"), "promotion")
    for key in PROMOTION_FLAGS:
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")
    if contract.get("rows_emitted") is not False or contract.get("rows_emitted_count") != 0 or list(contract.get("diagnostic_projections") or []):
        failures.append("rows_emitted")
    expected_hash = str(contract.get("contract_sha256", ""))
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        failures.append("contract_hash")
    else:
        body = dict(contract)
        body.pop("contract_sha256", None)
        # ``main`` appends a read-only validation summary for CLI consumers;
        # it is derived metadata and is intentionally outside the immutable
        # contract hash.
        body.pop("validation", None)
        if _sha256_json(body) != expected_hash:
            failures.append("contract_hash_mismatch")
    return {
        "status": "passed" if not failures else "blocked",
        "failures": sorted(set(failures)),
        "route_count": len(routes),
        "get_routes": method_counts.get("GET", 0),
        "post_routes": method_counts.get("POST", 0),
        "field_count": FIELD_COUNT,
        "slot_count": len(SLOTS),
    }


def _route_shape(route_class: str) -> Mapping[str, Any]:
    for route in ROUTE_SHAPES:
        if str(route.get("route_class")) == str(route_class):
            return route
    raise ValueError(f"PG-379 route_class is not in the abstract 12-route matrix: {route_class}")


def validate_pg379_projection(
    *,
    implementation: str,
    route_class: str,
    seed: int,
    role: str,
    html: str | None,
    headers: Mapping[str, Any] | None = None,
    request_projection: Mapping[str, Any] | None = None,
    response_projection: Mapping[str, Any] | None = None,
    reset: Mapping[str, Any] | None = None,
    evaluator_sidecar: Mapping[str, Any] | None = None,
    failure_projection: Mapping[str, Any] | None = None,
    belief_projection: Mapping[str, Any] | None = None,
    attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one de-identified projection without emitting a source row.

    The returned ``diagnostic_row`` is an in-memory PG-377 adapter view only;
    ``rows_emitted`` and ``training_eligible`` remain false.  This helper never
    binds an image or implementation, and callers cannot use it to bypass the
    live gate.
    """

    route = _route_shape(route_class)
    if role not in ROLES:
        raise ValueError("PG-379 role must be candidate, reference, negative, or replay")
    if int(seed) not in {int(value) for value in SEEDS}:
        raise ValueError("PG-379 seed is not in the planned seed set")
    expected_ids = _implementation_ids(build_pg379_source_collection_plan())
    lane = next((key for key, value in expected_ids.items() if value == implementation), None)
    if lane is None and implementation in IMPLEMENTATION_KEYS:
        lane = implementation
        implementation_id = expected_ids[lane]
    elif lane is not None:
        implementation_id = implementation
    else:
        raise ValueError("PG-379 implementation is not one of the two planned independent IDs")
    attestation_value = dict(attestation) if isinstance(attestation, Mapping) else _empty_attestation(implementation_id=implementation_id, lane=lane)
    if str(attestation_value.get("implementation_id", implementation_id)) != implementation_id:
        raise ValueError("PG-379 projection attestation implementation mismatch")
    if attestation_value.get("bound") is not False:
        raise ValueError("PG-379 projection requires an unbound diagnostic attestation")
    method = str(route.get("method", "")).upper()
    if isinstance(request_projection, Mapping) and request_projection.get("method") is not None and str(request_projection.get("method")).upper() != method:
        raise ValueError("PG-379 projection method does not match route shape")
    try:
        row = capture_pg377_webgoat_source_row(
            html=html,
            headers=headers,
            request_projection=request_projection,
            response_projection=response_projection,
            role=role,
            reset=reset,
            evaluator_sidecar=evaluator_sidecar,
            failure_projection=failure_projection,
            belief_projection=belief_projection,
            source_meta=None,
            record_id=None,
        )
    except (TypeError, ValueError) as error:
        return {
            "status": "projection_blocked",
            "implementation": implementation_id,
            "lane": lane,
            "route_class": route_class,
            "method": method,
            "seed": int(seed),
            "role": role,
            "rows_emitted": False,
            "training_eligible": False,
            "blocked_reasons": [f"adapter_rejected:{error}"],
            "diagnostic_row": None,
            "adapter_validation": {"valid": False, "failures": ["adapter_rejected"]},
        }
    adapter_validation = validate_pg377_webgoat_source_row(row)
    blocked_reasons: list[str] = []
    if row.get("method") != method:
        blocked_reasons.append("method_not_observed_or_mismatch")
    if reset is None:
        blocked_reasons.append("fresh_role_reset_missing")
    elif row.get("reset_attestation", {}).get("attested") is not True:
        blocked_reasons.append("fresh_role_reset_not_attested")
    if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        blocked_reasons.append("sidecar_context_firewall")
    if not adapter_validation.get("valid"):
        blocked_reasons.append("pg377_adapter_contract")
    return {
        "status": "projection_validated_diagnostic" if not blocked_reasons else "projection_blocked",
        "implementation": implementation_id,
        "lane": lane,
        "route_class": route_class,
        "method": method,
        "seed": int(seed),
        "role": role,
        "rows_emitted": False,
        "training_eligible": False,
        "blocked_reasons": blocked_reasons,
        "diagnostic_row": row,
        "adapter_validation": adapter_validation,
        "attestation_bound": False,
        "sidecar_off_context": row.get("context_firewall") == {"forbidden_token_count": 0, "sidecars_off_context": True},
    }


# Short aliases keep the contract easy to discover without coupling callers
# to the historical PG-379 planner naming.
build_contract = build_pg379_source_collection_contract
build_pg379_collector_contract = build_pg379_source_collection_contract
validate_contract = validate_pg379_source_collection_contract
validate_pg379_collector_contract = validate_pg379_source_collection_contract
validate_projection = validate_pg379_projection
collect_pg379_source_projection = validate_pg379_projection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="record a live request, which remains blocked until attestations are bound")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--json", action="store_true", help="print the contract JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_pg379_source_collection_contract(live_requested=bool(args.live))
    validation = validate_pg379_source_collection_contract(report)
    report["validation"] = validation
    # Hash excludes the post-build validation field; the immutable contract
    # hash remains available for audit and the report is still deterministic.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json or args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation.get("status") == "passed" else 2


if __name__ == "__main__":  # pragma: no cover - command-line smoke only
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_FIELDS",
    "IMPLEMENTATION_KEYS",
    "SCHEMA_VERSION",
    "build_contract",
    "build_pg379_collector_contract",
    "build_pg379_source_collection_contract",
    "collect_pg379_source_projection",
    "validate_contract",
    "validate_pg379_collector_contract",
    "validate_pg379_projection",
    "validate_pg379_source_collection_contract",
    "validate_projection",
]
