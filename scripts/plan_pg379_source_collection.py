"""PG-379 planning-only source/implementation-disjoint collection matrix.

This planner reads only existing attestation/row metadata.  It never starts a
container, opens a socket, reads a page, or creates a training row.  Existing
PG-333 and PG-377 splits are immutable: their counts and hashes are reported as
baseline evidence, while the proposed PG-379 collection uses a separate
``planned_collection_split`` field and cannot relabel those artifacts.

The proposed lane deliberately targets real dynamic GET/POST pages on two
future, independently attested local implementations.  Each implementation
has the same abstract route *shape* matrix (six GET and six POST classes), but
the implementation/image/runtime/route source must be independently reviewed.
Every future route/seed requires candidate/reference/negative plus replay and a
complete 13-slot Rule-IR target.  A route is not training-eligible merely
because it appears in the existing dynamic fixture registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plan_pg369_multitask_moe_candidate import SLOTS  # noqa: E402


SCHEMA_VERSION = "pg379-source-collection-matrix-plan-v1"
SEEDS = (37901, 37902, 37903)
ROLES = ("candidate", "reference", "negative", "replay")
SOURCE_ROLES = ("candidate", "reference", "negative")
METHODS = ("GET", "POST")

PG333_PATH = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json"
PG377_PATH = ROOT / "research" / "pg377_webgoat_source_rows_live_v2.json"
DYNAMIC_REGISTRY_PATH = ROOT / "fixtures" / "pg348" / "registry_v1.json"
DEFAULT_PG333 = PG333_PATH
DEFAULT_PG377 = PG377_PATH
DEFAULT_DYNAMIC_REGISTRY = DYNAMIC_REGISTRY_PATH

PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}

# These are abstract shape coordinates only.  No path, marker, body, route
# literal, or evaluator answer is included in the plan.
ROUTE_SHAPES: tuple[dict[str, Any], ...] = (
    {"route_class": "get_query_html_text", "method": "GET", "parameter_role": "query_text", "encoding_chain": "url_percent", "response_shape": "html_text", "script_surface": "none"},
    {"route_class": "get_path_dom_text", "method": "GET", "parameter_role": "path_segment", "encoding_chain": "identity", "response_shape": "html_dom_text", "script_surface": "inline_dom_text"},
    {"route_class": "get_fragment_js_navigation", "method": "GET", "parameter_role": "fragment_identifier", "encoding_chain": "fragment", "response_shape": "html_fragment", "script_surface": "spa_navigation"},
    {"route_class": "get_json_shape", "method": "GET", "parameter_role": "json_value", "encoding_chain": "json_string", "response_shape": "json_shape", "script_surface": "inline_json_data"},
    {"route_class": "get_redirect_control", "method": "GET", "parameter_role": "view_mode", "encoding_chain": "query_parameter", "response_shape": "redirect_shape", "script_surface": "history_navigation"},
    {"route_class": "get_failure_feedback", "method": "GET", "parameter_role": "query_term", "encoding_chain": "form_urlencoded", "response_shape": "error_shape", "script_surface": "none"},
    {"route_class": "post_form_dom_update", "method": "POST", "parameter_role": "form_field", "encoding_chain": "form_urlencoded", "response_shape": "html_dom_text", "script_surface": "inline_dom_text"},
    {"route_class": "post_json_state_transition", "method": "POST", "parameter_role": "json_value", "encoding_chain": "json_object_then_utf8", "response_shape": "state_delta", "script_surface": "module_fetch"},
    {"route_class": "post_redirect_control", "method": "POST", "parameter_role": "view_mode", "encoding_chain": "form_urlencoded_then_url_percent", "response_shape": "redirect_shape", "script_surface": "history_navigation"},
    {"route_class": "post_attribute_shape", "method": "POST", "parameter_role": "attribute_value", "encoding_chain": "form_urlencoded", "response_shape": "html_attribute", "script_surface": "none"},
    {"route_class": "post_parser_failure", "method": "POST", "parameter_role": "structured_value", "encoding_chain": "json_object_then_utf8", "response_shape": "error_shape", "script_surface": "dialog_shape"},
    {"route_class": "post_replay_shape", "method": "POST", "parameter_role": "record_cursor", "encoding_chain": "query_parameter_then_url_percent", "response_shape": "replay_shape", "script_surface": "module_fetch"},
)

_FORBIDDEN_KEYS = frozenset(
    {
        "url",
        "uri",
        "payload",
        "raw_payload",
        "request_body",
        "response_body",
        "raw_response",
        "wire",
        "evaluator_answer",
        "oracle_answer",
        "route_literal",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> tuple[Mapping[str, Any], str]:
    resolved = path.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value, sha256_file(resolved)


def _records(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = document.get("records")
    if not isinstance(values, list):
        raise ValueError("artifact records must be a list")
    return [value for value in values if isinstance(value, Mapping)]


def _implementation(row: Mapping[str, Any]) -> str:
    meta = row.get("source_meta")
    return str(meta.get("implementation", "unknown")) if isinstance(meta, Mapping) else "unknown"


def _method(row: Mapping[str, Any]) -> str:
    target = row.get("target_projection")
    if isinstance(target, Mapping):
        transport = str(target.get("transport_ref", "")).casefold()
        if transport.startswith("post"):
            return "POST"
        if transport.startswith("get"):
            return "GET"
    context = row.get("context_tokens") or []
    for token in context:
        if str(token).casefold() in {"request_method=get", "request_method=post"}:
            return str(token).split("=", 1)[1].upper()
    return "UNKNOWN"


def _artifact_summary(path: Path, *, kind: str) -> dict[str, Any]:
    document, digest = _load_json(path)
    rows = _records(document)
    implementations = Counter(_implementation(row) for row in rows)
    splits = Counter(str(row.get("split", "unknown")) for row in rows)
    methods = Counter(_method(row) for row in rows)
    eligible = Counter(_implementation(row) for row in rows if row.get("training_eligible") is True)
    target_lengths = Counter(len(row.get("target_tokens") or []) for row in rows)
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "file_sha256": digest,
        "schema_version": str(document.get("schema_version", "")),
        "status": str(document.get("status", "")),
        "record_count": len(rows),
        "implementation_counts": dict(sorted(implementations.items())),
        "split_counts": dict(sorted(splits.items())),
        "method_counts": dict(sorted(methods.items())),
        "training_eligible_by_implementation": dict(sorted(eligible.items())),
        "target_length_counts": dict(sorted((str(key), value) for key, value in target_lengths.items())),
        "split_relabelled": False,
        "rows_emitted": False,
    }


def _registry_summary(path: Path) -> dict[str, Any]:
    document, digest = _load_json(path)
    records = _records(document)
    implementations = Counter(str(row.get("implementation_group", "unknown")) for row in records)
    methods = Counter(str(row.get("transport_method", "unknown")).upper() for row in records)
    mechanism = {str(row.get("mechanism_id", "unknown")) for row in records}
    templates = {str(row.get("surface_template_id", "unknown")) for row in records}
    return {
        "path": str(path.resolve()),
        "file_sha256": digest,
        "schema_version": str(document.get("schema_version", "")),
        "status": str(document.get("status", "")),
        "record_count": len(records),
        "implementation_group_counts": dict(sorted(implementations.items())),
        "method_counts": dict(sorted(methods.items())),
        "mechanism_family_count": len(mechanism),
        "surface_template_count": len(templates),
        "all_loopback_only": bool((document.get("counts") or {}).get("all_urls_loopback_only", False)),
        "external_network_records": int((document.get("counts") or {}).get("external_network_records", 0) or 0),
        "state_write_records": int((document.get("counts") or {}).get("state_write_records", 0) or 0),
        "inventory_only": True,
        "training_rows_emitted": False,
    }


def _route_ref(route: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "schema": SCHEMA_VERSION,
            "route_class": str(route["route_class"]),
            "method": str(route["method"]),
            "parameter_role": str(route["parameter_role"]),
            "encoding_chain": str(route["encoding_chain"]),
            "response_shape": str(route["response_shape"]),
            "script_surface": str(route["script_surface"]),
        }
    )


def _planned_route_matrix(*, implementation: str, planned_split: str, seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in ROUTE_SHAPES:
        route_ref = _route_ref(route)
        for seed in seeds:
            rows.append(
                {
                    "implementation": implementation,
                    "planned_collection_split": planned_split,
                    "seed": int(seed),
                    "route_ref_sha256": route_ref,
                    "method": str(route["method"]),
                    "route_class": str(route["route_class"]),
                    "parameter_role": str(route["parameter_role"]),
                    "encoding_chain": str(route["encoding_chain"]),
                    "response_shape": str(route["response_shape"]),
                    "script_surface": str(route["script_surface"]),
                    "roles": list(ROLES),
                    "source_roles": list(SOURCE_ROLES),
                    "target_slots_required": list(SLOTS),
                    "fresh_reset_per_role": True,
                    "typed_candidate_reference_negative_replay": True,
                    "failure_repair_episode_required": True,
                    "training_eligible_before_audit": False,
                }
            )
    return rows


def _strict_gates() -> dict[str, Any]:
    return {
        "source_implementation_disjoint": {
            "required": True,
            "train_and_holdout_implementation_ids_differ": True,
            "image_digest_differ_or_independent_attestation": True,
            "runtime_module_and_process_boundary_differ": True,
            "shared_fixture_route_answer_forbidden": True,
            "status": "planned_unobserved",
        },
        "split_immutability": {
            "required": True,
            "existing_pg333_pg377_split_preserved": True,
            "relabel_existing_rows": False,
            "status": "planned_unobserved",
        },
        "full_page_ontology": {
            "required": True,
            "axis_count": 7,
            "field_count": 107,
            "unknown_or_not_observed_training_fields": 0,
            "context_firewall_forbidden_tokens": 0,
            "status": "planned_unobserved",
        },
        "rule_ir_target": {
            "required": True,
            "slot_count": len(SLOTS),
            "slots": list(SLOTS),
            "target_evaluator_answer_in_context": False,
            "status": "planned_unobserved",
        },
        "get_post_balance": {
            "required": True,
            "methods": list(METHODS),
            "per_implementation_route_balance": {"GET": 6, "POST": 6},
            "status": "planned_unobserved",
        },
        "role_typed_evidence": {
            "required": True,
            "roles": list(ROLES),
            "candidate_reference_required": True,
            "negative_violation_max": 0,
            "replay_consistent_required": True,
            "role_bound_evidence_sha256_required": True,
            "status": "planned_unobserved",
        },
        "failure_repair": {
            "required": True,
            "failure_action_change_required": True,
            "repair_action_observed_required": True,
            "failure_rows_without_action_change_max": 0,
            "status": "planned_unobserved",
        },
        "fresh_local_safety": {
            "required": True,
            "network_mode": "none",
            "loopback_relay_only": True,
            "external_network": False,
            "bind_or_volume_mounts": False,
            "fresh_reset_before_after_each_role": True,
            "database_clean_attestation_when_stateful": True,
            "teardown_after_each_episode": True,
            "status": "planned_unobserved",
        },
        "capacity_and_replay": {
            "required": True,
            "no_silent_context_truncation": True,
            "required_window_measured_before_training": True,
            "fresh_replay_recorded": True,
            "evidence_sha256_required": True,
            "status": "planned_unobserved",
        },
    }


def build_pg379_source_collection_plan(
    *,
    pg333_path: Path = PG333_PATH,
    pg377_path: Path = PG377_PATH,
    dynamic_registry_path: Path = DYNAMIC_REGISTRY_PATH,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("PG-379 requires at least one seed")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("PG-379 seeds must be unique")
    existing = {
        "pg333_three_impl_rows": _artifact_summary(pg333_path, kind="pg333_three_impl_rows"),
        "pg377_webgoat_rows": _artifact_summary(pg377_path, kind="pg377_webgoat_rows"),
    }
    registry = _registry_summary(dynamic_registry_path)
    registry["known_lab_inventory"] = [
        {"implementation_id": "pikachu-fixed", "evidence_source": "pg333", "planning_role": "existing_train_baseline", "rows_emitted": False},
        {"implementation_id": "webgoat", "evidence_source": "pg333_pg377", "planning_role": "existing_implementation_holdout", "rows_emitted": False},
        {"implementation_id": "vulnerables-web-dvwa", "evidence_source": "pg333", "planning_role": "existing_implementation_holdout", "rows_emitted": False},
        {"implementation_id": "juice-shop", "evidence_source": "prior_evaluation_only", "planning_role": "canary_only_not_source_row", "rows_emitted": False},
        {"implementation_id": "pg348_pages_a_b_c", "evidence_source": "pg348_registry", "planning_role": "synthetic_dynamic_inventory_only", "rows_emitted": False},
    ]
    train_impl = "pg379_dynamic_real_train_impl_a"
    holdout_impl = "pg379_dynamic_real_holdout_impl_b"
    train_matrix = _planned_route_matrix(implementation=train_impl, planned_split="new_train_collection", seeds=normalized_seeds)
    holdout_matrix = _planned_route_matrix(implementation=holdout_impl, planned_split="new_implementation_holdout_collection", seeds=normalized_seeds)
    routes_per_impl = len(ROUTE_SHAPES)
    source_rows_per_impl = routes_per_impl * len(normalized_seeds) * len(SOURCE_ROLES)
    replay_rows_per_impl = routes_per_impl * len(normalized_seeds)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only_blocked",
        "objective": {
            "source_implementation_disjoint_train_holdout": True,
            "real_dynamic_get_post_priority": True,
            "full_page_pg331_axes": 7,
            "field_capture_manifest_fields": 107,
            "rule_ir_target_slots": len(SLOTS),
            "existing_split_relabelled": False,
            "training_rows_created": False,
        },
        "immutable_existing_artifacts": existing,
        "dynamic_page_and_lab_inventory": registry,
        "existing_matrix": {
            "pg333_train_implementation": "pikachu-fixed",
            "pg333_holdout_implementations": ["vulnerables-web-dvwa", "webgoat"],
            "pg377_holdout_implementation": "webgoat",
            "existing_rows_are_read_only": True,
            "existing_pg377_operator_reviewed": False,
            "existing_pg377_training_eligible": False,
            "pg348_registry_is_inventory_only": True,
        },
        "new_implementation_requirements": {
            "train": {
                "implementation_id": train_impl,
                "attestation": ["new_image_digest", "new_runtime_module", "new_process_boundary", "authorized_local_loopback", "source_digest"],
                "shared_fixture_or_route_answer_forbidden": True,
            },
            "holdout": {
                "implementation_id": holdout_impl,
                "attestation": ["different_image_or_independent_runtime_attestation", "different_runtime_module", "different_process_boundary", "authorized_local_loopback", "source_digest"],
                "shared_fixture_or_route_answer_forbidden": True,
            },
        },
        "route_shape_matrix": [
            {
                **{key: value for key, value in route.items()},
                "route_ref_sha256": _route_ref(route),
                "target_slots_required": list(SLOTS),
                "raw_route_literal_stored": False,
            }
            for route in ROUTE_SHAPES
        ],
        "planned_collections": {
            "train": train_matrix,
            "implementation_holdout": holdout_matrix,
        },
        "strict_gates": _strict_gates(),
        "expected_source_row_scale": {
            "seeds": len(normalized_seeds),
            "routes_per_implementation": routes_per_impl,
            "get_routes_per_implementation": sum(route["method"] == "GET" for route in ROUTE_SHAPES),
            "post_routes_per_implementation": sum(route["method"] == "POST" for route in ROUTE_SHAPES),
            "source_roles_per_route": len(SOURCE_ROLES),
            "roles_per_route": len(ROLES),
            "source_rows_per_implementation": source_rows_per_impl,
            "replay_sidecar_rows_per_implementation": replay_rows_per_impl,
            "planned_train_source_rows": source_rows_per_impl,
            "planned_holdout_source_rows": source_rows_per_impl,
            "planned_source_rows_total": source_rows_per_impl * 2,
            "planned_role_episode_rows_total": routes_per_impl * len(normalized_seeds) * len(ROLES) * 2,
            "planned_failure_repair_pairs_total": routes_per_impl * len(normalized_seeds) * 2,
            "training_eligible_before_independent_audit": 0,
        },
        "execution": {
            "docker_started": False,
            "network_contacted": False,
            "gpu_touched": False,
            "training_started": False,
            "rows_written": False,
            "split_relabelled": False,
            "network_mode_required_for_future_live_run": "none",
            "explicit_operator_flag_required": "PG379_LOCAL_DOCKER_EVAL=1",
        },
        "promotion": dict(PROMOTION),
        "blocked_reasons": [
            "new_train_and_holdout_implementations_not_yet_attested",
            "future_fresh_get_post_role_replay_unobserved",
            "future_full_107_field_source_rows_unobserved",
            "future_13_slot_targets_unobserved",
            "existing_pg377_rows_holdout_only_and_operator_review_false",
            "dynamic_registry_is_inventory_only_not_training_gold",
        ],
        "interpretation": (
            "PG-379 is a collection plan, not a split operation or dataset. Existing PG-333/PG-377 rows retain their "
            "original split and status. New real dynamic implementations must independently supply complete seven-axis "
            "rows, GET/POST pairs, all 13 Rule-IR slots, typed candidate/reference/negative/replay evidence and "
            "failure-to-repair transitions before any source audit or training request."
        ),
    }
    report["plan_sha256"] = sha256_json(report)
    return report


def _find_forbidden_keys(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_find_forbidden_keys(item, path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, path=f"{path}[{index}]"))
    return found


def validate_pg379_source_collection_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if plan.get("status") != "planning_only_blocked":
        failures.append("status")
    objective = dict(plan.get("objective") or {})
    for key in ("source_implementation_disjoint_train_holdout", "real_dynamic_get_post_priority", "existing_split_relabelled", "training_rows_created"):
        expected = False if key in {"existing_split_relabelled", "training_rows_created"} else True
        if objective.get(key) is not expected:
            failures.append(f"objective:{key}")
    if int(objective.get("full_page_pg331_axes", 0)) != 7 or int(objective.get("field_capture_manifest_fields", 0)) != 107 or int(objective.get("rule_ir_target_slots", 0)) != len(SLOTS):
        failures.append("ontology_contract")
    existing = dict(plan.get("immutable_existing_artifacts") or {})
    for key in ("pg333_three_impl_rows", "pg377_webgoat_rows"):
        if dict(existing.get(key) or {}).get("split_relabelled") is not False or dict(existing.get(key) or {}).get("rows_emitted") is not False:
            failures.append(f"existing_immutable:{key}")
    routes = list(plan.get("route_shape_matrix") or [])
    if len(routes) != len(ROUTE_SHAPES) or {str(route.get("method")) for route in routes} != set(METHODS):
        failures.append("route_shape_get_post")
    if sum(str(route.get("method")) == "GET" for route in routes) != 6 or sum(str(route.get("method")) == "POST" for route in routes) != 6:
        failures.append("route_balance")
    for route in routes:
        if route.get("raw_route_literal_stored") is not False or list(route.get("target_slots_required") or []) != list(SLOTS):
            failures.append("route_target_contract")
    collections = dict(plan.get("planned_collections") or {})
    train = list(collections.get("train") or [])
    holdout = list(collections.get("implementation_holdout") or [])
    if not train or not holdout:
        failures.append("planned_collections_empty")
    train_impls = {str(row.get("implementation")) for row in train}
    holdout_impls = {str(row.get("implementation")) for row in holdout}
    if not train_impls or not holdout_impls or train_impls & holdout_impls:
        failures.append("planned_implementation_overlap")
    for row in [*train, *holdout]:
        if row.get("planned_collection_split") not in {"new_train_collection", "new_implementation_holdout_collection"}:
            failures.append("planned_split_label")
        if list(row.get("roles") or []) != list(ROLES) or list(row.get("target_slots_required") or []) != list(SLOTS):
            failures.append("planned_role_or_slot_contract")
        if row.get("fresh_reset_per_role") is not True or row.get("typed_candidate_reference_negative_replay") is not True or row.get("failure_repair_episode_required") is not True:
            failures.append("planned_fresh_typed_failure_contract")
        if row.get("training_eligible_before_audit") is not False:
            failures.append("planned_training_open")
    gates = dict(plan.get("strict_gates") or {})
    for gate_name in ("source_implementation_disjoint", "split_immutability", "full_page_ontology", "rule_ir_target", "get_post_balance", "role_typed_evidence", "failure_repair", "fresh_local_safety", "capacity_and_replay"):
        gate = dict(gates.get(gate_name) or {})
        if gate.get("required") is not True or gate.get("status") != "planned_unobserved":
            failures.append(f"gate:{gate_name}")
    scale = dict(plan.get("expected_source_row_scale") or {})
    if int(scale.get("planned_train_source_rows", 0)) <= 0 or int(scale.get("planned_holdout_source_rows", 0)) <= 0:
        failures.append("scale")
    execution = dict(plan.get("execution") or {})
    for key in ("docker_started", "network_contacted", "gpu_touched", "training_started", "rows_written", "split_relabelled"):
        if execution.get(key) is not False:
            failures.append(f"execution:{key}")
    promotion = dict(plan.get("promotion") or {})
    for key in PROMOTION:
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")
    forbidden = _find_forbidden_keys(plan)
    if forbidden:
        failures.append("forbidden_keys:" + ",".join(forbidden))
    return {
        "status": "passed" if not failures else "blocked",
        "failures": sorted(set(failures)),
        "route_count": len(routes),
        "planned_train_rows": len(train),
        "planned_holdout_rows": len(holdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg333", type=Path, default=PG333_PATH)
    parser.add_argument("--pg377", type=Path, default=PG377_PATH)
    parser.add_argument("--dynamic-registry", type=Path, default=DYNAMIC_REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg379_source_collection_matrix_plan_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = build_pg379_source_collection_plan(pg333_path=args.pg333, pg377_path=args.pg377, dynamic_registry_path=args.dynamic_registry)
        validation = validate_pg379_source_collection_plan(report)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"plan_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": report, "validation": validation} if args.json else {"status": report["status"], "validation": validation, "plan_sha256": report["plan_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if validation["status"] == "passed" else 2


# Short aliases make the planning API convenient for tests and research ops.
build_plan = build_pg379_source_collection_plan
validate_plan = validate_pg379_source_collection_plan
build_pg379_plan = build_pg379_source_collection_plan
validate_pg379_plan = validate_pg379_source_collection_plan
plan = build_pg379_source_collection_plan


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHODS",
    "PG333_PATH",
    "PG377_PATH",
    "DYNAMIC_REGISTRY_PATH",
    "DEFAULT_DYNAMIC_REGISTRY",
    "DEFAULT_PG333",
    "DEFAULT_PG377",
    "ROLES",
    "ROUTE_SHAPES",
    "SCHEMA_VERSION",
    "SEEDS",
    "SLOTS",
    "build_pg379_source_collection_plan",
    "build_pg379_plan",
    "build_plan",
    "plan",
    "validate_pg379_source_collection_plan",
    "validate_pg379_plan",
    "validate_plan",
]
