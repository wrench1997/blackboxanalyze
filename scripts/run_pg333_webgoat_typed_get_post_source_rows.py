"""PG-333 WebGoat GET/POST method-shape source-row collector.

This is a third-implementation structural canary.  It uses only invalid local
login data and a typed HTTP-shape oracle (GET page vs POST redirect); it is not
an XSS, SQL, authentication-bypass, or payload-generation experiment.  Each
seed/role/replay gets a disposable network-none container.  Raw request/body
bytes are held in memory only while the evaluator and structural adapter run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_pg331_evaluator_record, sha256_json  # noqa: E402
from app.pg331_source_row import RESET_KEYS, collect_pg331_source_row  # noqa: E402
from app.pg331_vulnerableapp_adapter import capture_vulnerableapp_projection  # noqa: E402
from app.pg333_webgoat_docker_relay import (  # noqa: E402
    IMAGE,
    IMAGE_DIGEST,
    ROUTE_PATH,
    DisposableWebGoat,
    build_container_command,
    container_name,
)


SCHEMA_VERSION = "pg333-webgoat-typed-method-shape-source-rows-v1"
SEEDS = (33340, 33341, 33342)
SOURCE_ROLES = ("candidate", "reference", "negative")
ALL_ROLES = (*SOURCE_ROLES, "replay")
ROUTES: tuple[dict[str, Any], ...] = (
    {"route_id": "webgoat-login-get", "expected_method": "GET", "surface_id": "login_page_shape"},
    {"route_id": "webgoat-login-post", "expected_method": "POST", "surface_id": "login_redirect_shape"},
)


def _route_ref(route: Mapping[str, Any]) -> str:
    return sha256_json({"schema": SCHEMA_VERSION, "route_id": str(route["route_id"]), "method": str(route["expected_method"]), "path_digest": sha256_json(ROUTE_PATH)})


def _action_method(expected_method: str, role: str) -> str:
    if role == "negative":
        return "POST" if str(expected_method).upper() == "GET" else "GET"
    return str(expected_method).upper()


def _form_body(method: str) -> bytes:
    # This is an invalid, non-secret login value, not a vulnerability probe.
    return b"username=probe.invalid&password=probe.invalid" if method == "POST" else b""


def _typed_effect(*, expected_method: str, action_method: str, action: Mapping[str, Any], body: bytes) -> bool:
    if action_method != expected_method:
        return False
    status = int(action.get("status", 0) or 0)
    if expected_method == "GET":
        return bool(status == 200 and str(action.get("content_type_class", "")) == "text/html" and len(body) > 100)
    return bool(status == 302 and str(action.get("location_class", "")) == "loopback")


def _failure(*, expected_method: str, action_method: str, typed: bool) -> dict[str, Any]:
    if typed or action_method != expected_method:
        return {
            "failure_class": "none",
            "failure_stage": "none",
            "error_shape": "empty",
            "parse_error_class": "none",
            "encoding_error_class": "none",
            "redirect_error_class": "none",
            "blocked_reason_class": "none",
            "previous_action": "baseline_observe",
            "next_action": "select_probe_variant",
            "repair_delta_axis": "none",
            "repair_outcome": "not_applicable",
            "new_observation": "present",
            "retry_count": 0,
            "timeout_bucket": "none",
            "environment_failure_class": "none",
        }
    return {
        "failure_class": "response_shape_mismatch",
        "failure_stage": "evaluator",
        "error_shape": "shape_difference",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "unexpected_method_surface",
        "blocked_reason_class": "none",
        "previous_action": "baseline_observe",
        "next_action": "ask_typed",
        "repair_delta_axis": "response_shape",
        "repair_outcome": "abstain_until_typed",
        "new_observation": "present",
        "retry_count": 0,
        "timeout_bucket": "none",
        "environment_failure_class": "none",
    }


def _belief(*, expected_method: str, action_method: str, typed: bool, role: str, csrf_class: str, cookie_class: str) -> dict[str, Any]:
    return {
        "observation_presence": "present",
        "observation_delta_axis": "response_shape",
        "belief_prior_bucket": "low",
        "belief_posterior_bucket": "high" if typed else "low",
        "belief_delta_axis": "response_shape",
        "history_action": "baseline_observe_then_post" if action_method == "POST" else "baseline_observe",
        "history_length": 2,
        "typed_available": "present",
        "evidence_present": "present",
        "negative_control": "present",
        "fresh_reset": "present",
        "replay_ready": "present",
        "reference_present": "present",
        "candidate_present": "present",
        "step_budget": "present",
        "probe_count": 1,
        "evidence_hash_present": "present",
        "failure_class": "none" if typed or role == "negative" else "response_shape_mismatch",
        "failure_stage": "none" if typed or role == "negative" else "evaluator",
        "error_shape": "empty" if typed or role == "negative" else "shape_difference",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none" if typed or role == "negative" else "unexpected_method_surface",
        "timeout_bucket": "none",
        "blocked_reason_class": "none",
        "previous_action": "baseline_observe",
        "next_action": "select_probe_variant" if typed or role == "negative" else "ask_typed",
        "repair_delta_axis": "none" if typed or role == "negative" else "response_shape",
        "repair_outcome": "not_applicable" if typed or role == "negative" else "abstain_until_typed",
        "method": action_method,
        "placement": "form" if action_method == "POST" else "path",
        "content_type_class": "form_urlencoded" if action_method == "POST" else "html",
        "query_count": 0,
        "form_count": 2 if action_method == "POST" else 0,
        "json_field_count": 0,
        "multipart_part_count": 0,
        "parameter_role": "credential_pair" if action_method == "POST" else "none",
        "parameter_name_shape": "abstract" if action_method == "POST" else "none",
        "parameter_value_type": "text" if action_method == "POST" else "none",
        "parameter_presence": "present" if action_method == "POST" else "absent",
        "parameter_order": 1 if action_method == "POST" else 0,
        "header_presence_class": "basic",
        "cookie_presence_class": cookie_class,
        "csrf_presence_class": csrf_class,
        "content_length_bucket": "tiny" if action_method == "POST" else "empty",
        "encoding_chain": "form_urlencoded" if action_method == "POST" else "none",
        "charset_class": "utf8" if action_method == "GET" else "absent",
        "body_shape": "form" if action_method == "POST" else "empty",
        "status_class": "2xx" if expected_method == "GET" else "3xx",
        "status_shape": "numeric",
        "body_length_bucket": "medium" if expected_method == "GET" else "empty",
        "cache_shape": "absent",
        "redirect_hop_count": 0 if expected_method == "GET" else 1,
        "redirect_location_class": "none" if expected_method == "GET" else "loopback",
        "redirect_chain_shape": "empty" if expected_method == "GET" else "single_hop",
        "connection_outcome": "complete",
    }


def _role_target(*, expected_method: str, action_method: str, role: str, complete: bool) -> dict[str, Any]:
    variant = {
        "candidate": "source_attested_candidate",
        "reference": "reference",
        "negative": "negative_control",
    }[role]
    if not complete:
        return {
            "question": "ask_typed",
            "next_action": "ask_typed",
            "repair_action": "observe",
            "transport_ref": "request_method",
            "field_role_ref": "parameter_role",
            "encoding_ref": "encoding_chain",
            "probe_variant_ref": variant,
            "safe_to_send": False,
        }
    return {
        "question": "none",
        "next_action": "select_probe_variant",
        "repair_action": "none",
        "transport_ref": "request_method",
        "field_role_ref": "parameter_role",
        "encoding_ref": "encoding_chain",
        "probe_variant_ref": variant,
        "safe_to_send": False,
    }


def _abstract_projection(*, action: Mapping[str, Any], body: bytes, typed: bool, expected_method: str) -> dict[str, Any]:
    status = int(action.get("status", 0) or 0)
    body_shape = "html" if body.lstrip().lower().startswith((b"<!doctype", b"<html", b"<head", b"<body")) else "empty" if not body else "text"
    effect_class = "redirect_hop" if expected_method == "POST" else "result_shape"
    content_type = str(action.get("content_type_class", "unknown")).casefold()
    content_type = "html" if "html" in content_type else "json" if "json" in content_type else "text" if content_type.startswith("text") else "unknown"
    return {
        "status_class": str(action.get("status_class", "transport_error")),
        "content_type_class": content_type,
        "body_shape": body_shape,
        "body_length_bucket": "empty" if len(body) == 0 else "medium" if len(body) > 4096 else "small",
        "redirect_hop_count": 1 if status == 302 else 0,
        "redirect_location_class": str(action.get("location_class", "none")),
        "redirect_chain_shape": "single_hop" if status == 302 else "empty",
        "connection_outcome": "complete" if status else "transport_error",
        "shape_sha256": sha256_json({"status": status, "method": str(action.get("method")), "content": action.get("content_type_class"), "body_length": len(body), "typed": typed}),
        "effect_marker": typed,
        "effect_shape": "redirect" if status == 302 else "html_page" if status == 200 else "none",
        "response_shape_changed": typed,
        "row_shape_changed": typed,
        "navigation_allowed": True,
        "database_touched": False,
        "external_network_blocked": True,
        "non_destructive": True,
        "dom_script_execution": False,
        "script_execution": False,
        "sink_present": False,
        "network_request_count": 0,
    }, effect_class


def _capture_role(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    expected = str(route["expected_method"]).upper()
    action_method = _action_method(expected, role)
    name = container_name(seed=seed, route_ref_sha256=route_ref, role=role)
    target = DisposableWebGoat(name=name, seed=seed, role=role, command=build_container_command(name=name, seed=seed, role=role))
    try:
        reset = target.start()
        baseline = target.request(method="GET")
        action = baseline if action_method == "GET" else target.request(method="POST", form_body=_form_body("POST"))
        action_body = bytes(action.get("body") or b"")
        typed = _typed_effect(expected_method=expected, action_method=action_method, action=action, body=action_body)
        # The POST row is a two-step abstract trace: baseline page structure is
        # retained in memory while the POST redirect is the response shape.
        html_body = bytes(baseline.get("body") or b"") if action_method == "POST" else action_body
        headers = dict(action.get("headers") or {})
        if action.get("location_class") != "none":
            headers["Location"] = "loopback"
        if action.get("content_type_class") not in {"unknown", ""}:
            headers["Content-Type"] = str(action.get("content_type_class"))
        csrf_class = "present" if any(fragment in html_body.lower() for fragment in (b"csrf", b"xsrf")) else "absent"
        cookie_class = "present" if "set-cookie" in dict(action.get("headers") or {}) else "absent"
        request_projection = {
            "method": action_method,
            "parameters": ([{"role": "identity", "value_type": "text", "presence": "present"}, {"role": "secret", "value_type": "text", "presence": "present"}] if action_method == "POST" else []),
            "csrf_presence_class": csrf_class,
            "cookie_presence_class": cookie_class,
            "content_length": len(_form_body("POST")) if action_method == "POST" else 0,
        }
        response_projection = {
            "status": int(action.get("status", 0) or 0),
            "body_length": len(action_body),
            "body_shape": "empty" if not action_body else "html",
            "connection_outcome": "complete" if action.get("status") else "transport_error",
            "charset_class": "utf8" if action_method == "GET" else "absent",
            "cache_shape": "absent",
            "csrf_presence_class": csrf_class,
            "failure_class": "none" if typed or role == "negative" else "response_shape_mismatch",
            "failure_stage": "none" if typed or role == "negative" else "evaluator",
            "error_shape": "empty" if typed or role == "negative" else "shape_difference",
        }
        capture = capture_vulnerableapp_projection(
            html=html_body.decode("utf-8", errors="replace"),
            headers=headers,
            request_projection=request_projection,
            response_projection=response_projection,
            post_supported=True,
            failure_projection=_failure(expected_method=expected, action_method=action_method, typed=typed),
            belief_projection=_belief(expected_method=expected, action_method=action_method, typed=typed, role=role, csrf_class=csrf_class, cookie_class=cookie_class),
        )
        abstract, effect_class = _abstract_projection(action=action, body=action_body, typed=typed, expected_method=expected)
        source_evidence = sha256_json({"seed": seed, "role": role, "route_ref": route_ref, "reset_id": reset.get("reset_id"), "shape": abstract.get("shape_sha256")})
        role_input = {
            "sent": True,
            "available": True,
            "executed": True,
            "typed_effect_confirmed": typed,
            "effect_class": effect_class,
            "projection": abstract,
            "evidence_sha256": source_evidence,
            "non_destructive": True,
        }
        return {"reset": reset, "capture": capture, "action": action, "role_input": role_input, "typed": typed, "action_method": action_method, "source_evidence": source_evidence}
    finally:
        target.stop()


def _row_evaluator(*, role: str, sidecar: Mapping[str, Any], role_input: Mapping[str, Any]) -> dict[str, Any]:
    bound = dict(dict(sidecar.get("roles") or {}).get(role) or {})
    return {
        "typed_available": True,
        "negative_control": True,
        "reference_present": True,
        "candidate_present": True,
        "fresh_reset": True,
        "evidence_hash": str(bound.get("evidence_sha256", "")),
        "confirmed_positive": bool(bound.get("typed_effect_confirmed")) if role != "negative" else False,
        "effect_class": str(role_input.get("effect_class", "none")),
        "evaluator_version": "pg333-webgoat-method-shape-v1",
    }


def collect_typed_method_shape(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    if os.environ.get("PG333_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-333 WebGoat requires PG333_LOCAL_DOCKER_EVAL=1")
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed_value in seeds:
        seed = int(seed_value)
        route_summaries: list[dict[str, Any]] = []
        for route in ROUTES:
            route_ref = _route_ref(route)
            roles = {role: _capture_role(seed=seed, role=role, route=route, route_ref=route_ref) for role in ALL_ROLES}
            candidate, reference, negative, replay = (roles[name] for name in ALL_ROLES)
            record = build_pg331_evaluator_record(
                record_id=f"pg333-webgoat-{seed}-{route['route_id']}",
                reset=candidate["reset"] | {"volume_mount_count": 0, "container_restart_used": False},
                candidate=candidate["role_input"],
                reference=reference["role_input"],
                negative=negative["role_input"],
                replay_consistent=bool(candidate["typed"] == replay["typed"] and candidate["typed"]),
                reference_agreement=bool(candidate["typed"] and reference["typed"]),
                negative_control_clean=not bool(negative["typed"]),
                evaluator_id="pg333-webgoat-method-shape-v1",
            )
            sidecars.append({"seed": seed, "route_id": str(route["route_id"]), "sidecar": record["evaluator_sidecar"], "record_sha256": record["record_sha256"]})
            route_row_failures: dict[str, list[str]] = {}
            for role in SOURCE_ROLES:
                item = roles[role]
                evaluator = _row_evaluator(role=role, sidecar=record["evaluator_sidecar"], role_input=item["role_input"])
                row_reset = {str(key): item["reset"][key] for key in RESET_KEYS if key in item["reset"]}
                row = collect_pg331_source_row(
                    record_id=f"pg333-webgoat-{seed}-{route['route_id']}-{role}",
                    observation=item["capture"]["observation"],
                    source_meta={
                        "source_id": "pg333-webgoat-local",
                        "implementation": "webgoat",
                        "family_id": "webgoat_login_surface",
                        "surface_id": str(route["surface_id"]),
                        "collector_id": SCHEMA_VERSION,
                        "authorization_id": "operator-authorized-local-network-none",
                        "image_digest": IMAGE_DIGEST,
                        "source_digest": sha256_json({"seed": seed, "route_ref": route_ref, "role": role, "evidence": evaluator["evidence_hash"]}),
                    },
                    reset=row_reset,
                    evaluator=evaluator,
                    field_capture_manifest=item["capture"]["field_capture_manifest"],
                    target_projection=_role_target(expected_method=str(route["expected_method"]), action_method=str(item["action_method"]), role=role, complete=not any(str(status) in {"not_observed", "unknown"} for fields in item["capture"]["field_capture_manifest"].values() if isinstance(fields, Mapping) for status in fields.values())),
                    split="implementation_holdout",
                    operator_reviewed=True,
                    hard_negative=role == "negative",
                )
                rows.append(row)
                route_row_failures[role] = list(row.get("failures") or [])
            route_summaries.append({"route_id": str(route["route_id"]), "expected_method": str(route["expected_method"]), "candidate_typed": bool(candidate["typed"]), "reference_typed": bool(reference["typed"]), "negative_typed": bool(negative["typed"]), "replay_consistent": bool(candidate["typed"] == replay["typed"] and candidate["typed"]), "row_failures": route_row_failures})
        seed_summaries.append({"seed": seed, "routes": route_summaries})
    typed_positive = sum(int(route["candidate_typed"] and route["reference_typed"]) for seed in seed_summaries for route in seed["routes"])
    negative_violations = sum(int(route["negative_typed"]) for seed in seed_summaries for route in seed["routes"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_typed_method_shape_diagnostic_only",
        "runtime": {"image": IMAGE, "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False, "target_contacted": True, "disposable_internal_db": True, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "counts": {"seed_count": len(seed_summaries), "route_count": len(ROUTES), "role_replay_count": len(seed_summaries) * len(ROUTES) * 4, "source_row_count": len(rows), "typed_positive_route_seed_count": typed_positive, "negative_violation_count": negative_violations, "training_eligible_row_count": sum(int(row.get("training_eligible") is True) for row in rows)},
        "methods": {"GET": sum(int(item["expected_method"] == "GET") for item in ROUTES), "POST": sum(int(item["expected_method"] == "POST") for item in ROUTES)},
        "seed_summaries": seed_summaries,
        "hard_gate": {"typed_candidate_reference": typed_positive == len(seed_summaries) * len(ROUTES), "negative_zero_violation": negative_violations == 0, "replay_consistent": all(route["replay_consistent"] for seed in seed_summaries for route in seed["routes"]), "fresh_reset_per_role": True, "role_bound_evidence": True, "context_firewall": all(int(dict(row.get("context_firewall") or {}).get("forbidden_token_count", 1)) == 0 for row in rows), "network_none": True, "no_bind_or_volume": True, "status": "blocked_method_shape_canary_not_vulnerability"},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "WebGoat third-implementation GET/POST method-shape canary completed in disposable network-none containers. Typed effects are HTTP page/redirect shape only; this is not an XSS, SQL, authentication-bypass, or payload result. Keep as diagnostic until merged information/family/ASK/failure audits pass.",
    }
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any]) -> dict[str, str]:
    report_path = ROOT / "research" / "pg333_webgoat_typed_method_shape_report_v1.json"
    rows_path = ROOT / "research" / "pg333_webgoat_typed_method_shape_source_rows_v1.json"
    sidecar_path = ROOT / "research" / "pg333_webgoat_typed_method_shape_sidecars_v1.json"
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_doc = {"schema_version": SCHEMA_VERSION, "status": "diagnostic_source_rows", "records": result["rows"], "promotion": result["report"]["promotion"]}
    rows_doc["dataset_sha256"] = sha256_json(rows_doc)
    rows_path.write_text(json.dumps(rows_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    side_doc = {"schema_version": SCHEMA_VERSION, "status": "evaluator_sidecar_only", "sidecars": result["sidecars"], "promotion": result["report"]["promotion"]}
    side_doc["sidecars_sha256"] = sha256_json(side_doc)
    sidecar_path.write_text(json.dumps(side_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(report_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)), "sidecars": str(sidecar_path.relative_to(ROOT))}


if __name__ == "__main__":
    result = collect_typed_method_shape()
    print(json.dumps({"report": result["report"], "artifacts": write_artifacts(result)}, ensure_ascii=False, indent=2))
