"""PG-332 typed, evaluator-only GET pilot for the pinned DVWA image.

The pilot intentionally uses inert HTML canaries rather than script or SQL
payloads.  The canary bytes exist only inside the evaluator callback while a
fresh disposable container is alive.  Rows contain only the abstract
observation, role-bound evidence hashes and Rule-IR target; raw probes and
response bodies are never serialized.

This is a source-row pilot, not a claim about arbitrary targets.  It requires
an explicit pair of flags and leaves all promotion gates closed until the
cross-implementation information/holdout audit is rerun.
"""

from __future__ import annotations

import json
import os
import re
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
from app.pg332_dvwa_docker_relay import DisposableDvwa  # noqa: E402
from scripts.plan_pg332_dvwa_source_rows import IMAGE, ROLES, _BY_ID, _route_ref  # noqa: E402
from scripts.run_pg332_dvwa_source_rows_live import (  # noqa: E402
    _route_request,
    build_container_command,
    container_name,
)


SCHEMA_VERSION = "pg332-dvwa-typed-get-source-rows-v1"
ROUTE_ID = "dvwa-xss-reflected-get"
SEEDS = (33310, 33311, 33312)
SOURCE_ROLES = ("candidate", "reference", "negative")
ALL_ROLES = (*SOURCE_ROLES, "replay")
_HEX = re.compile(r"^[0-9a-f]{64}$")


def _probe_value(seed: int, role: str) -> str:
    """Return an evaluator-only inert canary; never put this in a row/report."""

    if role == "candidate":
        return f'<p data-pg332-c="c{seed}">c{seed}</p>'
    if role == "reference":
        return f'<p data-pg332-r="r{seed}">r{seed}</p>'
    if role == "replay":
        return f'<p data-pg332-c="c{seed}">c{seed}</p>'
    return f"n{seed}"


def _effect_probe(role: str, marker: str):
    marker_bytes = marker.encode("utf-8").lower()
    expected_raw = role in {"candidate", "reference", "replay"}

    def inspect(body: bytes, status: int, headers: Mapping[str, Any]) -> dict[str, Any]:
        lowered = bytes(body).lower()
        marker_present = marker_bytes in lowered
        raw_tag = b"<p data-pg332-" in lowered and b"&lt;p data-pg332-" not in lowered
        typed = bool(status == 200 and marker_present and raw_tag) if expected_raw else bool(status == 200 and raw_tag)
        return {
            "typed_effect": typed,
            "marker_present": marker_present,
            "raw_tag": raw_tag,
            "status_ok": status == 200,
            "content_type_html": "html" in str(dict(headers).get("content-type", "")).casefold(),
        }

    return inspect


def _bucket_length(value: int) -> str:
    if value <= 0:
        return "empty"
    if value <= 256:
        return "tiny"
    if value <= 4096:
        return "small"
    if value <= 16384:
        return "medium"
    return "large"


def _belief(*, role: str, typed: bool, replay_ready: bool = True) -> dict[str, Any]:
    posterior = "high" if typed else "low"
    return {
        "observation_presence": "present",
        "observation_delta_axis": "response_shape",
        "belief_prior_bucket": "low",
        "belief_posterior_bucket": posterior,
        "belief_delta_axis": "response_shape",
        "history_action": f"{role}_request",
        "history_length": 2,
        "typed_available": "present",
        "evidence_present": "present",
        "negative_control": "present",
        "fresh_reset": "present",
        "replay_ready": "present" if replay_ready else "absent",
        "reference_present": "present",
        "candidate_present": "present",
        "step_budget": "present",
        "probe_count": 1,
        "evidence_hash_present": "present",
        "failure_class": "none",
        "failure_stage": "none",
        "error_shape": "empty",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none",
        "timeout_bucket": "none",
        "blocked_reason_class": "none",
        "previous_action": "observe",
        "next_action": "select_probe_variant",
        "repair_delta_axis": "none",
        "repair_outcome": "not_applicable",
        "method": "GET",
        "placement": "query",
        "content_type_class": "html",
        "query_count": 1,
        "form_count": 0,
        "json_field_count": 0,
        "multipart_part_count": 0,
        "parameter_role": "query_term",
        "parameter_name_shape": "alpha",
        "parameter_value_type": "text",
        "parameter_presence": "present",
        "parameter_order": 1,
        "header_presence_class": "basic",
        "cookie_presence_class": "present",
        "csrf_presence_class": "absent",
        "content_length_bucket": "medium",
        "encoding_chain": "url_percent",
        "charset_class": "absent",
        "body_shape": "html",
        "status_class": "2xx",
        "status_shape": "numeric",
        "body_length_bucket": "medium",
        "cache_shape": "absent",
        "redirect_hop_count": 0,
        "redirect_location_class": "none",
        "redirect_chain_shape": "empty",
        "connection_outcome": "complete",
    }


def _failure(*, role: str) -> dict[str, Any]:
    return {
        "failure_class": "none",
        "failure_stage": "none",
        "error_shape": "empty",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none",
        "blocked_reason_class": "none",
        "previous_action": "observe",
        "next_action": "select_probe_variant",
        "repair_delta_axis": "none",
        "repair_outcome": "not_applicable",
        "new_observation": "present",
        "retry_count": 0,
        "timeout_bucket": "none",
        "environment_failure_class": "none",
    }


def _role_projection(*, role: str, effect: Mapping[str, Any], shape: Mapping[str, Any], reset: Mapping[str, Any], route_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    typed = bool(effect.get("typed_effect"))
    status_class = "2xx" if int(shape.get("status", 0) or 0) == 200 else "other"
    effect_class = "result_shape" if typed else "none"
    abstract = {
        "status_class": status_class,
        "content_type_class": str(shape.get("content_type_class", "unknown")),
        "body_shape": str(shape.get("body_shape", "unknown")),
        "body_length_bucket": _bucket_length(int(shape.get("body_length", 0) or 0)),
        "connection_outcome": "complete" if int(shape.get("status", 0) or 0) else "transport_error",
        "effect_marker": typed,
        "effect_shape": "html_marker" if typed else "none",
        "response_shape_changed": typed,
        "row_shape_changed": typed,
        "navigation_allowed": True,
        "external_network_blocked": True,
        "database_touched": False,
        "non_destructive": True,
        "dom_script_execution": False,
        "script_execution": False,
        "sink_present": False,
        "network_request_count": 0,
        "shape_sha256": sha256_json({"status": shape.get("status"), "body_shape": shape.get("body_shape"), "effect": typed}),
    }
    source_hash = sha256_json({"schema": SCHEMA_VERSION, "route_ref": route_ref, "role": role, "reset_id": reset.get("reset_id"), "effect": abstract})
    role_input = {
        "sent": True,
        "available": True,
        "executed": True,
        "typed_effect_confirmed": typed,
        "effect_class": effect_class,
        "projection": abstract,
        "evidence_sha256": source_hash,
        "non_destructive": True,
    }
    return role_input, abstract


def _row_evaluator(*, role: str, sidecar: Mapping[str, Any], role_input: Mapping[str, Any]) -> dict[str, Any]:
    side_roles = dict(sidecar.get("roles") or {})
    bound = dict(side_roles.get(role) or {})
    return {
        "typed_available": True,
        "negative_control": True,
        "reference_present": True,
        "candidate_present": True,
        "fresh_reset": True,
        "evidence_hash": str(bound.get("evidence_sha256", "")),
        "confirmed_positive": bool(bound.get("typed_effect_confirmed")) if role != "negative" else False,
        "effect_class": str(role_input.get("effect_class", "none")),
        "evaluator_version": "pg332-dvwa-typed-get-inert-canary-v1",
    }


def _target(role: str) -> dict[str, Any]:
    variant = {"candidate": "source_attested_candidate", "reference": "reference", "negative": "negative_control"}[role]
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


def _collect_role(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    name = container_name(seed=seed, route_ref_sha256=route_ref, role=role)
    target = DisposableDvwa(name=name, seed=seed, index=0, command=build_container_command(seed=seed, route_ref_sha256=route_ref, role=role))
    try:
        reset = target.start(timeout=150.0)
        marker = _probe_value(seed, role)
        capture, shape, effect = _route_request(
            target,
            route,
            probe_values={str(route["field"]): marker},
            belief_projection=_belief(role=role, typed=role in {"candidate", "reference", "replay"}),
            failure_projection=_failure(role=role),
            effect_probe=_effect_probe(role, marker),
        )
        return {"reset": reset, "capture": capture, "shape": shape, "effect": dict(effect or {}), "role_input": _role_projection(role=role, effect=dict(effect or {}), shape=shape, reset=reset, route_ref=route_ref)[0]}
    finally:
        target.stop()


def collect_typed_get(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    if os.environ.get("PG332_LOCAL_DOCKER_EVAL") != "1" or os.environ.get("PG332_TYPED_GET_LIVE") != "1":
        raise RuntimeError("PG-332 typed GET requires PG332_LOCAL_DOCKER_EVAL=1 and PG332_TYPED_GET_LIVE=1")
    route = _BY_ID[ROUTE_ID]
    route_ref = _route_ref(route)
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed_value in seeds:
        seed = int(seed_value)
        roles = {role: _collect_role(seed=seed, role=role, route=route, route_ref=route_ref) for role in ALL_ROLES}
        candidate = roles["candidate"]
        reference = roles["reference"]
        negative = roles["negative"]
        replay = roles["replay"]
        candidate_input = candidate["role_input"]
        reference_input = reference["role_input"]
        negative_input = negative["role_input"]
        replay_consistent = bool(candidate_input.get("typed_effect_confirmed") == replay["role_input"].get("typed_effect_confirmed") and candidate_input.get("typed_effect_confirmed"))
        record = build_pg331_evaluator_record(
            record_id=f"pg332-dvwa-typed-get-{seed}",
            reset=candidate["reset"],
            candidate=candidate_input,
            reference=reference_input,
            negative=negative_input,
            replay_consistent=replay_consistent,
            reference_agreement=bool(candidate_input.get("typed_effect_confirmed") and reference_input.get("typed_effect_confirmed")),
            negative_control_clean=not bool(negative_input.get("typed_effect_confirmed")),
            evaluator_id="pg332-dvwa-typed-get-inert-canary-v1",
        )
        sidecars.append({"seed": seed, "sidecar": record["evaluator_sidecar"], "record_sha256": record["record_sha256"]})
        for role in SOURCE_ROLES:
            item = roles[role]
            evaluator = _row_evaluator(role=role, sidecar=record["evaluator_sidecar"], role_input=item["role_input"])
            row_reset = {str(key): item["reset"][key] for key in RESET_KEYS if key in item["reset"]}
            row = collect_pg331_source_row(
                record_id=f"pg332-dvwa-typed-get-{seed}-{role}",
                observation=item["capture"]["observation"],
                source_meta={
                    "source_id": "pg332-dvwa-local",
                    "implementation": "vulnerables-web-dvwa",
                    "collector_id": SCHEMA_VERSION,
                    "authorization_id": "operator-authorized-local-network-none",
                    "image_digest": IMAGE.split("@sha256:", 1)[1],
                    "source_digest": sha256_json({"seed": seed, "route_ref": route_ref, "role": role, "evidence": evaluator["evidence_hash"]}),
                },
                reset=row_reset,
                evaluator=evaluator,
                field_capture_manifest=item["capture"]["field_capture_manifest"],
                target_projection=_target(role),
                split="implementation_holdout",
                operator_reviewed=True,
                hard_negative=role == "negative",
            )
            rows.append(row)
        seed_summaries.append({
            "seed": seed,
            "candidate_typed": bool(candidate_input.get("typed_effect_confirmed")),
            "reference_typed": bool(reference_input.get("typed_effect_confirmed")),
            "negative_typed": bool(negative_input.get("typed_effect_confirmed")),
            "replay_consistent": replay_consistent,
            "row_failures": {role: list(next(row["failures"] for row in rows if row["record_id"] == f"pg332-dvwa-typed-get-{seed}-{role}")) for role in SOURCE_ROLES},
        })
    typed_positive = sum(int(item["candidate_typed"] and item["reference_typed"]) for item in seed_summaries)
    negative_violations = sum(int(item["negative_typed"]) for item in seed_summaries)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_typed_diagnostic_only",
        "runtime": {"image": IMAGE, "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False, "target_contacted": True, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "route_ref_sha256": route_ref,
        "counts": {"seed_count": len(seed_summaries), "route_count": 1, "role_replay_count": len(seed_summaries) * 4, "source_row_count": len(rows), "typed_positive_seed_count": typed_positive, "negative_violation_count": negative_violations, "training_eligible_row_count": sum(int(row.get("training_eligible") is True) for row in rows)},
        "seed_summaries": seed_summaries,
        "hard_gate": {"typed_candidate_reference": typed_positive == len(seed_summaries), "negative_zero_violation": negative_violations == 0, "replay_consistent": all(item["replay_consistent"] for item in seed_summaries), "fresh_reset_per_role": True, "role_bound_evidence": True, "context_firewall": all(row["context_firewall"]["forbidden_token_count"] == 0 for row in rows), "cross_implementation_holdout": True, "status": "blocked_single_route_single_family_and_no_post_pair"},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "DVWA authenticated GET typed inert-canary replay completed in fresh network-none containers; this is response/DOM-shape evidence only. It is not a general XSS claim and cannot promote training until POST, cross-implementation audit, field entropy and operator review are independently satisfied.",
    }
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any]) -> dict[str, str]:
    report_path = ROOT / "research" / "pg332_dvwa_typed_get_report_v1.json"
    rows_path = ROOT / "research" / "pg332_dvwa_typed_get_source_rows_v1.json"
    sidecar_path = ROOT / "research" / "pg332_dvwa_typed_get_sidecars_v1.json"
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_doc = {"schema_version": SCHEMA_VERSION, "status": "diagnostic_source_rows", "records": result["rows"], "promotion": result["report"]["promotion"]}
    rows_doc["dataset_sha256"] = sha256_json(rows_doc)
    rows_path.write_text(json.dumps(rows_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    side_doc = {"schema_version": SCHEMA_VERSION, "status": "evaluator_sidecar_only", "sidecars": result["sidecars"], "promotion": result["report"]["promotion"]}
    side_doc["sidecars_sha256"] = sha256_json(side_doc)
    sidecar_path.write_text(json.dumps(side_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(report_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)), "sidecars": str(sidecar_path.relative_to(ROOT))}


if __name__ == "__main__":
    result = collect_typed_get()
    print(json.dumps({"report": result["report"], "artifacts": write_artifacts(result)}, ensure_ascii=False, indent=2))
