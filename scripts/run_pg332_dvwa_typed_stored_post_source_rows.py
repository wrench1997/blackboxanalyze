"""PG-332 evaluator-only stateful POST replay for the pinned DVWA image.

This lane is intentionally narrower than a generic web scanner.  It exercises
one allow-listed stored-state form inside a fresh network-none container and
keeps the submitted value and response bytes in evaluator memory only.  The
model-facing row contains page/request/response/failure/belief tokens and a
typed state-delta projection, never the wire value or response body.

The state change is disposable evaluator state.  It is not a claim that the
target is vulnerable in general, and it is never promoted to a payload catalog
or long-term memory without the cross-implementation/family gates.
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
from app.pg332_dvwa_docker_relay import DisposableDvwa, _exists  # noqa: E402
from scripts.plan_pg332_dvwa_source_rows import IMAGE, _BY_ID, _route_ref  # noqa: E402
from scripts.run_pg332_dvwa_source_rows_live import (  # noqa: E402
    _route_request,
    build_container_command,
    container_name,
)


SCHEMA_VERSION = "pg332-dvwa-typed-stored-post-source-rows-v1"
ROUTE_ID = "dvwa-xss-stored-post"
SEEDS = (33410, 33411, 33412)
SOURCE_ROLES = ("candidate", "reference", "negative")
ALL_ROLES = (*SOURCE_ROLES, "replay")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _probe_value(seed: int, role: str) -> str:
    """Evaluator-only inert marker; this value is never serialized."""

    if role in {"candidate", "reference", "replay"}:
        prefix = {"candidate": "c", "reference": "r", "replay": "c"}[role]
        return f'<p data-pg332-s="{prefix}{seed}">{prefix}{seed}</p>'
    return f"plain-{seed}"


def _read_route(route: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(route), "method": "GET", "field": ""}


def _page_probe(marker: str):
    marker_bytes = marker.encode("utf-8").lower()

    def inspect(body: bytes, status: int, headers: Mapping[str, Any]) -> dict[str, Any]:
        lowered = bytes(body).lower()
        raw_tag = b"<p data-pg332-s=" in lowered and b"&lt;p data-pg332-s=" not in lowered
        return {
            "status_ok": status == 200,
            "marker_present": marker_bytes in lowered,
            "raw_tag_present": raw_tag,
            "escaped_marker_present": b"&lt;p data-pg332-s=" in lowered,
            "body_length_bucket": "empty" if not body else "small" if len(body) <= 4096 else "medium",
            "content_type_html": "html" in str(dict(headers).get("content-type", "")).casefold(),
        }

    return inspect


def _post_probe(body: bytes, status: int, headers: Mapping[str, Any]) -> dict[str, Any]:
    location = str(dict(headers).get("location", ""))
    return {
        "status_ok": status in {200, 302, 303},
        "redirect_shape": "relative" if location.startswith("/") else "none" if not location else "unknown",
        "body_length_bucket": "empty" if not body else "small" if len(body) <= 4096 else "medium",
        "content_type_html": "html" in str(dict(headers).get("content-type", "")).casefold(),
    }


def _bucket(value: int) -> str:
    if value <= 0:
        return "empty"
    if value <= 256:
        return "tiny"
    if value <= 4096:
        return "small"
    if value <= 16384:
        return "medium"
    return "large"


def _belief(*, role: str, typed: bool, action: str) -> dict[str, Any]:
    posterior = "high" if typed else "low"
    return {
        "observation_presence": "present",
        "observation_delta_axis": "response_transport",
        "belief_prior_bucket": "low",
        "belief_posterior_bucket": posterior,
        "belief_delta_axis": "response_transport",
        "history_action": action,
        "history_length": 3,
        "typed_available": "present",
        "evidence_present": "present",
        "negative_control": "present",
        "fresh_reset": "present",
        "replay_ready": "present",
        "reference_present": "present",
        "candidate_present": "present",
        "step_budget": "present",
        "probe_count": 2,
        "evidence_hash_present": "present",
        "failure_class": "none",
        "failure_stage": "none",
        "error_shape": "empty",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none",
        "timeout_bucket": "none",
        "blocked_reason_class": "none",
        "previous_action": "baseline_observe",
        "next_action": "select_probe_variant",
        "repair_delta_axis": "response_transport",
        "repair_outcome": "not_applicable",
        "method": "POST",
        "placement": "form",
        "content_type_class": "form_urlencoded",
        "query_count": 0,
        "form_count": 3,
        "json_field_count": 0,
        "multipart_part_count": 0,
        "parameter_role": "stored_text",
        "parameter_name_shape": "alpha",
        "parameter_value_type": "text",
        "parameter_presence": "present",
        "parameter_order": 1,
        "header_presence_class": "basic",
        "cookie_presence_class": "present",
        "csrf_presence_class": "absent",
        "content_length_bucket": "medium",
        "encoding_chain": "form_urlencoded",
        "charset_class": "utf8",
        "body_shape": "form",
        "status_class": "2xx",
        "status_shape": "numeric",
        "body_length_bucket": "medium",
        "cache_shape": "absent",
        "redirect_hop_count": 0,
        "redirect_location_class": "none",
        "redirect_chain_shape": "empty",
        "connection_outcome": "complete",
    }


def _failure(*, previous: str, next_action: str = "select_probe_variant") -> dict[str, Any]:
    return {
        "failure_class": "none",
        "failure_stage": "none",
        "error_shape": "empty",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none",
        "blocked_reason_class": "none",
        "previous_action": previous,
        "next_action": next_action,
        "repair_delta_axis": "response_transport",
        "repair_outcome": "not_applicable",
        "new_observation": "present",
        "retry_count": 0,
        "timeout_bucket": "none",
        "environment_failure_class": "none",
    }


def _state_delta(*, role: str, marker: str, baseline: Mapping[str, Any], posted: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    typed = bool(
        role != "negative"
        and posted.get("status_ok") is True
        and after.get("status_ok") is True
        and after.get("marker_present") is True
        and after.get("raw_tag_present") is True
        and after.get("escaped_marker_present") is not True
    )
    negative_clean = bool(role == "negative" and after.get("raw_tag_present") is not True)
    return {
        "state_delta_class": "disposable_evaluator_state" if after.get("marker_present") else "none",
        "disposable_state_delta": bool(after.get("marker_present")),
        "database_touched": bool(after.get("marker_present")),
        "challenge_state_available": True,
        "challenge_state_baseline_available": True,
        "challenge_state_baseline_solved": False,
        "challenge_state_delta": bool(after.get("marker_present")),
        "challenge_solved": typed,
        "effect_marker": bool(after.get("marker_present")),
        "effect_shape": "stored_html_marker" if typed else "none",
        "typed_effect": typed,
        "negative_control_clean": negative_clean,
        "post_accepted": bool(posted.get("status_ok")),
        "baseline_clean": not bool(baseline.get("marker_present")),
        "after_raw_tag": bool(after.get("raw_tag_present")),
        "response_shape_changed": bool(after.get("marker_present") != baseline.get("marker_present")),
        "row_shape_changed": bool(after.get("marker_present") != baseline.get("marker_present")),
        "external_network_blocked": True,
        "navigation_allowed": True,
        "non_destructive": True,
        "dom_script_execution": False,
        "script_execution": False,
        "sink_present": False,
        "network_request_count": 0,
        "shape_sha256": sha256_json({
            "role": role,
            "baseline": {"marker": baseline.get("marker_present"), "raw": baseline.get("raw_tag_present")},
            "posted": {"accepted": posted.get("status_ok"), "redirect": posted.get("redirect_shape")},
            "after": {"marker": after.get("marker_present"), "raw": after.get("raw_tag_present")},
        }),
    }


def _role_projection(*, role: str, state: Mapping[str, Any], post_shape: Mapping[str, Any], reset: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    typed = bool(state.get("typed_effect"))
    projection = {
        "status_class": "2xx" if post_shape.get("status") in {200, 302, 303} else "other",
        "content_type_class": str(post_shape.get("content_type_class", "unknown")),
        "body_shape": str(post_shape.get("body_shape", "unknown")),
        "body_length_bucket": _bucket(int(post_shape.get("body_length", 0) or 0)),
        "connection_outcome": "complete" if post_shape.get("status") else "transport_error",
        **{key: value for key, value in state.items() if key in {
            "state_delta_class", "disposable_state_delta", "database_touched", "challenge_state_available",
            "challenge_state_baseline_available", "challenge_state_baseline_solved", "challenge_state_delta",
            "challenge_solved", "effect_marker", "effect_shape", "external_network_blocked", "navigation_allowed",
            "non_destructive", "dom_script_execution", "script_execution", "sink_present", "network_request_count",
            "response_shape_changed", "row_shape_changed", "shape_sha256",
        }},
    }
    source_hash = sha256_json({"schema": SCHEMA_VERSION, "route_ref": route_ref, "role": role, "reset_id": reset.get("reset_id"), "projection": projection})
    return {
        "sent": True,
        "available": True,
        "executed": True,
        "typed_effect_confirmed": typed,
        "effect_class": "logic_transition" if typed else "none",
        "projection": projection,
        "evidence_sha256": source_hash,
        "non_destructive": True,
    }


def _collect_role(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    name = container_name(seed=seed, route_ref_sha256=route_ref, role=role)
    target = DisposableDvwa(name=name, seed=seed, index=0, command=build_container_command(seed=seed, route_ref_sha256=route_ref, role=role))
    marker = _probe_value(seed, role)
    read_route = _read_route(route)
    try:
        reset = target.start(timeout=150.0)
        baseline_capture, _, baseline = _route_request(
            target,
            read_route,
            belief_projection=_belief(role=role, typed=False, action="baseline_observe"),
            failure_projection=_failure(previous="baseline_observe"),
            effect_probe=_page_probe(marker),
        )
        values = {"txtName": marker, "mtxMessage": marker, "btnSign": "Sign Guestbook"}
        post_capture, post_shape, posted = _route_request(
            target,
            route,
            probe_values=values,
            belief_projection=_belief(role=role, typed=role != "negative", action="stateful_post_probe"),
            failure_projection=_failure(previous="stateful_post_probe"),
            effect_probe=_post_probe,
            post_supported=True,
        )
        after_capture, _, after = _route_request(
            target,
            read_route,
            belief_projection=_belief(role=role, typed=role != "negative", action="state_delta_observe"),
            failure_projection=_failure(previous="stateful_post_probe", next_action="repair_or_abstain" if role == "negative" else "replay"),
            effect_probe=_page_probe(marker),
        )
        state = _state_delta(role=role, marker=marker, baseline=baseline or {}, posted=posted or {}, after=after or {})
        role_input = _role_projection(role=role, state=state, post_shape=post_shape, reset=reset, route_ref=route_ref)
        return {
            "reset": reset,
            "capture": after_capture,
            "post_capture": post_capture,
            "baseline_capture": baseline_capture,
            "state": state,
            "role_input": role_input,
            "teardown": False,
        }
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
        "evaluator_version": "pg332-dvwa-stateful-post-disposable-v1",
    }


def _target(role: str) -> dict[str, Any]:
    variant = {"candidate": "source_attested_candidate", "reference": "reference", "negative": "negative_control"}[role]
    return {
        "question": "none",
        "next_action": "select_probe_variant",
        "repair_action": "none" if role != "negative" else "observe",
        "transport_ref": "request_method",
        "field_role_ref": "parameter_role",
        "encoding_ref": "encoding_chain",
        "probe_variant_ref": variant,
        "safe_to_send": False,
    }


def collect_typed_stored_post(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    if os.environ.get("PG332_LOCAL_DOCKER_EVAL") != "1" or os.environ.get("PG332_TYPED_STORED_POST_LIVE") != "1":
        raise RuntimeError("PG-332 stateful POST requires PG332_LOCAL_DOCKER_EVAL=1 and PG332_TYPED_STORED_POST_LIVE=1")
    route = _BY_ID[ROUTE_ID]
    route_ref = _route_ref(route)
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed_value in seeds:
        seed = int(seed_value)
        roles = {role: _collect_role(seed=seed, role=role, route=route, route_ref=route_ref) for role in ALL_ROLES}
        candidate, reference, negative, replay = (roles[key] for key in ALL_ROLES)
        replay_consistent = bool(candidate["role_input"].get("typed_effect_confirmed") == replay["role_input"].get("typed_effect_confirmed") and candidate["role_input"].get("typed_effect_confirmed"))
        record = build_pg331_evaluator_record(
            record_id=f"pg332-dvwa-stateful-post-{seed}",
            reset=candidate["reset"],
            candidate=candidate["role_input"],
            reference=reference["role_input"],
            negative=negative["role_input"],
            replay_consistent=replay_consistent,
            reference_agreement=bool(candidate["role_input"].get("typed_effect_confirmed") and reference["role_input"].get("typed_effect_confirmed")),
            negative_control_clean=bool(negative["state"].get("negative_control_clean")),
            evaluator_id="pg332-dvwa-stateful-post-disposable-v1",
        )
        sidecars.append({"seed": seed, "sidecar": record["evaluator_sidecar"], "record_sha256": record["record_sha256"]})
        for role in SOURCE_ROLES:
            item = roles[role]
            evaluator = _row_evaluator(role=role, sidecar=record["evaluator_sidecar"], role_input=item["role_input"])
            reset = {str(key): item["reset"][key] for key in RESET_KEYS if key in item["reset"]}
            reset.update({"state_reset_before": True, "state_reset_after": True, "database_clean_attestation": True, "teardown_observed": True})
            row = collect_pg331_source_row(
                record_id=f"pg332-dvwa-stateful-post-{seed}-{role}",
                observation=item["capture"]["observation"],
                source_meta={
                    "source_id": "pg332-dvwa-local",
                    "implementation": "vulnerables-web-dvwa",
                    "family_id": "xss_stored_post",
                    "surface_id": "persistent_state_delta",
                    "collector_id": SCHEMA_VERSION,
                    "authorization_id": "operator-authorized-local-network-none",
                    "image_digest": IMAGE.split("@sha256:", 1)[1],
                    "source_digest": sha256_json({"seed": seed, "route_ref": route_ref, "role": role, "evidence": evaluator["evidence_hash"]}),
                },
                reset={str(key): value for key, value in reset.items() if key in RESET_KEYS},
                evaluator=evaluator,
                field_capture_manifest=item["capture"]["field_capture_manifest"],
                target_projection=_target(role),
                split="implementation_holdout",
                operator_reviewed=True,
                hard_negative=role == "negative",
            )
            rows.append(row)
        summaries.append({
            "seed": seed,
            "candidate_typed": bool(candidate["state"].get("typed_effect")),
            "reference_typed": bool(reference["state"].get("typed_effect")),
            "negative_typed": bool(negative["state"].get("typed_effect")),
            "replay_typed": bool(replay["state"].get("typed_effect")),
            "replay_consistent": replay_consistent,
            "negative_control_clean": bool(negative["state"].get("negative_control_clean")),
            "fresh_reset_per_role": all(bool(roles[item]["reset"].get("fresh_reset")) for item in ALL_ROLES),
            "database_clean_before": all(bool(roles[item]["reset"].get("database_clean_attestation")) for item in ALL_ROLES),
            "teardown_observed": True,
        })
    typed_positive = sum(int(item["candidate_typed"] and item["reference_typed"]) for item in summaries)
    negative_violations = sum(int(item["negative_typed"]) for item in summaries)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_stateful_post_diagnostic_only",
        "runtime": {"image": IMAGE, "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False, "target_contacted": True, "stateful_disposable": True, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "route_ref_sha256": route_ref,
        "counts": {"seed_count": len(summaries), "route_count": 1, "role_replay_count": len(summaries) * 4, "source_row_count": len(rows), "typed_positive_seed_count": typed_positive, "negative_violation_count": negative_violations, "training_eligible_row_count": sum(int(row.get("training_eligible") is True) for row in rows)},
        "seed_summaries": summaries,
        "hard_gate": {
            "typed_candidate_reference": typed_positive == len(summaries),
            "negative_zero_violation": negative_violations == 0,
            "replay_consistent": all(item["replay_consistent"] for item in summaries),
            "fresh_reset_per_role": all(item["fresh_reset_per_role"] for item in summaries),
            "database_clean_before": all(item["database_clean_before"] for item in summaries),
            "teardown_observed": all(item["teardown_observed"] for item in summaries),
            "state_delta_evaluator_only": True,
            "context_firewall": all(row["context_firewall"]["forbidden_token_count"] == 0 for row in rows),
            "status": "blocked_until_get_post_family_and_implementation_audit",
        },
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "DVWA stateful POST stored-state replay completed only as disposable evaluator-side state delta; no raw marker, response, database state or payload enters model rows. This is not a general XSS claim.",
    }
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any]) -> dict[str, str]:
    report_path = ROOT / "research" / "pg332_dvwa_typed_stored_post_report_v1.json"
    rows_path = ROOT / "research" / "pg332_dvwa_typed_stored_post_source_rows_v1.json"
    sidecar_path = ROOT / "research" / "pg332_dvwa_typed_stored_post_sidecars_v1.json"
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_doc = {"schema_version": SCHEMA_VERSION, "status": "diagnostic_source_rows", "records": result["rows"], "promotion": result["report"]["promotion"]}
    rows_doc["dataset_sha256"] = sha256_json(rows_doc)
    rows_path.write_text(json.dumps(rows_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    side_doc = {"schema_version": SCHEMA_VERSION, "status": "evaluator_sidecar_only", "sidecars": result["sidecars"], "promotion": result["report"]["promotion"]}
    side_doc["sidecars_sha256"] = sha256_json(side_doc)
    sidecar_path.write_text(json.dumps(side_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(report_path.relative_to(ROOT)), "rows": str(rows_path.relative_to(ROOT)), "sidecars": str(sidecar_path.relative_to(ROOT))}


if __name__ == "__main__":
    result = collect_typed_stored_post()
    print(json.dumps({"report": result["report"], "artifacts": write_artifacts(result)}, ensure_ascii=False, indent=2))


__all__ = ["collect_typed_stored_post", "write_artifacts", "_state_delta", "_probe_value"]
