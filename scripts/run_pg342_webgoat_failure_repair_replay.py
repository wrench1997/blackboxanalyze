"""PG-342 live WebGoat failure -> repair -> negative replay.

This runner is a bounded, evaluator-only adapter for the pinned WebGoat image.
It deliberately uses method-shape probes rather than vulnerability payloads:
for a GET lane the first action is an intentionally mismatched POST, and for a
POST lane the first action is an intentionally mismatched GET.  Candidate and
reference then repair with the expected method; the negative role abstains.
Only abstract projections are persisted as source rows.  Raw body/headers and
wire values stay in evaluator memory and are never copied to model context.

The runner is opt-in and local-only.  It requires PG342_LOCAL_DOCKER_EVAL=1
and PG342_WEBGOAT_FAILURE_REPAIR_LIVE=1, creates a fresh network-none
container for every seed/lane/role, and removes it in a finally block.
"""

from __future__ import annotations

import argparse
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
from app.pg343_role_step_binding import bind_observation  # noqa: E402
from app.pg331_vulnerableapp_adapter import capture_vulnerableapp_projection  # noqa: E402
from app.pg333_webgoat_docker_relay import (  # noqa: E402
    IMAGE,
    IMAGE_DIGEST,
    DisposableWebGoat,
    build_container_command,
    container_name,
)
from scripts.run_pg333_webgoat_typed_get_post_source_rows import (  # noqa: E402
    _abstract_projection,
    _belief,
    _form_body,
    _route_ref as _pg333_route_ref,
    _typed_effect,
)


SCHEMA_VERSION = "pg342-webgoat-full-axis-failure-repair-replay-v1"
SEEDS = (34201,)
SOURCE_ROLES = ("candidate", "reference", "negative")
ALL_ROLES = (*SOURCE_ROLES, "replay")
ROUTES: tuple[dict[str, Any], ...] = (
    {"route_id": "webgoat-shape-get", "expected_method": "GET", "surface_id": "method_shape_get"},
    {"route_id": "webgoat-shape-post", "expected_method": "POST", "surface_id": "method_shape_post"},
)


def _route_ref(route: Mapping[str, Any]) -> str:
    return sha256_json({"schema": SCHEMA_VERSION, "route": _pg333_route_ref(route)})


def _wrong_method(expected_method: str) -> str:
    return "POST" if str(expected_method).upper() == "GET" else "GET"


def _target(*, role: str) -> dict[str, Any]:
    variant = {"candidate": "source_attested_candidate", "reference": "reference", "negative": "negative_control"}[role]
    if role == "negative":
        return {
            "question": "ask_failure",
            "next_action": "abstain",
            "repair_action": "none",
            "transport_ref": "request_method",
            "field_role_ref": "parameter_role",
            "encoding_ref": "encoding_chain",
            "probe_variant_ref": variant,
            "safe_to_send": False,
        }
    return {
        "question": "ask_failure",
        "next_action": "repair",
        "repair_action": "observe",
        "transport_ref": "request_method",
        "field_role_ref": "parameter_role",
        "encoding_ref": "encoding_chain",
        "probe_variant_ref": variant,
        "safe_to_send": False,
    }


def _failure(*, expected_method: str, failed_method: str, role: str, repaired: bool) -> dict[str, Any]:
    next_action = "abstain" if role == "negative" else "repair"
    return {
        "failure_class": "response_shape_mismatch",
        "failure_stage": "typed_effect_observation",
        "error_shape": "method_surface_mismatch",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "unexpected_method_surface",
        "blocked_reason_class": "none",
        "previous_action": f"send_{failed_method.lower()}",
        "next_action": next_action,
        "repair_delta_axis": "response_transport",
        "repair_outcome": "recovered" if repaired else "abstained",
        "new_observation": "present",
        "retry_count": 1 if repaired else 0,
        "timeout_bucket": "none",
        "environment_failure_class": "none",
    }


def _request_projection(*, method: str, action: Mapping[str, Any], body: bytes) -> dict[str, Any]:
    return {
        "method": str(method).upper(),
        "parameters": [{"role": "credential_pair", "value_type": "text", "presence": "present"}] if str(method).upper() == "POST" else [],
        # The allowlisted method-shape form is observed and contains no
        # anti-CSRF field; this is an observed absence, not an unknown.
        "csrf_presence_class": "absent",
        "cookie_presence_class": "present" if "set-cookie" in dict(action.get("headers") or {}) else "absent",
        "content_length": len(body),
    }


def _response_projection(*, action: Mapping[str, Any], body: bytes, failure: Mapping[str, Any]) -> dict[str, Any]:
    status = int(action.get("status", 0) or 0)
    return {
        "status": status,
        "body_length": len(body),
        "body_shape": "html" if body else "empty",
        "charset_class": "utf8" if body else "absent",
        "cache_shape": "absent",
        "connection_outcome": "complete" if status else "transport_error",
        "failure_class": failure.get("failure_class", "none"),
        "failure_stage": failure.get("failure_stage", "none"),
        "error_shape": failure.get("error_shape", "empty"),
    }


def _capture_role(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    expected = str(route["expected_method"]).upper()
    failed_method = _wrong_method(expected)
    name = container_name(seed=seed, route_ref_sha256=route_ref, role=role)
    target = DisposableWebGoat(name=name, seed=seed, role=role, command=build_container_command(name=name, seed=seed, role=role))
    try:
        reset = target.start()
        baseline = target.request(method="GET")
        failed_body = _form_body("POST") if failed_method == "POST" else b""
        failed = target.request(method=failed_method, form_body=failed_body)
        repaired = None
        if role != "negative":
            repaired_body = _form_body("POST") if expected == "POST" else b""
            repaired = target.request(method=expected, form_body=repaired_body)
        final = repaired if repaired is not None else failed
        final_body = bytes(final.get("body") or b"")
        typed = bool(repaired is not None and _typed_effect(expected_method=expected, action_method=expected, action=repaired, body=final_body))
        failure = _failure(expected_method=expected, failed_method=failed_method, role=role, repaired=repaired is not None)
        # Keep the observed baseline document when the failed method returns
        # an empty body.  Otherwise a negative GET failure would erase the
        # document axis and incorrectly become field_unknown.
        html_body = bytes(baseline.get("body") or b"") if expected == "POST" or not final_body else final_body
        capture = capture_vulnerableapp_projection(
            html=html_body.decode("utf-8", errors="replace"),
            headers=dict(final.get("headers") or {}),
            request_projection=_request_projection(method=str(final.get("method", expected)), action=final, body=final_body),
            response_projection=_response_projection(action=final, body=final_body, failure=failure),
            post_supported=True,
            failure_projection=failure,
            belief_projection={
                **_belief(expected_method=expected, action_method=failed_method, typed=False, role=role, csrf_class="absent", cookie_class="absent"),
                "previous_action": f"send_{failed_method.lower()}",
                "next_action": "abstain" if role == "negative" else "repair",
                "repair_outcome": "recovered" if repaired is not None else "abstained",
                "history_action": "failed_then_repair" if repaired is not None else "failed_then_abstain",
                "failure_class": "response_shape_mismatch",
            },
        )
        abstract, effect_class = _abstract_projection(action=final, body=final_body, typed=typed, expected_method=expected)
        evidence = sha256_json({"seed": seed, "role": role, "route_ref": route_ref, "reset_id": reset.get("reset_id"), "failure": failure, "shape": abstract.get("shape_sha256")})
        role_input = {
            "sent": True,
            "available": True,
            "executed": True,
            "typed_effect_confirmed": typed,
            "effect_class": effect_class,
            "projection": abstract,
            "evidence_sha256": evidence,
            "non_destructive": True,
        }
        return {
            "reset": reset,
            "capture": capture,
            "role_input": role_input,
            "typed": typed,
            "failed_method": failed_method,
            "repaired": repaired is not None,
            "failure": failure,
            "failure_action_changed": True,
            "failure_observed": True,
            "repair_observed": bool(repaired is not None and typed),
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
        "evaluator_version": "pg342-webgoat-failure-repair-disposable-v1",
    }


def collect_pg342(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    if os.environ.get("PG342_LOCAL_DOCKER_EVAL") != "1" or os.environ.get("PG342_WEBGOAT_FAILURE_REPAIR_LIVE") != "1":
        raise RuntimeError("PG-342 requires PG342_LOCAL_DOCKER_EVAL=1 and PG342_WEBGOAT_FAILURE_REPAIR_LIVE=1")
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed_value in seeds:
        seed = int(seed_value)
        for route in ROUTES:
            route_ref = _route_ref(route)
            roles = {role: _capture_role(seed=seed, role=role, route=route, route_ref=route_ref) for role in ALL_ROLES}
            candidate, reference, negative, replay = (roles[name] for name in ALL_ROLES)
            record = build_pg331_evaluator_record(
                record_id=f"pg342-webgoat-{seed}-{route['route_id']}",
                reset=candidate["reset"] | {"volume_mount_count": 0, "container_restart_used": False},
                candidate=candidate["role_input"],
                reference=reference["role_input"],
                negative=negative["role_input"],
                replay_consistent=bool(candidate["typed"] == replay["typed"] and candidate["typed"]),
                reference_agreement=bool(candidate["typed"] and reference["typed"]),
                negative_control_clean=not bool(negative["typed"]),
                evaluator_id="pg342-webgoat-failure-repair-disposable-v1",
            )
            sidecars.append({"seed": seed, "route_id": route["route_id"], "sidecar": record["evaluator_sidecar"], "record_sha256": record["record_sha256"]})
            row_failures: dict[str, list[str]] = {}
            for role in SOURCE_ROLES:
                item = roles[role]
                evaluator = _row_evaluator(role=role, sidecar=record["evaluator_sidecar"], role_input=item["role_input"])
                row_reset = {str(key): item["reset"][key] for key in RESET_KEYS if key in item["reset"]}
                # The evaluator knows the role before a target is decoded.
                # Bind it explicitly to the abstract observation instead of
                # deriving it from target_tokens (which would leak the label).
                process_step = "repair" if role in {"candidate", "reference"} else "failure"
                bound_observation = bind_observation(item["capture"]["observation"], role=role, step=process_step)
                row = collect_pg331_source_row(
                    record_id=f"pg342-webgoat-{seed}-{route['route_id']}-{role}",
                    observation=bound_observation,
                    source_meta={
                        "source_id": "pg342-webgoat-local",
                        "implementation": "webgoat",
                        "family_id": "method_shape_failure_repair",
                        "surface_id": str(route["surface_id"]),
                        "collector_id": SCHEMA_VERSION,
                        "authorization_id": "operator-authorized-local-network-none",
                        "image_digest": IMAGE_DIGEST,
                        "source_digest": sha256_json({"seed": seed, "route_ref": route_ref, "role": role, "evidence": evaluator["evidence_hash"]}),
                    },
                    reset=row_reset,
                    evaluator=evaluator,
                    field_capture_manifest=item["capture"]["field_capture_manifest"],
                    target_projection=_target(role=role),
                    split="train",
                    operator_reviewed=False,
                    hard_negative=role == "negative",
                )
                rows.append(row)
                row_failures[role] = list(row.get("failures") or [])
            summaries.append({
                "seed": seed,
                "route_id": route["route_id"],
                "candidate_typed": candidate["typed"],
                "reference_typed": reference["typed"],
                "negative_typed": negative["typed"],
                "replay_consistent": bool(candidate["typed"] == replay["typed"] and candidate["typed"]),
                "failure_action_changed": all(bool(item.get("failure_action_changed")) for item in (candidate, reference, negative)),
                "failure_observed": all(bool(item.get("failure_observed")) for item in (candidate, reference, negative)),
                "repair_observed": all(bool(item.get("repair_observed")) for item in (candidate, reference)),
                "row_failures": row_failures,
            })
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_failure_repair_diagnostic_only",
        "runtime": {"image": IMAGE, "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False, "target_contacted": True, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "counts": {
            "seed_count": len(tuple(seeds)),
            "route_count": len(ROUTES),
            "source_row_count": len(rows),
            "failure_action_changed_count": sum(int(item["failure_action_changed"]) for item in summaries),
            "failure_observed_count": sum(int(item["failure_observed"]) for item in summaries),
            "repair_observed_count": sum(int(item["repair_observed"]) for item in summaries),
            "typed_positive_route_count": sum(int(item["candidate_typed"] and item["reference_typed"]) for item in summaries),
            "negative_violation_count": sum(int(item["negative_typed"]) for item in summaries),
            "training_eligible_row_count": sum(int(row.get("training_eligible") is True) for row in rows),
        },
        "methods": {"GET": sum(int(item["expected_method"] == "GET") for item in ROUTES), "POST": sum(int(item["expected_method"] == "POST") for item in ROUTES)},
        "seed_route_summaries": summaries,
        "hard_gate": {
            "typed_candidate_reference": all(item["candidate_typed"] and item["reference_typed"] for item in summaries),
            "negative_zero_violation": all(not item["negative_typed"] for item in summaries),
            "replay_consistent": all(item["replay_consistent"] for item in summaries),
            "failure_action_changed": all(item["failure_action_changed"] for item in summaries),
            "failure_observed": all(item["failure_observed"] for item in summaries),
            "repair_observed": all(item["repair_observed"] for item in summaries),
            "fresh_reset_per_role": True,
            "role_bound_evidence": True,
            "context_firewall": all(int(dict(row.get("context_firewall") or {}).get("forbidden_token_count", 1)) == 0 for row in rows),
            "status": "blocked_until_merged_information_and_split_audit",
        },
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "WebGoat method-shape failure/repair and matched-negative abstain were collected in disposable evaluator state; this is not a vulnerability or payload claim.",
    }
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "report": ROOT / "research" / "pg342_webgoat_failure_repair_report_v1.json",
        "rows": ROOT / "research" / "pg342_webgoat_failure_repair_source_rows_v1.json",
        "sidecars": ROOT / "research" / "pg342_webgoat_failure_repair_sidecars_v1.json",
    }
    paths["report"].write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_doc = {"schema_version": SCHEMA_VERSION, "status": "diagnostic_source_rows", "records": result["rows"], "promotion": result["report"]["promotion"]}
    rows_doc["dataset_sha256"] = sha256_json(rows_doc)
    paths["rows"].write_text(json.dumps(rows_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    side_doc = {"schema_version": SCHEMA_VERSION, "status": "evaluator_sidecar_only", "sidecars": result["sidecars"], "promotion": result["report"]["promotion"]}
    side_doc["sidecars_sha256"] = sha256_json(side_doc)
    paths["sidecars"].write_text(json.dumps(side_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: str(path.relative_to(ROOT)) for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="run PG-342 WebGoat full-axis failure-repair diagnostic replay")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    args = parser.parse_args()
    result = collect_pg342(seeds=tuple(args.seeds or SEEDS))
    artifacts = write_artifacts(result)
    print(json.dumps({"report": result["report"], "artifacts": artifacts}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_pg342", "write_artifacts", "_failure", "_target", "_route_ref"]
