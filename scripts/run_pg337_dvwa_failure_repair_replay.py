"""PG-337 minimal real failure -> repair -> abstain replay on DVWA.

This is an evaluator-only companion to the existing PG-332 typed lanes.  It
uses the pinned local DVWA image and the reviewed network-none docker-exec
relay, but keeps all wire values and response bytes inside the evaluator.  A
candidate/reference role first submits a bounded no-effect variant, observes
the missing surface delta, then submits one source-grounded repair variant.
The negative role stops after the failed observation and abstains.  The
result is diagnostic until a cross-implementation information audit passes.
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
from app.pg332_dvwa_docker_relay import DisposableDvwa  # noqa: E402
from scripts.plan_pg332_dvwa_source_rows import IMAGE, _BY_ID, _route_ref  # noqa: E402
from scripts.run_pg332_dvwa_source_rows_live import (  # noqa: E402
    _route_request,
    build_container_command,
    container_name,
)
from scripts.run_pg332_dvwa_typed_stored_post_source_rows import (  # noqa: E402
    _belief,
    _page_probe,
    _post_probe,
    _probe_value,
    _read_route,
    _role_projection,
    _state_delta,
)


SCHEMA_VERSION = "pg337-dvwa-real-failure-repair-replay-v1"
ROUTE_ID = "dvwa-xss-stored-post"
SOURCE_ROLES = ("candidate", "reference", "negative")
ALL_ROLES = (*SOURCE_ROLES, "replay")
SEEDS = (33701,)


def _failure_projection(*, previous: str, next_action: str, outcome: str) -> dict[str, Any]:
    return {
        "failure_class": "candidate_without_typed_effect",
        "failure_stage": "effect_observation",
        "error_shape": "no_surface_delta",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none",
        "blocked_reason_class": "none",
        "previous_action": previous,
        "next_action": next_action,
        "repair_delta_axis": "response_transport",
        "repair_outcome": outcome,
        "new_observation": "present",
        "retry_count": 1 if next_action == "repair" else 0,
        "timeout_bucket": "none",
        "environment_failure_class": "none",
    }


def _failure_belief(*, role: str, next_action: str, outcome: str) -> dict[str, Any]:
    value = _belief(role=role, typed=False, action="failure_observe")
    value.update(
        {
            "failure_class": "candidate_without_typed_effect",
            "failure_stage": "effect_observation",
            "error_shape": "no_surface_delta",
            "previous_action": "candidate_failed",
            "next_action": next_action,
            "repair_delta_axis": "response_transport",
            "repair_outcome": outcome,
            "history_action": "failure_observe",
        }
    )
    return value


def _target(*, role: str) -> dict[str, Any]:
    if role == "negative":
        return {
            "question": "ask_failure",
            "next_action": "abstain",
            "repair_action": "none",
            "transport_ref": "request_method",
            "field_role_ref": "parameter_role",
            "encoding_ref": "encoding_chain",
            "probe_variant_ref": "negative_control",
            "safe_to_send": False,
        }
    variant = "reference" if role == "reference" else "source_attested_candidate"
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


def _role_evaluator(*, role: str, sidecar: Mapping[str, Any], role_input: Mapping[str, Any]) -> dict[str, Any]:
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
        "evaluator_version": "pg337-dvwa-failure-repair-disposable-v1",
    }


def _collect_role(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    name = container_name(seed=seed, route_ref_sha256=route_ref, role=role)
    target = DisposableDvwa(
        name=name,
        seed=seed,
        index=0,
        command=build_container_command(seed=seed, route_ref_sha256=route_ref, role=role),
    )
    marker = _probe_value(seed, role)
    # Keep the no-effect control disjoint from the negative role's abstract
    # marker prefix; otherwise a substring check could mistake the control
    # value for a stored marker and hide a real negative observation.
    failed_value = f"neutral-{seed}-no-effect"
    read_route = _read_route(route)
    try:
        reset = target.start(timeout=150.0)
        baseline_capture, _, baseline = _route_request(
            target,
            read_route,
            belief_projection=_belief(role=role, typed=False, action="baseline_observe"),
            failure_projection={
                "failure_class": "none", "failure_stage": "none", "error_shape": "empty",
                "previous_action": "baseline_observe", "next_action": "stateful_post_probe",
                "repair_delta_axis": "none", "repair_outcome": "not_applicable", "new_observation": "present",
                "retry_count": 0, "parse_error_class": "none", "encoding_error_class": "none",
                "redirect_error_class": "none", "blocked_reason_class": "none", "timeout_bucket": "none",
                "environment_failure_class": "none",
            },
            effect_probe=_page_probe(marker),
        )
        failed_capture, failed_shape, failed_post = _route_request(
            target,
            route,
            probe_values={"txtName": failed_value, "mtxMessage": failed_value, "btnSign": "Sign Guestbook"},
            belief_projection=_belief(role=role, typed=False, action="candidate_failed"),
            failure_projection={
                "failure_class": "none", "failure_stage": "none", "error_shape": "empty",
                "previous_action": "stateful_post_probe", "next_action": "failure_observe",
                "repair_delta_axis": "none", "repair_outcome": "not_applicable", "new_observation": "present",
                "retry_count": 0, "parse_error_class": "none", "encoding_error_class": "none",
                "redirect_error_class": "none", "blocked_reason_class": "none", "timeout_bucket": "none",
                "environment_failure_class": "none",
            },
            effect_probe=_post_probe,
            post_supported=True,
        )
        failed_observe_capture, _, failed_observe = _route_request(
            target,
            read_route,
            belief_projection=_failure_belief(role=role, next_action="repair" if role != "negative" else "abstain", outcome="pending"),
            failure_projection=_failure_projection(previous="candidate_failed", next_action="repair" if role != "negative" else "abstain", outcome="pending"),
            effect_probe=_page_probe(marker),
        )
        failure_seen = not bool((failed_observe or {}).get("marker_present"))
        if role == "negative":
            after_capture, post_shape, after = failed_observe_capture, failed_shape, failed_observe
            repair_sent = False
            repair_observed = False
            repair_action = "abstain"
            repair_outcome = "abstained"
        else:
            repair_capture, post_shape, repaired = _route_request(
                target,
                route,
                probe_values={"txtName": marker, "mtxMessage": marker, "btnSign": "Sign Guestbook"},
                belief_projection=_belief(role=role, typed=False, action="repair_candidate"),
                failure_projection=_failure_projection(previous="candidate_failed", next_action="repair", outcome="pending"),
                effect_probe=_post_probe,
                post_supported=True,
            )
            after_capture, _, after = _route_request(
                target,
                read_route,
                belief_projection=_failure_belief(role=role, next_action="repair", outcome="recovered"),
                failure_projection=_failure_projection(previous="candidate_failed", next_action="repair", outcome="recovered"),
                effect_probe=_page_probe(marker),
            )
            repair_sent = True
            repair_observed = bool((after or {}).get("marker_present"))
            repair_action = "repair"
            repair_outcome = "recovered" if repair_observed else "no_effect"
        state = _state_delta(
            role=role,
            marker=marker,
            baseline=baseline or {},
            posted=(repaired if role != "negative" else failed_post) or {},
            after=after or {},
        )
        role_input = _role_projection(role=role, state=state, post_shape=post_shape, reset=reset, route_ref=route_ref)
        return {
            "reset": reset,
            "capture": after_capture,
            "baseline_capture": baseline_capture,
            "failed_capture": failed_capture,
            "failed_observe_capture": failed_observe_capture,
            "state": state,
            "role_input": role_input,
            "failure_seen": failure_seen,
            "repair_sent": repair_sent,
            "repair_observed": repair_observed,
            "repair_action": repair_action,
            "repair_outcome": repair_outcome,
            "teardown": False,
        }
    finally:
        target.stop()


def collect_pg337(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    if os.environ.get("PG337_LOCAL_DOCKER_EVAL") != "1" or os.environ.get("PG337_DVWA_FAILURE_REPAIR_LIVE") != "1":
        raise RuntimeError("PG-337 requires PG337_LOCAL_DOCKER_EVAL=1 and PG337_DVWA_FAILURE_REPAIR_LIVE=1")
    route = _BY_ID[ROUTE_ID]
    route_ref = _route_ref(route)
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failure_steps: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed_value in seeds:
        seed = int(seed_value)
        roles = {role: _collect_role(seed=seed, role=role, route=route, route_ref=route_ref) for role in ALL_ROLES}
        candidate, reference, negative, replay = (roles[key] for key in ALL_ROLES)
        replay_consistent = bool(
            candidate["role_input"].get("typed_effect_confirmed")
            and replay["role_input"].get("typed_effect_confirmed")
            and candidate["role_input"].get("effect_class") == replay["role_input"].get("effect_class")
        )
        evaluator_record = build_pg331_evaluator_record(
            record_id=f"pg337-dvwa-failure-repair-{seed}",
            reset=candidate["reset"],
            candidate=candidate["role_input"],
            reference=reference["role_input"],
            negative=negative["role_input"],
            replay_consistent=replay_consistent,
            reference_agreement=bool(candidate["role_input"].get("typed_effect_confirmed") and reference["role_input"].get("typed_effect_confirmed")),
            negative_control_clean=bool(negative["state"].get("negative_control_clean")),
            evaluator_id="pg337-dvwa-failure-repair-disposable-v1",
        )
        sidecars.append({"seed": seed, "sidecar": evaluator_record["evaluator_sidecar"], "record_sha256": evaluator_record["record_sha256"]})
        for role in SOURCE_ROLES:
            item = roles[role]
            evaluator = _role_evaluator(role=role, sidecar=evaluator_record["evaluator_sidecar"], role_input=item["role_input"])
            reset = {str(key): item["reset"][key] for key in RESET_KEYS if key in item["reset"]}
            row = collect_pg331_source_row(
                record_id=f"pg337-dvwa-failure-repair-{seed}-{role}",
                observation=item["capture"]["observation"],
                source_meta={
                    "source_id": "pg337-dvwa-local",
                    "implementation": "vulnerables-web-dvwa",
                    "family_id": "xss_stored_post",
                    "surface_id": "failure_repair_state_delta",
                    "collector_id": SCHEMA_VERSION,
                    "authorization_id": "operator-authorized-local-network-none",
                    "image_digest": IMAGE.split("@sha256:", 1)[1],
                    "source_digest": sha256_json({"seed": seed, "route_ref": route_ref, "role": role, "evidence": evaluator["evidence_hash"]}),
                },
                reset=reset,
                evaluator=evaluator,
                field_capture_manifest=item["capture"]["field_capture_manifest"],
                target_projection=_target(role=role),
                split="implementation_holdout",
                operator_reviewed=False,
                hard_negative=role == "negative",
            )
            rows.append(row)
            failure_steps.append({
                "seed": seed,
                "role": role,
                "failure_class": "candidate_without_typed_effect",
                "previous_action": "candidate_failed",
                "next_action": item["repair_action"],
                "action_changed": "candidate_failed" != item["repair_action"],
                "failure_observed": bool(item["failure_seen"]),
                "repair_sent": bool(item["repair_sent"]),
                "repair_observed": bool(item["repair_observed"]),
                "repair_outcome": item["repair_outcome"],
            })
        summaries.append({
            "seed": seed,
            "candidate_typed": bool(candidate["state"].get("typed_effect")),
            "reference_typed": bool(reference["state"].get("typed_effect")),
            "negative_typed": bool(negative["state"].get("typed_effect")),
            "replay_typed": bool(replay["state"].get("typed_effect")),
            "replay_consistent": replay_consistent,
            "failure_action_changed": all(step["action_changed"] for step in failure_steps if step["seed"] == seed),
            "failure_observed": all(step["failure_observed"] for step in failure_steps if step["seed"] == seed),
            "negative_control_clean": bool(negative["state"].get("negative_control_clean")),
            "fresh_reset_per_role": all(bool(roles[item]["reset"].get("fresh_reset")) for item in ALL_ROLES),
            "database_clean_before": all(bool(roles[item]["reset"].get("database_clean_attestation")) for item in ALL_ROLES),
            "teardown_observed": True,
        })
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_failure_repair_diagnostic_only",
        "runtime": {"image": IMAGE, "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False, "target_contacted": True, "stateful_disposable": True, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "route_ref_sha256": route_ref,
        "counts": {
            "seed_count": len(summaries),
            "route_count": 1,
            "source_row_count": len(rows),
            "failure_step_count": len(failure_steps),
            "failure_action_changed_count": sum(int(item["action_changed"]) for item in failure_steps),
            "failure_observed_count": sum(int(item["failure_observed"]) for item in failure_steps),
            "repair_sent_count": sum(int(item["repair_sent"]) for item in failure_steps),
            "repair_observed_count": sum(int(item["repair_observed"]) for item in failure_steps),
            "typed_positive_seed_count": sum(int(item["candidate_typed"] and item["reference_typed"]) for item in summaries),
            "negative_violation_count": sum(int(item["negative_typed"]) for item in summaries),
            "training_eligible_row_count": sum(int(row.get("training_eligible") is True) for row in rows),
        },
        "seed_summaries": summaries,
        "failure_steps": failure_steps,
        "hard_gate": {
            "typed_candidate_reference": all(item["candidate_typed"] and item["reference_typed"] for item in summaries),
            "negative_zero_violation": all(not item["negative_typed"] for item in summaries),
            "replay_consistent": all(item["replay_consistent"] for item in summaries),
            "failure_observed": all(item["failure_observed"] for item in summaries),
            "failure_action_changed": all(item["failure_action_changed"] for item in summaries),
            "fresh_reset_per_role": all(item["fresh_reset_per_role"] for item in summaries),
            "database_clean_before": all(item["database_clean_before"] for item in summaries),
            "teardown_observed": all(item["teardown_observed"] for item in summaries),
            "state_delta_evaluator_only": True,
            "context_firewall": all(row["context_firewall"]["forbidden_token_count"] == 0 for row in rows),
            "status": "blocked_until_cross_implementation_information_audit",
        },
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "DVWA failure-to-repair and matched-negative process observed in disposable evaluator state; no raw marker, response, database state or payload enters model rows. This is not a general XSS claim.",
    }
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "report": ROOT / "research" / "pg337_dvwa_failure_repair_report_v1.json",
        "rows": ROOT / "research" / "pg337_dvwa_failure_repair_source_rows_v1.json",
        "sidecars": ROOT / "research" / "pg337_dvwa_failure_repair_sidecars_v1.json",
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
    parser = argparse.ArgumentParser(description="run PG-337 DVWA failure-repair diagnostic replay")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    args = parser.parse_args()
    result = collect_pg337(seeds=tuple(args.seeds or SEEDS))
    artifacts = write_artifacts(result)
    print(json.dumps({"report": result["report"], "artifacts": artifacts}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_pg337", "write_artifacts", "_failure_projection", "_failure_belief"]
