"""PG-331 evaluator-side SQL replay that emits abstract source rows only.

This runner is the first bridge from real local GET/POST traffic to the strict
whole-page source-row schema.  It uses three fixed Pikachu SQL routes and four
fresh, network-none containers per route (candidate/reference/negative/replay).
The reviewed probe values and response bytes exist only in evaluator memory;
the written dataset contains structural observations, typed projections,
role-bound evidence SHA-256 values and safe Rule-IR targets.

The output is never a promotion artifact.  By default ``operator_reviewed`` is
false, so even a fully observed row remains a diagnostic candidate until a
human reviews the evaluator report.  No model checkpoint or arbitrary target
is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_pg331_evaluator_record, sha256_json
from app.pg331_loopback_adapter import _field_capture_manifest, capture_loopback
from app.pg331_pikachu_docker_relay import DisposablePikachu, IMAGE
from app.pg331_source_row import collect_pg331_source_row, validate_pg331_source_row


SCHEMA_VERSION = "pg331-pikachu-typed-source-row-v1"
TIMEZONE = "Asia/Shanghai"
SEED = 33102
REPORT_PATH = ROOT / "research" / "pg331_pikachu_typed_source_rows_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg331_pikachu_typed_source_rows_v1.json"
EVALUATOR_PATH = ROOT / "research" / "pg331_pikachu_typed_evaluator_sidecars_v1.json"
SPLIT = "implementation_holdout"
ROLES = ("candidate", "reference", "negative")
ROUTES: tuple[dict[str, Any], ...] = (
    {"id": "sql-string-get", "method": "GET", "path": "/vul/sqli/sqli_str.php", "field": "name", "submit": "submit"},
    {"id": "sql-search-get", "method": "GET", "path": "/vul/sqli/sqli_search.php", "field": "name", "submit": "submit"},
    {"id": "sql-numeric-post", "method": "POST", "path": "/vul/sqli/sqli_id.php", "field": "id", "submit": "submit"},
)

# Source-grounded, non-destructive values.  They never leave evaluator
# memory; only their evidence digest is written below.
PROBES: dict[str, dict[str, str]] = {
    "sql-string-get": {
        "candidate": "kobe' OR '1'='1",
        "reference": "kobe' OR 'a'='a",
        "negative": "kobe' AND '1'='2",
    },
    "sql-search-get": {
        "candidate": "%' OR 1=1 OR '%",
        "reference": "%' OR 2=2 OR '%",
        "negative": "%' AND 1=2 AND '%",
    },
    "sql-numeric-post": {
        "candidate": "1 OR 1=1",
        "reference": "1 OR 2=2",
        "negative": "1 AND 1=2",
    },
}
MARKER_RE = re.compile(r"your uid:|hello,|username：", re.IGNORECASE)


def _require_window() -> None:
    if os.environ.get("PG331_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-331 typed source rows require PG331_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo(TIMEZONE))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-331 local collection is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")


def _probe_values(route: Mapping[str, Any], role: str) -> dict[str, str]:
    values = {str(route["field"]): PROBES[str(route["id"])][str(role)]}
    values[str(route["submit"])] = "submit"
    return values


def _length_bucket(value: int) -> str:
    return "zero" if value <= 0 else "one" if value == 1 else "two" if value == 2 else "few" if value <= 5 else "many"


def _typed_callback(route_id: str, role: str, details: dict[str, Any]):
    def callback(body: bytes, headers: Mapping[str, Any], status: int | None) -> dict[str, Any]:
        digest = hashlib.sha256(body).hexdigest()
        text = body.decode("utf-8", errors="replace")
        marker_present = bool(MARKER_RE.search(text))
        content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].casefold() or "unknown"
        body_shape = "html" if b"<html" in body.lower() or b"<body" in body.lower() else "text" if body else "empty"
        status_class = f"{int(status) // 100}xx" if status is not None and 100 <= int(status) < 600 else "unknown"
        details.update(
            {
                "route_id": route_id,
                "role": role,
                "body_sha256": digest,
                "body_length": len(body),
                "marker_present": marker_present,
                "status": int(status or 0),
            }
        )
        return {
            "status_class": status_class,
            "content_type_class": content_type if re.fullmatch(r"[a-z0-9.+-]{1,64}", content_type) else "unknown",
            "body_shape": body_shape,
            "body_length_bucket": _length_bucket(len(body)),
            "effect_marker": "present" if marker_present else "absent",
            "effect_shape": "row_shape",
            "connection_outcome": "complete" if status is not None else "transport_error",
            "non_destructive": True,
            "database_touched": False,
        }

    return callback


def _indicator(details: Mapping[str, Any], negative: Mapping[str, Any]) -> bool:
    return bool(details.get("marker_present")) or int(details.get("body_length", 0) or 0) > int(negative.get("body_length", 0) or 0) + 200


def _capture_role(route: Mapping[str, Any], role: str, index: int) -> dict[str, Any]:
    name = f"sift-pg331-typed-{SEED}-{index}-{role}"
    target = DisposablePikachu(name, seed=SEED, index=index * 10 + ROLES.index(role))
    details: dict[str, Any] = {}
    values = _probe_values(route, role)
    try:
        reset = target.start()
        if str(route["method"]) == "GET":
            origin = target.origin + str(route["path"]) + "?" + urlencode(values)
            capture = capture_loopback(origin, method="GET", timeout=15.0, evaluator=_typed_callback(str(route["id"]), role, details))
        else:
            origin = target.origin + str(route["path"])
            capture = capture_loopback(origin, method="POST", form_data=values, timeout=15.0, evaluator=_typed_callback(str(route["id"]), role, details))
        observation = dict(capture["observation"])
        observation["failure_feedback"] = {**dict(observation.get("failure_feedback") or {}), "previous_action": "baseline_observe", "next_action": f"{role}_request", "failure_class": "none", "repair_outcome": "not_applicable"}
        observation["belief_and_replay"] = {
            "observation_presence": "present",
            "observation_delta_axis": "response_transport",
            "belief_prior_bucket": "low",
            "belief_posterior_bucket": "mid",
            "belief_delta_axis": "response_transport",
            "history_action": f"{role}_request",
            "typed_available": "present",
            "evidence_present": "present",
            "negative_control": "present",
            "fresh_reset": "present",
            "replay_ready": "present",
            "reference_present": "present",
            "candidate_present": "present",
            "step_budget": "present",
            "evidence_hash_present": "present",
            "history_length": 3,
            "probe_count": 3,
        }
        digest = sha256_json({"schema": SCHEMA_VERSION, "route_id": route["id"], "role": role, "body_sha256": details.get("body_sha256", ""), "status": details.get("status", 0)})
        return {"role": role, "reset": reset, "capture": capture, "details": details, "observation": observation, "values_digest": sha256_json(values), "evidence_sha256": digest, "target_instance_digest": reset["target_instance_digest"]}
    finally:
        target.stop()


def _capture_replay(route: Mapping[str, Any], index: int, negative: Mapping[str, Any]) -> dict[str, Any]:
    name = f"sift-pg331-typed-{SEED}-{index}-replay"
    target = DisposablePikachu(name, seed=SEED, index=index * 10 + 3)
    details: dict[str, Any] = {}
    values = _probe_values(route, "candidate")
    try:
        target.start()
        if str(route["method"]) == "GET":
            capture_loopback(target.origin + str(route["path"]) + "?" + urlencode(values), method="GET", timeout=15.0, evaluator=_typed_callback(str(route["id"]), "replay", details))
        else:
            capture_loopback(target.origin + str(route["path"]), method="POST", form_data=values, timeout=15.0, evaluator=_typed_callback(str(route["id"]), "replay", details))
        return {"details": details, "consistent": _indicator(details, negative), "evidence_sha256": sha256_json({"schema": SCHEMA_VERSION, "route_id": route["id"], "role": "replay", "body_sha256": details.get("body_sha256", ""), "status": details.get("status", 0)})}
    finally:
        target.stop()


def _target(role: str) -> dict[str, Any]:
    return {
        "question": "none",
        "next_action": "send_probe",
        "repair_action": "none",
        "transport_ref": "request_method",
        "field_role_ref": "parameter_role",
        "encoding_ref": "encoding_chain",
        "probe_variant_ref": "source_attested_candidate" if role == "candidate" else "reference" if role == "reference" else "negative_control",
        "safe_to_send": True,
    }


def _source_meta(route: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": "pg331-pikachu-typed-local",
        "implementation": "pikachu-fixed",
        "family_id": "sql",
        "surface_id": "row_shape",
        "collector_id": "pg331-typed-source-row-v1",
        "authorization_id": "operator-authorized-local-docker-loopback",
        "image_digest": IMAGE.split("@sha256:", 1)[1],
        "source_digest": sha256_json({"route_id": route["id"], "method": route["method"], "path_shape": route["path"], "evidence": evidence.get("evidence_sha256", "")}),
    }


def _summarize_evaluator(route: Mapping[str, Any], role: str, details: Mapping[str, Any], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    effect = bool(aggregate.get(f"{role}_positive", False))
    return {
        "typed_available": True,
        "negative_control": True,
        "reference_present": True,
        "candidate_present": True,
        "fresh_reset": True,
        "evidence_hash": str(aggregate["sidecar_evidence_sha256"]),
        "confirmed_positive": effect and bool(aggregate.get("typed_effect_confirmed")),
        "effect_class": "result_shape" if effect else "none",
        "evaluator_version": "pg331-sql-row-shape-regex-v1",
    }


def _run_route(route: Mapping[str, Any], index: int, *, operator_reviewed: bool) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    role_records = {role: _capture_role(route, role, index) for role in ROLES}
    candidate = role_records["candidate"]
    reference = role_records["reference"]
    negative = role_records["negative"]
    candidate_positive = _indicator(candidate["details"], negative["details"])
    reference_positive = _indicator(reference["details"], negative["details"])
    negative_clean = not bool(negative["details"].get("marker_present"))
    replay = _capture_replay(route, index, negative["details"])
    replay_consistent = bool(replay["consistent"] == candidate_positive)
    typed_effect = bool(candidate_positive and reference_positive and negative_clean and replay_consistent)
    evidence = {
        "schema": SCHEMA_VERSION,
        "route_id": route["id"],
        "candidate": {"body_sha256": candidate["details"].get("body_sha256", ""), "status": candidate["details"].get("status", 0), "marker_present": candidate["details"].get("marker_present", False)},
        "reference": {"body_sha256": reference["details"].get("body_sha256", ""), "status": reference["details"].get("status", 0), "marker_present": reference["details"].get("marker_present", False)},
        "negative": {"body_sha256": negative["details"].get("body_sha256", ""), "status": negative["details"].get("status", 0), "marker_present": negative["details"].get("marker_present", False)},
        "replay_evidence_sha256": replay["evidence_sha256"],
        "candidate_positive": candidate_positive,
        "reference_positive": reference_positive,
        "negative_clean": negative_clean,
        "replay_consistent": replay_consistent,
        "typed_effect_confirmed": typed_effect,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }
    sidecar_record = build_pg331_evaluator_record(
        record_id=f"pg331:{SEED}:{route['id']}",
        reset=candidate["reset"],
        candidate={"sent": True, "available": True, "executed": True, "typed_effect_confirmed": candidate_positive, "effect_class": "result_shape" if candidate_positive else "none", "projection": candidate["capture"].get("evaluator_projection") or {}, "evidence_sha256": candidate["evidence_sha256"]},
        reference={"sent": True, "available": True, "executed": True, "typed_effect_confirmed": reference_positive, "effect_class": "result_shape" if reference_positive else "none", "projection": reference["capture"].get("evaluator_projection") or {}, "evidence_sha256": reference["evidence_sha256"]},
        negative={"sent": True, "available": True, "executed": True, "typed_effect_confirmed": False, "effect_class": "none", "projection": negative["capture"].get("evaluator_projection") or {}, "evidence_sha256": negative["evidence_sha256"]},
        replay_consistent=replay_consistent,
        reference_agreement=bool(candidate_positive and reference_positive),
        negative_control_clean=negative_clean,
        evaluator_id="pg331-sql-row-shape-regex-v1",
    )
    aggregate = {"candidate_positive": candidate_positive, "reference_positive": reference_positive, "typed_effect_confirmed": typed_effect, "sidecar_evidence_sha256": sidecar_record["evaluator_sidecar"]["evidence_sha256"]}
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        item = role_records[role]
        evaluator = _summarize_evaluator(route, role, item["details"], aggregate)
        row = collect_pg331_source_row(
            record_id=f"pg331:{SEED}:{route['id']}:{role}",
            observation=item["observation"],
            source_meta=_source_meta(route, aggregate),
            reset=item["reset"],
            evaluator=evaluator,
            field_capture_manifest=_field_capture_manifest(item["observation"]),
            target_projection=_target(role),
            split=SPLIT,
            operator_reviewed=operator_reviewed,
            hard_negative=role == "negative",
        )
        rows.append(row)
    route_report = {"route_id": route["id"], "typed_effect_confirmed": typed_effect, "candidate_positive": candidate_positive, "reference_positive": reference_positive, "negative_clean": negative_clean, "replay_consistent": replay_consistent, "evidence_sha256": sidecar_record["evaluator_sidecar"]["evidence_sha256"], "sidecar": sidecar_record["evaluator_sidecar"]}
    return rows, route_report, sidecar_record


def run() -> dict[str, Any]:
    _require_window()
    operator_reviewed = os.environ.get("PG331_OPERATOR_REVIEWED") == "1"
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    route_reports: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, route in enumerate(ROUTES):
        try:
            route_rows, route_report, sidecar = _run_route(route, index, operator_reviewed=operator_reviewed)
            rows.extend(route_rows)
            route_reports.append(route_report)
            sidecars.append(sidecar)
        except Exception as error:
            errors.append({"route_id": str(route["id"]), "error_class": type(error).__name__})
    training_rows = sum(int(row.get("training_eligible") is True) for row in rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_diagnostic_only" if route_reports and not errors else "incomplete",
        "runtime": {"image": IMAGE, "network": "none", "fresh_container_per_role": True, "routes": len(ROUTES), "roles_per_route": 4, "external_network": False, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "operator_reviewed": operator_reviewed,
        "counts": {"route_count": len(ROUTES), "row_count": len(rows), "typed_positive_routes": sum(int(item.get("typed_effect_confirmed")) for item in route_reports), "training_eligible": training_rows, "errors": len(errors)},
        "route_reports": [{key: value for key, value in item.items() if key != "sidecar"} for item in route_reports],
        "errors": errors,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "typed effect is evaluator-only local response-shape evidence; rows remain diagnostic until source/implementation/family/entropy gates and explicit operator review pass.",
    }
    report["report_sha256"] = sha256_json(report)
    dataset: dict[str, Any] = {"schema_version": "pg331-source-row-collection-v1", "collector": "scripts/run_pg331_pikachu_typed_source_rows.py", "records": rows, "counts": {"input": len(ROUTES) * len(ROLES), "accepted": len(rows), "training_eligible": training_rows, "incomplete": sum(int(row.get("training_eligible") is not True) for row in rows), "rejected": len(errors)}, "source": {"image": IMAGE, "network": "none", "loopback_only": True, "external_network": False}, "promotion": report["promotion"]}
    dataset["dataset_sha256"] = sha256_json(dataset)
    evaluator_artifact = {"schema_version": "pg331-typed-evaluator-sidecars-v1", "route_reports": route_reports, "raw_payload_stored": False, "raw_response_bodies_stored": False, "promotion": report["promotion"]}
    evaluator_artifact["artifact_sha256"] = sha256_json(evaluator_artifact)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    EVALUATOR_PATH.write_text(json.dumps(evaluator_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": report, "dataset": dataset, "evaluator": evaluator_artifact}


def main() -> int:
    global SEED, SPLIT, REPORT_PATH, DATASET_PATH, EVALUATOR_PATH
    parser = argparse.ArgumentParser(description="PG-331 local typed source rows; evaluator bytes never persist")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="fresh replay seed; recorded in row IDs and target lifecycle")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "implementation_holdout", "family_holdout", "unassigned"),
        default=SPLIT,
        help="data split label; never changes operator review or training eligibility",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--evaluator", type=Path, default=EVALUATOR_PATH)
    args = parser.parse_args()
    SEED = int(args.seed)
    SPLIT = str(args.split)
    REPORT_PATH = (ROOT / args.report).resolve() if not args.report.is_absolute() else args.report.resolve()
    DATASET_PATH = (ROOT / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    EVALUATOR_PATH = (ROOT / args.evaluator).resolve() if not args.evaluator.is_absolute() else args.evaluator.resolve()
    for output in (REPORT_PATH, DATASET_PATH, EVALUATOR_PATH):
        if ROOT not in output.parents:
            raise SystemExit("PG-331 output paths must remain inside the workspace")
    result = run()
    print(json.dumps(result if args.json else {"status": result["report"]["status"], "counts": result["report"]["counts"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
