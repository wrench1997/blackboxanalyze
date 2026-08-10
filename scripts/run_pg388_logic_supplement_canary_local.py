"""Replay the fine-grained PG-388 logic contracts through the local WSGI fixture.

This is an evaluator-only lane.  It deliberately invokes the checked-in
``fixtures.pg388.logic_lab`` application in memory instead of opening a
socket, starting Docker, or accepting arbitrary values.  Each role gets a
fresh reset before and after its lane.  The replay lane is seeded by an
abstract candidate transition inside the same disposable episode so the
state-difference oracle can distinguish a replay effect without persisting
business data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg388_logic_invariant_projection import SUPPLEMENTAL_LOGIC_CASES
from fixtures.pg388.logic_lab import IMPLEMENTATION_ID, SCHEMA_VERSION as FIXTURE_SCHEMA, application, source_digest


SCHEMA_VERSION = "pg388-logic-supplement-canary-local-v1"
SEEDS = (38851, 38852, 38853)
ROLES = ("candidate", "reference", "negative", "replay")
_OBSERVATION_KEYS = (
    "status",
    "case_ref",
    "role",
    "phase",
    "state_before",
    "state_after",
    "state_delta",
    "effect_shape",
    "action_shape",
    "invariant_holds",
    "vulnerable_effect",
    "typed_observation",
    "negative_control_clean",
    "safe_to_send",
    "target_contacted",
    "external_network",
    "persistent_storage",
    "fresh_reset_required",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _invoke(method: str, path: str, payload: Mapping[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    """Call only the fixed local WSGI application and return a JSON object."""

    encoded = b"" if payload is None else json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
    status_line: list[str] = []

    def start_response(status: str, _headers: list[tuple[str, str]]) -> None:
        status_line.append(status)

    environ: dict[str, Any] = {
        "REQUEST_METHOD": method.upper(),
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(encoded)),
        "CONTENT_TYPE": "application/json" if payload is not None else "",
        "wsgi.input": BytesIO(encoded),
        "wsgi.url_scheme": "http",
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "0",
    }
    body = b"".join(application(environ, start_response))
    if not status_line:
        raise RuntimeError("supplement_canary_missing_status")
    try:
        status_code = int(status_line[0].split(" ", 1)[0])
        document = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("supplement_canary_invalid_response") from exc
    if not isinstance(document, dict):
        raise RuntimeError("supplement_canary_response_object_required")
    return status_code, document


def _reset() -> dict[str, Any]:
    status, document = _invoke("POST", "/api/reset", {})
    if status != 200 or document.get("status") != "fresh_reset" or document.get("state_clean") is not True:
        raise RuntimeError("supplement_canary_reset_contract_failure")
    if document.get("external_network") is not False or document.get("persistent_storage") is not False:
        raise RuntimeError("supplement_canary_reset_safety_failure")
    return {
        "fresh_reset": True,
        "state_clean": True,
        "state_delta": "zero",
        "external_network": False,
        "persistent_storage": False,
    }


def _project(document: Mapping[str, Any], *, case_ref: str, role: str, phase: str) -> dict[str, Any]:
    missing = [key for key in _OBSERVATION_KEYS if key not in document]
    if missing:
        raise RuntimeError("supplement_canary_missing_typed_fields")
    row = {key: document[key] for key in _OBSERVATION_KEYS}
    if row["status"] != "typed_local_canary_result" or row["typed_observation"] is not True:
        raise RuntimeError("supplement_canary_not_typed")
    if row["case_ref"] != case_ref or row["role"] != role or row["phase"] != phase:
        raise RuntimeError("supplement_canary_role_binding_failure")
    if row["safe_to_send"] is not False or row["target_contacted"] is not False:
        raise RuntimeError("supplement_canary_send_boundary_failure")
    if row["external_network"] is not False or row["persistent_storage"] is not False:
        raise RuntimeError("supplement_canary_state_boundary_failure")
    return row


def _observe(case_ref: str, role: str, phase: str) -> dict[str, Any]:
    status, document = _invoke(
        "POST",
        "/api/canary",
        {"case_ref": case_ref, "role": role, "phase": phase},
    )
    if status != 200:
        raise RuntimeError("supplement_canary_http_contract_failure")
    return _project(document, case_ref=case_ref, role=role, phase=phase)


def _lane(case_ref: str, seed: int, role: str) -> tuple[dict[str, Any], int, int]:
    """Return one role row plus before/after reset counts and setup count."""

    reset_before = _reset()
    setup_observations = 0
    if role == "candidate":
        row = _observe(case_ref, "candidate", "candidate")
    elif role == "reference":
        row = _observe(case_ref, "reference", "reference")
    elif role == "negative":
        row = _observe(case_ref, "negative", "negative")
    else:
        # The evaluator-only setup is intentionally not emitted as a model row.
        _observe(case_ref, "candidate", "baseline")
        _observe(case_ref, "candidate", "candidate")
        setup_observations = 2
        row = _observe(case_ref, "replay", "replay")
    reset_after = _reset()
    projected = dict(row)
    projected.update(
        {
            "seed": seed,
            "lane": "replay_seeded_candidate" if role == "replay" else "fresh_role_lane",
            "fresh_reset_before": reset_before["fresh_reset"],
            "fresh_reset_after": reset_after["fresh_reset"],
            "state_clean_before": reset_before["state_clean"],
            "state_clean_after": reset_after["state_clean"],
            "replay_seeded": role == "replay",
            "setup_observations": setup_observations,
        }
    )
    projected["evidence_sha256"] = _sha(
        {
            "schema_version": SCHEMA_VERSION,
            "fixture_schema": FIXTURE_SCHEMA,
            "fixture_source_sha256": source_digest(),
            "case_ref": case_ref,
            "seed": seed,
            "role": role,
            "phase": projected["phase"],
            "state_before": projected["state_before"],
            "state_after": projected["state_after"],
            "state_delta": projected["state_delta"],
            "effect_shape": projected["effect_shape"],
            "action_shape": projected["action_shape"],
        }
    )
    return projected, 1, 1


def _blocked(reason: str, rows: list[dict[str, Any]], *, failures: int = 1) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_incomplete_local_supplemental_canary",
        "implementation_id": IMPLEMENTATION_ID,
        "fixture_source_sha256": source_digest(),
        "rows": rows,
        "counts": {
            "cases": len(SUPPLEMENTAL_LOGIC_CASES),
            "seeds": len(SEEDS),
            "roles": len(ROLES),
            "role_rows": len(rows),
            "fresh_resets_before": len(rows),
            "fresh_resets_after": len(rows),
            "setup_observations": sum(int(row.get("setup_observations", 0)) for row in rows),
            "typed_observations": sum(bool(row.get("typed_observation")) for row in rows),
            "candidate_effects": sum(bool(row.get("vulnerable_effect")) and row.get("role") == "candidate" for row in rows),
            "replay_effects": sum(bool(row.get("vulnerable_effect")) and row.get("role") == "replay" for row in rows),
            "negative_control_clean": sum(bool(row.get("negative_control_clean")) for row in rows),
            "negative_violation": sum(row.get("role") == "negative" and bool(row.get("vulnerable_effect")) for row in rows),
            "unsafe_allow": sum(row.get("safe_to_send") is True for row in rows),
            "failures": failures,
        },
        "failure_reason": reason,
        "execution": {
            "in_process_only": True,
            "docker_started": False,
            "target_contacted": False,
            "external_network": False,
            "wire_created": False,
        },
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    report["report_sha256"] = _sha(report)
    return report


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        health_status, health = _invoke("GET", "/health")
        manifest_status, manifest = _invoke("GET", "/api/manifest")
        if health_status != 200 or health.get("status") != "ok":
            return _blocked("supplement_canary_health_contract_failure", rows)
        if manifest_status != 200 or manifest.get("implementation_id") != IMPLEMENTATION_ID:
            return _blocked("supplement_canary_manifest_contract_failure", rows)
        expected_cases = {item["case_ref"] for item in SUPPLEMENTAL_LOGIC_CASES}
        if not expected_cases.issubset({item.get("case_ref") for item in manifest.get("canary", {}).get("cases", [])}):
            return _blocked("supplement_canary_case_manifest_incomplete", rows)
        for seed in SEEDS:
            for case in SUPPLEMENTAL_LOGIC_CASES:
                case_ref = case["case_ref"]
                for role in ROLES:
                    row, _before, _after = _lane(case_ref, seed, role)
                    rows.append(row)
    except RuntimeError as exc:
        return _blocked(str(exc), rows)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_local_supplemental_canary_diagnostic",
        "implementation_id": IMPLEMENTATION_ID,
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_source_sha256": source_digest(),
        "cases": [item["case_ref"] for item in SUPPLEMENTAL_LOGIC_CASES],
        "seeds": list(SEEDS),
        "roles": list(ROLES),
        "rows": rows,
        "counts": {
            "cases": len(SUPPLEMENTAL_LOGIC_CASES),
            "seeds": len(SEEDS),
            "roles": len(ROLES),
            "role_rows": len(rows),
            "fresh_resets_before": len(rows),
            "fresh_resets_after": len(rows),
            "setup_observations": sum(int(row["setup_observations"]) for row in rows),
            "typed_observations": sum(bool(row["typed_observation"]) for row in rows),
            "candidate_effects": sum(bool(row["vulnerable_effect"]) and row["role"] == "candidate" for row in rows),
            "replay_effects": sum(bool(row["vulnerable_effect"]) and row["role"] == "replay" for row in rows),
            "negative_control_clean": sum(bool(row["negative_control_clean"]) for row in rows),
            "negative_violation": sum(row["role"] == "negative" and bool(row["vulnerable_effect"]) for row in rows),
            "unsafe_allow": sum(row["safe_to_send"] is True for row in rows),
        },
        "contract": {
            "fresh_reset_per_role_before_after": True,
            "replay_seeded_by_candidate_in_same_disposable_lane": True,
            "source_seed_variation": False,
            "source_seed_note": "seeds partition repeated evaluator lanes; fixture implementation is unchanged",
            "abstract_enum_input_only": True,
            "raw_values_stored": False,
            "raw_response_stored": False,
            "evaluator_answer_in_context": False,
        },
        "execution": {
            "in_process_only": True,
            "docker_started": False,
            "target_contacted": False,
            "external_network": False,
            "wire_created": False,
            "persistent_storage": False,
        },
        "model_boundary": {
            "context_rows_emitted": False,
            "raw_request_stored": False,
            "raw_response_stored": False,
            "evaluator_answer_in_context": False,
            "safe_to_send": False,
        },
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    report["report_sha256"] = _sha(report)
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg388_logic_supplement_canary_local_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run()
    write_report(args.output, report)
    print(json.dumps({"output": args.output, "status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
