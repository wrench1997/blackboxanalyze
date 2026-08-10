"""Read-only audit for the PG-388 supplemental local canary report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg388_logic_invariant_projection import SUPPLEMENTAL_LOGIC_CASES  # noqa: E402
from fixtures.pg388.logic_lab import source_digest  # noqa: E402
from scripts.run_pg388_logic_supplement_canary_local import ROLES, SCHEMA_VERSION, SEEDS, _sha  # noqa: E402


AUDIT_SCHEMA_VERSION = "pg388-logic-supplement-canary-local-audit-v1"
MAX_BYTES = 32 * 1024 * 1024
_FORBIDDEN_MARKERS = ("http://", "https://", "payload=", "wire=", "response_body=", "<script")


def _failure(code: str) -> dict[str, Any]:
    return {"code": code}


def _evidence_basis(row: dict[str, Any], fixture_schema: str, fixture_source_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_schema": fixture_schema,
        "fixture_source_sha256": fixture_source_sha256,
        "case_ref": row.get("case_ref"),
        "seed": row.get("seed"),
        "role": row.get("role"),
        "phase": row.get("phase"),
        "state_before": row.get("state_before"),
        "state_after": row.get("state_after"),
        "state_delta": row.get("state_delta"),
        "effect_shape": row.get("effect_shape"),
        "action_shape": row.get("action_shape"),
    }


def audit(path: str | Path = "research/pg388_logic_supplement_canary_local_v1.json") -> dict[str, Any]:
    source = Path(path)
    failures: list[dict[str, Any]] = []
    if not source.exists():
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_missing_report", "failures": [_failure("missing_report")], "training_eligible": 0}
    if source.stat().st_size > MAX_BYTES:
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_report_too_large", "failures": [_failure("report_too_large")], "training_eligible": 0}
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_invalid_report", "failures": [_failure("invalid_report")], "training_eligible": 0}
    if not isinstance(document, dict):
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_invalid_report", "failures": [_failure("report_object_required")], "training_eligible": 0}

    claimed_report_hash = document.get("report_sha256")
    without_hash = dict(document)
    without_hash.pop("report_sha256", None)
    if not isinstance(claimed_report_hash, str) or claimed_report_hash != _sha(without_hash):
        failures.append(_failure("report_hash_mismatch"))
    if document.get("schema_version") != SCHEMA_VERSION:
        failures.append(_failure("schema_version_mismatch"))
    if document.get("status") != "completed_local_supplemental_canary_diagnostic":
        failures.append(_failure("unexpected_status"))
    if document.get("fixture_source_sha256") != source_digest():
        failures.append(_failure("fixture_source_hash_mismatch"))
    if document.get("training_eligible") != 0:
        failures.append(_failure("training_eligible_not_zero"))
    if document.get("promotion") != {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }:
        failures.append(_failure("promotion_not_closed"))
    if document.get("execution") != {
        "in_process_only": True,
        "docker_started": False,
        "target_contacted": False,
        "external_network": False,
        "wire_created": False,
        "persistent_storage": False,
    }:
        failures.append(_failure("execution_boundary_failure"))

    rows = document.get("rows")
    if not isinstance(rows, list):
        rows = []
        failures.append(_failure("rows_list_required"))
    expected_cases = {item["case_ref"] for item in SUPPLEMENTAL_LOGIC_CASES}
    expected_rows = len(expected_cases) * len(SEEDS) * len(ROLES)
    if len(rows) != expected_rows:
        failures.append(_failure("row_count_mismatch"))
    role_counts = {role: 0 for role in ROLES}
    case_counts = {case_ref: 0 for case_ref in expected_cases}
    evidence_hashes: set[str] = set()
    candidate_effects = replay_effects = negative_clean = negative_violation = unsafe_allow = 0
    for row in rows:
        if not isinstance(row, dict):
            failures.append(_failure("row_object_required"))
            continue
        serialized = json.dumps(row, ensure_ascii=False).casefold()
        if any(marker in serialized for marker in _FORBIDDEN_MARKERS):
            failures.append(_failure("raw_marker_in_row"))
        role = row.get("role")
        case_ref = row.get("case_ref")
        if role not in role_counts:
            failures.append(_failure("unknown_role"))
        else:
            role_counts[role] += 1
        if case_ref not in case_counts:
            failures.append(_failure("unknown_case"))
        else:
            case_counts[case_ref] += 1
        if row.get("fresh_reset_before") is not True or row.get("fresh_reset_after") is not True:
            failures.append(_failure("fresh_reset_missing"))
        if row.get("state_clean_before") is not True or row.get("state_clean_after") is not True:
            failures.append(_failure("state_clean_missing"))
        if row.get("safe_to_send") is not False or row.get("target_contacted") is not False:
            failures.append(_failure("send_boundary_failure"))
        if row.get("external_network") is not False or row.get("persistent_storage") is not False:
            failures.append(_failure("state_boundary_failure"))
        claimed_evidence = row.get("evidence_sha256")
        if not isinstance(claimed_evidence, str) or claimed_evidence != _sha(_evidence_basis(row, str(document.get("fixture_schema", "")), str(document.get("fixture_source_sha256", "")))):
            failures.append(_failure("evidence_hash_mismatch"))
        else:
            evidence_hashes.add(claimed_evidence)
        if role == "candidate" and row.get("vulnerable_effect") is True:
            candidate_effects += 1
        if role == "replay" and row.get("vulnerable_effect") is True:
            replay_effects += 1
        if role == "negative" and row.get("negative_control_clean") is True:
            negative_clean += 1
        if role == "negative" and row.get("vulnerable_effect") is True:
            negative_violation += 1
        if row.get("safe_to_send") is True:
            unsafe_allow += 1

    counts = {
        "cases": len(expected_cases),
        "seeds": len(SEEDS),
        "roles": len(ROLES),
        "role_rows": len(rows),
        "role_counts": role_counts,
        "case_counts": case_counts,
        "unique_evidence_hashes": len(evidence_hashes),
        "candidate_effects": candidate_effects,
        "replay_effects": replay_effects,
        "negative_control_clean": negative_clean,
        "negative_violation": negative_violation,
        "unsafe_allow": unsafe_allow,
    }
    if any(count != len(expected_cases) * len(SEEDS) for role, count in role_counts.items()):
        failures.append(_failure("role_count_mismatch"))
    if any(count != len(SEEDS) * len(ROLES) for count in case_counts.values()):
        failures.append(_failure("case_count_mismatch"))
    if len(evidence_hashes) != len(rows):
        failures.append(_failure("evidence_not_role_bound"))
    if candidate_effects != len(expected_cases) * len(SEEDS) or replay_effects != len(expected_cases) * len(SEEDS):
        failures.append(_failure("typed_effect_missing"))
    if negative_clean != len(expected_cases) * len(SEEDS) or negative_violation != 0:
        failures.append(_failure("negative_control_failure"))
    if unsafe_allow != 0:
        failures.append(_failure("unsafe_allow_present"))

    result: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed_candidate_only" if not failures else "blocked_supplement_canary_contract",
        "input_report": str(source),
        "input_report_sha256": claimed_report_hash,
        "counts": counts,
        "failures": failures,
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "scope": "local_in_process_evaluator_only;not_a_general_vulnerability_or_payload_claim",
    }
    result["audit_sha256"] = _sha(result)
    return result


def write_audit(path: str | Path, result: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="research/pg388_logic_supplement_canary_local_v1.json")
    parser.add_argument("--output", default="research/pg388_logic_supplement_canary_local_audit_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.input)
    write_audit(args.output, result)
    print(json.dumps({"output": args.output, "status": result["status"], "counts": result.get("counts", {}), "audit_sha256": result["audit_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()

