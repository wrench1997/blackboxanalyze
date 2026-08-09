"""Read-only audit for a PG-379 dynamic whole-page source-row replay.

The collector writes abstract source rows and evaluator sidecars separately.
This audit binds the two file hashes, validates every strict PG-331 row, checks
per-row authorization provenance, and keeps the result candidate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import validate_pg331_source_row  # noqa: E402


SCHEMA_VERSION = "pg379-dynamic-source-rows-audit-v1"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}
FORBIDDEN = ("http://", "https://", "payload=", "wire=", "response_body_text=", "oracle_answer=", "evaluator_answer=")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(*, report_path: Path, rows_path: Path, sidecars_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_document = json.loads(rows_path.read_text(encoding="utf-8"))
    sidecars_document = json.loads(sidecars_path.read_text(encoding="utf-8"))
    records = list(rows_document.get("records") or [])
    checks = [validate_pg331_source_row(record) for record in records]
    strict_failures = sorted({str(failure) for check in checks for failure in check.get("failures", [])})
    attestations = {
        str(value.get("implementation_id")): str(value.get("authorization_id"))
        for value in (report.get("attestations") or {}).values()
        if isinstance(value, dict)
    }
    authorization_matches = sum(
        attestations.get(str(record.get("source_meta", {}).get("implementation")))
        == str(record.get("source_meta", {}).get("authorization_id"))
        for record in records
    )
    row_text = rows_path.read_text(encoding="utf-8").casefold()
    forbidden_hits = {marker: row_text.count(marker) for marker in FORBIDDEN if marker in row_text}
    unique_ids = len({str(record.get("record_id")) for record in records}) == len(records)
    expected = report.get("counts") or {}
    gate = report.get("hard_gate") or {}
    failures: list[str] = []
    if report.get("status") != "completed_source_row_candidate_only":
        failures.append("report_status")
    if rows_document.get("status") != "diagnostic_candidate_only":
        failures.append("rows_status")
    if len(records) != int(expected.get("source_row_expected", -1)):
        failures.append("source_row_count")
    if sum(bool(check.get("valid")) for check in checks) != len(records):
        failures.append("strict_source_row_validation")
    if strict_failures:
        failures.append("strict_row_failures")
    if authorization_matches != len(records):
        failures.append("authorization_provenance")
    if not unique_ids:
        failures.append("duplicate_record_id")
    if forbidden_hits:
        failures.append("raw_literal_firewall")
    for key in (
        "image_attestation",
        "fresh_reset_per_role",
        "network_none_loopback_only",
        "candidate_reference_negative_replay",
        "target_slots_13",
        "pg377_adapter",
        "context_firewall",
        "negative_zero_violation",
    ):
        if gate.get(key) is not True:
            failures.append(f"hard_gate:{key}")
    if int(expected.get("training_eligible_count", 0)) != 0:
        failures.append("training_rows_present")
    if int(expected.get("capture_failure_count", 0)) != 0:
        failures.append("capture_failures")
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_candidate_source_row_audit" if not failures else "blocked_source_row_audit",
        "source_report_sha256": _sha256(report_path),
        "source_rows_sha256": _sha256(rows_path),
        "sidecars_sha256": _sha256(sidecars_path),
        "counts": {
            "records": len(records),
            "strict_valid_records": sum(bool(check.get("valid")) for check in checks),
            "strict_failure_count": len(strict_failures),
            "authorization_matches": authorization_matches,
            "unique_record_ids": len({str(record.get("record_id")) for record in records}),
            "typed_roles": int(expected.get("typed_role_count", 0)),
            "failure_repair": int(expected.get("failure_action_changed_count", 0)),
            "negative_violations": int(expected.get("negative_violation_count", 0)),
            "implementations": int(expected.get("implementation_count", 0)),
            "routes": int(expected.get("route_count", 0)),
            "seeds": int(expected.get("seed_count", 0)),
            "sidecars": len(sidecars_document.get("sidecars") or []),
        },
        "strict_failures": strict_failures,
        "forbidden_literal_hits": forbidden_hits,
        "hard_gate": dict(gate),
        "training_eligible_count": 0,
        "promotion": dict(PROMOTION),
        "interpretation": "PG-379 rows are fresh local candidate/evaluator evidence only; no training or vulnerability claim is authorized.",
    }
    output["report_sha256"] = hashlib.sha256(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(report_path=args.report, rows_path=args.rows, sidecars_path=args.sidecars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"], "counts": result["counts"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed_candidate_source_row_audit" else 2)


if __name__ == "__main__":
    main()
