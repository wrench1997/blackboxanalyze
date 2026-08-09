"""Fail-closed audit for the PG-388 supplemental abstract dataset."""

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

from app.pg388_logic_invariant_projection import ROLES, SUPPLEMENTAL_LOGIC_CASES  # noqa: E402


SCHEMA_VERSION = "pg388-logic-supplement-audit-v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_dataset(document: dict[str, Any]) -> dict[str, Any]:
    rows = document.get("rows") if isinstance(document, dict) else None
    reasons: list[str] = []
    if not isinstance(rows, list):
        reasons.append("rows_missing")
        rows = []
    expected_cases = {item["case_ref"] for item in SUPPLEMENTAL_LOGIC_CASES}
    observed_cases = {str(row.get("case_ref")) for row in rows if isinstance(row, dict)}
    if observed_cases != expected_cases:
        reasons.append("supplement_case_refs_mismatch")
    if len(rows) != 1600:
        reasons.append("record_count_mismatch")
    invalid_rows = 0
    raw_hits = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        core = {key: value for key, value in row.items() if key != "row_sha256"}
        if row.get("row_sha256") != _sha(core):
            invalid_rows += 1
        if any(bool(row.get(key)) for key in ("raw_source_stored", "raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            raw_hits += 1
        if row.get("role") not in ROLES or row.get("training_eligible") is not False:
            invalid_rows += 1
    if invalid_rows:
        reasons.append("invalid_rows")
    if raw_hits:
        reasons.append("raw_context_firewall_failure")
    status = "passed_candidate_audit" if not reasons else "blocked_supplement_audit"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_id": document.get("dataset_id"),
        "records": len(rows),
        "cases": len(expected_cases),
        "invalid_rows": invalid_rows,
        "raw_context_hits": raw_hits,
        "failure_reasons": reasons,
        "source_contract": document.get("source_contract", {}),
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["audit_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="research/pg388_logic_supplement_dataset_v1.json")
    parser.add_argument("--output", default="research/pg388_logic_supplement_dataset_audit_v1.json")
    args = parser.parse_args()
    document = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    report = audit_dataset(document)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
