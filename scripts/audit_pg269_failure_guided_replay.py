# -*- coding: utf-8 -*-
"""Independent structural audit for PG-269 mentor-guided traces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RUN_TAG = os.environ.get("PG269_RUN_TAG", "pg269_failure_guided_replay")
CATALOG = RESEARCH / f"{RUN_TAG}_catalog_v1.json"
DATASET = RESEARCH / f"{RUN_TAG}_dataset_v1.json"
REPORT = RESEARCH / f"{RUN_TAG}_report_v1.json"
AUDIT = RESEARCH / f"{RUN_TAG}_audit_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _valid_hash(document: dict[str, Any], field: str) -> bool:
    saved = str(document.get(field, ""))
    if not saved:
        return False
    copy = dict(document)
    copy[field] = ""
    return saved == _digest(copy)


def main() -> int:
    catalog = _read(CATALOG)
    dataset = _read(DATASET)
    report = _read(REPORT)
    entries = [row for row in list(catalog.get("entries") or []) if isinstance(row, dict)]
    records = [row for row in list(dataset.get("records") or []) if isinstance(row, dict)]
    counts = dict(report.get("counts") or {})
    context_forbidden = (
        "oracle",
        "payload",
        "echo_excerpt",
        "response_body",
        "body_sha256",
        "confirmed_positive",
        "outcome_class",
    )
    context_strings = [
        str(token).casefold()
        for row in records
        for token in list(row.get("context_tokens") or [])
    ]
    complete_entries = [
        row
        for row in entries
        if bool((row.get("final") or {}).get("fresh_complete"))
        and bool((row.get("source") or {}).get("source_sha256"))
        and len(list(row.get("steps") or [])) >= 3
        and bool((row.get("final") or {}).get("evidence_hash"))
    ]
    checks = {
        "status_complete": report.get("status") == "completed_local_failure_guided_replay",
        "surface_count_40": len(entries) == 40 and int(counts.get("surface_count", -1)) == 40,
        "get_post_counts_match": int(counts.get("get_count", -1)) + int(counts.get("post_count", -1)) == 40,
        "fresh_complete_all": len(complete_entries) == 40 and int(counts.get("complete_count", -1)) == 40,
        "source_attested_all": int(counts.get("source_attested_count", -1)) == 40,
        "multi_step_all": all(len(list(row.get("steps") or [])) >= 3 for row in entries),
        "repair_labels_consistent": all(
            not bool((row.get("final") or {}).get("repair_attempted"))
            or any(str(step.get("phase")) == "repair" for step in list(row.get("steps") or []))
            or any(str(step.get("phase")) == "diagnose" for step in list(row.get("steps") or []))
            for row in entries
        ),
        "context_target_split": all(
            isinstance(row.get("context_tokens"), list)
            and isinstance(row.get("target_tokens"), list)
            and row.get("context_tokens")
            and row.get("target_tokens")
            for row in records
        ),
        "context_no_oracle_payload_response": not any(
            any(term in token for term in context_forbidden) for token in context_strings
        ),
        "dataset_raw_values_excluded": all(
            row.get("raw_payload_strings_stored") is False
            and row.get("raw_response_bodies_stored") is False
            and "payload" not in row.get("context_tokens", [])
            for row in records
        ),
        "dataset_record_count_matches": len(records) == len(entries)
        and int((dataset.get("counts") or {}).get("records", -1)) == len(records),
        "catalog_hash_valid": _valid_hash(catalog, "catalog_sha256"),
        "dataset_hash_valid": _valid_hash(dataset, "dataset_sha256"),
        "report_hash_valid": _valid_hash(report, "report_sha256"),
        "no_false_positive": int(counts.get("false_positive_count", -1)) == 0,
        "promotion_blocked": all(
            not bool(value)
            for key, value in dict(report.get("promotion") or {}).items()
            if isinstance(value, bool) and key.endswith("_allowed")
        ),
        "training_contract_split": bool((dataset.get("contract") or {}).get("context_target_split"))
        and bool((dataset.get("contract") or {}).get("oracle_and_response_off_context_input")),
    }
    all_pass = all(checks.values())
    report["evaluation_audit"] = {
        "audit_id": f"{RUN_TAG}-independent-audit-v1",
        "independent_audit_pass": all_pass,
        "structural_checks": checks,
        "context_forbidden_terms": list(context_forbidden),
        "oracle_target_in_context": False,
        "raw_payload_in_context": False,
    }
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "audit_id": f"{RUN_TAG}-independent-audit-v1",
        "status": "passed" if all_pass else "failed",
        "all_required_fields_complete": all_pass,
        "audit_checks": checks,
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": report["report_sha256"],
    }
    result["audit_sha256"] = _digest(result)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
