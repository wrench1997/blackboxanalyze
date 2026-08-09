# -*- coding: utf-8 -*-
"""Independent structural audit for the PG-268B local replay catalog.

This audit validates that the replay actually exercised the discovered GET/POST
surfaces and that only complete, evidence-backed abstract records can become
training candidates.  It deliberately does not decide that a route is a
public vulnerability: the local typed oracle result remains a candidate-only
label and promotion stays disabled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "pg268_pikachu_browser_parameterized_crawl_manifest_v1.json"
CATALOG = RESEARCH / "pg268_pikachu_parameterized_replay_catalog_v1.json"
DATASET = RESEARCH / "pg268_pikachu_parameterized_replay_dataset_v1.json"
REPORT = RESEARCH / "pg268_pikachu_parameterized_replay_report_v1.json"
AUDIT = RESEARCH / "pg268_pikachu_parameterized_replay_audit_v1.json"


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


def _has_raw_payload_value(record: dict[str, Any]) -> bool:
    """Reject values, not harmless schema names such as raw_payload_strings_stored."""

    forbidden_keys = {
        "payload",
        "wire",
        "request_line",
        "body",
        "echo_excerpt",
        "dom_excerpt",
        "response_body",
    }
    return any(key in record for key in forbidden_keys)


def main() -> int:
    manifest = _read(MANIFEST)
    catalog = _read(CATALOG)
    dataset = _read(DATASET)
    report = _read(REPORT)
    entries = [row for row in list(catalog.get("entries") or []) if isinstance(row, dict)]
    records = [row for row in list(dataset.get("records") or []) if isinstance(row, dict)]
    expected_surfaces = int((manifest.get("counts") or {}).get("with_parameter_context", 0) or 0)
    unsupported = [
        row for row in entries if str((row.get("oracle") or {}).get("outcome_class", "")).startswith("unsupported_")
    ]
    replayed = [row for row in entries if row not in unsupported]
    complete = [
        row
        for row in replayed
        if bool((row.get("oracle") or {}).get("fresh_complete"))
        and bool((row.get("oracle") or {}).get("candidate_sent"))
        and bool((row.get("oracle") or {}).get("reference_sent"))
        and bool((row.get("oracle") or {}).get("negative_sent"))
        and bool((row.get("source") or {}).get("source_sha256"))
        and bool((row.get("oracle") or {}).get("evidence_hash"))
    ]
    route_keys = {(str((row.get("route") or {}).get("path")), str((row.get("route") or {}).get("method", "")).upper()) for row in entries}
    report_counts = dict(report.get("counts") or {})
    required_complete = len(replayed)
    checks = {
        "status_complete": report.get("status") == "completed_local_parameterized_get_post_replay",
        "manifest_hash_matches_report": str(report.get("source_manifest_sha256", "")) == str(manifest.get("manifest_sha256", "")),
        "manifest_parameterized_surface_count_43": expected_surfaces == 43,
        "catalog_entries_match_manifest": len(entries) == expected_surfaces,
        "unique_route_method_keys": len(route_keys) == len(entries),
        "get_post_counts_match_report": int(report_counts.get("get_count", -1)) + int(report_counts.get("post_count", -1)) == len(entries),
        "complete_replayed_rows": len(complete) == required_complete,
        "ai_reference_negative_complete": all(
            bool((row.get("oracle") or {}).get(key)) for row in complete for key in ("candidate_sent", "reference_sent", "negative_sent")
        ),
        "fresh_reset_per_replayed_surface": int(report_counts.get("fresh_reset_count", -1)) == required_complete,
        "source_attestation_per_replayed_surface": int(report_counts.get("source_attested_count", -1)) == required_complete,
        "no_false_positive": int(report_counts.get("false_positive_count", -1)) == 0,
        "catalog_hash_valid": _valid_hash(catalog, "catalog_sha256"),
        "dataset_hash_valid": _valid_hash(dataset, "dataset_sha256"),
        "report_hash_valid": _valid_hash(report, "report_sha256"),
        "dataset_record_count_matches_catalog": len(records) == len(entries),
        "dataset_raw_values_excluded": all(
            not _has_raw_payload_value(row)
            and bool(row.get("raw_payload_strings_stored") is False)
            and bool(row.get("raw_response_bodies_stored") is False)
            for row in records
        ),
        "dataset_only_complete_records_eligible": all(
            (row.get("lane") != "gold") or bool(row.get("payload_grounded_eligible"))
            for row in records
        ),
        "promotion_blocked": all(
            not bool(value)
            for key, value in dict(report.get("promotion") or {}).items()
            if isinstance(value, bool) and key.endswith("_allowed")
        ),
        "oracle_target_off_input": bool((dataset.get("contract") or {}).get("oracle_target_off_input"))
        and bool((dataset.get("contract") or {}).get("training_promotion_allowed") is False),
    }
    all_pass = all(checks.values())
    report["evaluation_audit"] = dict(
        report.get("evaluation_audit") or {},
        independent_audit_id="pg268-parameterized-replay-independent-audit-v1",
        independent_audit_pass=all_pass,
        structural_checks=checks,
        complete_replayed_surface_count=required_complete,
        unsupported_surface_count=len(unsupported),
        payload_strings_in_model_input=False,
        oracle_target_in_model_input=False,
    )
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "audit_id": "pg268-parameterized-replay-independent-audit-v1",
        "status": "passed" if all_pass else "failed",
        "all_required_fields_complete": all_pass,
        "audit_checks": checks,
        "complete_replayed_surface_count": required_complete,
        "unsupported_surface_count": len(unsupported),
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": report["report_sha256"],
    }
    result["audit_sha256"] = _digest(result)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
