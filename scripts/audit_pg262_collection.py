# -*- coding: utf-8 -*-
"""Audit PG-262 child replays and publish a bounded evidence sidecar.

The training dataset intentionally keeps abstract tokens only.  This sidecar
keeps the human/reviewer-facing causal facts needed to prove that every row
really had a fresh reset, AI/reference/negative sends, a typed oracle and an
evidence hash.  Raw payloads, request wires and response bodies are never
copied into the sidecar.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RUN_DIR = RESEARCH / "pg262_child_runs"
REPORT = RESEARCH / "pg262_targeted_paired_trace_collection_report_v1.json"
DATASET = RESEARCH / "pg262_targeted_paired_trace_collection_dataset_v1.json"
TRACE = RESEARCH / "pg262_targeted_paired_trace_collection_trace_v1.json"
AUDIT = RESEARCH / "pg262_targeted_paired_trace_collection_audit_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _bounded(value: Any) -> Any:
    """Retain projections/hashes while dropping raw material defensively."""
    if isinstance(value, list):
        return [_bounded(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        lower = str(key).lower()
        if lower in {"wire", "payload", "raw_payload", "raw_payload_value", "body", "raw_body", "response_body"}:
            continue
        if lower.startswith("raw_") and lower not in {"raw_payload_stored", "raw_response_stored", "raw_payload_strings_stored", "raw_response_bodies_stored"}:
            continue
        result[str(key)] = _bounded(item)
    return result


def _row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row.get("seed", 0) or 0), str(row.get("route") or row.get("path") or ""), str(row.get("method", "GET")).upper())


def _records_by_key(dataset: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {_row_key(row): row for row in list(dataset.get("records") or []) if isinstance(row, dict)}


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    dataset_rows = _records_by_key(dataset)
    audit_rows: list[dict[str, Any]] = []
    child_reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted(RUN_DIR.glob("*_report.json")):
        child = json.loads(path.read_text(encoding="utf-8-sig"))
        child_reports.append({"path": str(path.relative_to(ROOT)), "report_sha256": str(child.get("report_sha256", "")), "status": child.get("status")})
        family = path.name.split("_", 1)[0]
        rows = list(child.get("episodes") or child.get("results") or [])
        for source_row in rows:
            if not isinstance(source_row, dict):
                continue
            key = _row_key(source_row)
            dataset_row = dataset_rows.get(key, {})
            reset = dict(source_row.get("reset") or {})
            typed = dict(source_row.get("typed_oracle") or source_row.get("oracle") or {})
            evidence = dict(typed.get("evidence") or {})
            evidence_hash = str(typed.get("evidence_hash") or evidence.get("evidence_hash") or "")
            ai = dict(source_row.get("ai") or {})
            reference = dict(source_row.get("reference") or {})
            negative = dict(source_row.get("negative") or {})
            fresh = bool(source_row.get("fresh_target") or reset.get("fresh_target"))
            completed = bool(reset.get("completed"))
            external_network = bool(reset.get("external_network") or (source_row.get("safety") or {}).get("external_network"))
            row = {
                "family": family,
                "seed": key[0],
                "route": key[1],
                "method": key[2],
                "fields": [str(item) for item in list(source_row.get("fields") or [])],
                "target_instance_hash": str(source_row.get("target_instance_hash") or reset.get("container_id_sha256") or ""),
                "fresh_reset": fresh,
                "reset_completed": completed,
                "database_health_gate": str(reset.get("database_health_gate") or ""),
                "database_clean_contract": str(reset.get("database_clean_contract") or ""),
                "external_network": external_network,
                "ai": {"sent": bool(ai.get("sent")), "candidate_id": str((ai.get("candidate") or ai.get("selected") or {}).get("candidate_id", "")), "response_projection": _bounded((ai.get("response") or {}).get("response_projection") or (ai.get("response") or {}).get("response_projection") or {}), "evidence_hash": str((ai.get("feedback") or {}).get("evidence_hash", ""))},
                "reference": {"sent": bool(reference.get("sent")), "response_projection": _bounded((reference.get("response") or {}).get("response_projection") or (reference.get("projection") or {}).get("response_projection") or {})},
                "negative": {"sent": bool(negative.get("sent", bool(negative))), "response_projection": _bounded((negative.get("response") or negative.get("projection") or {}).get("response_projection") if isinstance(negative.get("response") or negative.get("projection"), dict) else {})},
                "typed_oracle": _bounded(typed),
                "confirmed_positive": bool(source_row.get("confirmed_positive") or typed.get("confirmed_positive")),
                "training_eligible": bool(source_row.get("training_eligible")),
                "lane": str(dataset_row.get("lane", "")),
                "failure_kind": str(dataset_row.get("failure_kind", source_row.get("failure_kind", ""))),
                "repair_action": str(dataset_row.get("repair_action", source_row.get("repair_action", ""))),
                "source_report": str(path.relative_to(ROOT)),
                "source_report_sha256": str(child.get("report_sha256", "")),
                "evidence_hash": evidence_hash,
                "raw_payload_strings_stored": bool(source_row.get("raw_payload_strings_stored", False)),
                "raw_response_bodies_stored": bool(source_row.get("raw_response_bodies_stored", False)),
            }
            required_ok = all((row["route"], row["method"], row["seed"], row["fresh_reset"], row["reset_completed"], row["ai"]["sent"], row["reference"]["sent"], row["negative"]["sent"], row["evidence_hash"])) and not row["external_network"] and not row["raw_payload_strings_stored"] and not row["raw_response_bodies_stored"]
            row["required_fields_complete"] = bool(required_ok)
            if not required_ok:
                missing.append(f"{path.name}:{key}")
            audit_rows.append(row)
    expected = int((report.get("counts") or {}).get("records", 0) or 0)
    audit = {
        "schema_version": "pg262-targeted-paired-trace-collection-audit-v1",
        "audit_id": "pg262-fresh-replay-integrity-v1",
        "report": str(REPORT.relative_to(ROOT)),
        "dataset": str(DATASET.relative_to(ROOT)),
        "trace": str(TRACE.relative_to(ROOT)),
        "loopback_only": True,
        "expected_record_count": expected,
        "audited_record_count": len(audit_rows),
        "all_required_fields_complete": not missing and len(audit_rows) == expected,
        "missing_records": missing,
        "lane_counts": dict(Counter(str(row.get("lane", "")) for row in audit_rows)),
        "family_counts": dict(Counter(str(row.get("family", "")) for row in audit_rows)),
        "confirmed_positive_count": sum(int(row["confirmed_positive"]) for row in audit_rows),
        "fresh_reset_count": sum(int(row["fresh_reset"]) for row in audit_rows),
        "ai_send_count": sum(int(row["ai"]["sent"]) for row in audit_rows),
        "reference_send_count": sum(int(row["reference"]["sent"]) for row in audit_rows),
        "negative_send_count": sum(int(row["negative"]["sent"]) for row in audit_rows),
        "evidence_hash_count": sum(int(bool(row["evidence_hash"])) for row in audit_rows),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "child_reports": child_reports,
        "records": audit_rows,
    }
    audit["audit_sha256"] = _digest(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["collection_audit"] = {"audit_id": audit["audit_id"], "audit_file": str(AUDIT.relative_to(ROOT)), "audit_sha256": audit["audit_sha256"], "all_required_fields_complete": audit["all_required_fields_complete"], "audited_record_count": len(audit_rows), "training_eligible": False}
    counts = dict(report.get("counts") or {})
    counts["source_counts"] = dict(Counter(str(row.get("source", "")) for row in list(dataset.get("records") or [])))
    report["counts"] = counts
    report["promotion"] = {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_id": audit["audit_id"], "records": len(audit_rows), "required_fields_complete": audit["all_required_fields_complete"], "fresh_reset": audit["fresh_reset_count"], "ai": audit["ai_send_count"], "reference": audit["reference_send_count"], "negative": audit["negative_send_count"], "evidence_hash": audit["evidence_hash_count"], "missing": missing, "audit_sha256": audit["audit_sha256"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["all_required_fields_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
