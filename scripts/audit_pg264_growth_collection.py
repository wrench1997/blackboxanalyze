# -*- coding: utf-8 -*-
"""Independent integrity audit for the PG-264 fresh Pikachu tranche."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RUN_DIR = RESEARCH / "pg264_growth_child_runs"
REPORT = RESEARCH / "pg264_pikachu_growth_collection_report_v1.json"
DATASET = RESEARCH / "pg264_pikachu_growth_collection_dataset_v1.json"
TRACE = RESEARCH / "pg264_pikachu_growth_collection_trace_v1.json"
AUDIT = RESEARCH / "pg264_pikachu_growth_collection_audit_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row.get("seed", 0) or 0), str(row.get("route") or row.get("path") or ""), str(row.get("method", "GET")).upper()


def _bounded_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    forbidden = {"wire", "payload", "raw_payload", "raw_payload_value", "body", "raw_body", "response_body"}
    return {str(key): item for key, item in value.items() if str(key).lower() not in forbidden and not str(key).lower().startswith("raw_")}


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    dataset_by_key = {_row_key(row): row for row in list(dataset.get("records") or []) if isinstance(row, dict)}
    audited: list[dict[str, Any]] = []
    missing: list[str] = []
    child_reports: list[dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("*_report.json")):
        child = json.loads(path.read_text(encoding="utf-8-sig"))
        child_reports.append({"path": str(path.relative_to(ROOT)), "status": child.get("status"), "report_sha256": str(child.get("report_sha256", ""))})
        episodes = list(child.get("episodes") or child.get("results") or [])
        for source in episodes:
            if not isinstance(source, dict):
                continue
            seed, route, method = _row_key(source)
            reset = dict(source.get("reset") or {})
            ai = dict(source.get("ai") or {})
            reference = dict(source.get("reference") or {})
            negative = dict(source.get("negative") or {})
            typed = dict(source.get("typed_oracle") or source.get("oracle") or {})
            evidence = dict(typed.get("evidence") or {})
            evidence_hash = str(typed.get("evidence_hash") or evidence.get("evidence_hash") or "")
            external_network = bool(reset.get("external_network") or (source.get("safety") or {}).get("external_network"))
            row = {
                "family": path.name.split("_", 1)[0],
                "seed": seed,
                "route": route,
                "method": method,
                "target_instance_hash": str(source.get("target_instance_hash") or reset.get("container_id_sha256") or ""),
                "fresh_reset": bool(source.get("fresh_target") or reset.get("fresh_target")),
                "reset_completed": bool(reset.get("completed")),
                "ai_sent": bool(ai.get("sent")),
                "reference_sent": bool(reference.get("sent")),
                "negative_sent": bool(negative.get("sent", bool(negative))),
                "typed_oracle_available": bool(typed.get("oracle_available", True)),
                "confirmed_positive": bool(source.get("confirmed_positive") or typed.get("confirmed_positive")),
                "evidence_hash": evidence_hash,
                "external_network": external_network,
                "raw_payload_strings_stored": bool(source.get("raw_payload_strings_stored", False) or typed.get("raw_payload_strings_stored", False)),
                "raw_response_bodies_stored": bool(source.get("raw_response_bodies_stored", False) or typed.get("raw_response_bodies_stored", False)),
                "typed_projection": _bounded_projection(typed),
                "lane": str(dataset_by_key.get((seed, route, method), {}).get("lane", "")),
            }
            row["required_fields_complete"] = bool(
                route and method in {"GET", "POST"} and seed and row["fresh_reset"] and row["reset_completed"] and row["ai_sent"] and row["reference_sent"] and row["negative_sent"] and evidence_hash and not external_network and not row["raw_payload_strings_stored"] and not row["raw_response_bodies_stored"]
            )
            if not row["required_fields_complete"]:
                missing.append(f"{path.name}:{seed}:{method}:{route}")
            audited.append(row)
    expected = int((report.get("counts") or {}).get("records", 0) or 0)
    expected_seeds = set(range(26401, 26409)) | set(range(26411, 26419)) | set(range(26421, 26429)) | set(range(26431, 26439))
    actual_seeds = {int(row["seed"]) for row in audited}
    audit = {
        "schema_version": "pg264-pikachu-growth-collection-audit-v1",
        "audit_id": "pg264-fresh-replay-integrity-v1",
        "report": str(REPORT.relative_to(ROOT)),
        "dataset": str(DATASET.relative_to(ROOT)),
        "trace": str(TRACE.relative_to(ROOT)),
        "loopback_only": True,
        "expected_record_count": expected,
        "audited_record_count": len(audited),
        "expected_seed_count": len(expected_seeds),
        "actual_seed_count": len(actual_seeds),
        "seed_schedule_complete": actual_seeds == expected_seeds,
        "all_required_fields_complete": not missing and len(audited) == expected and actual_seeds == expected_seeds,
        "missing_records": missing,
        "family_counts": dict(Counter(str(row["family"]) for row in audited)),
        "method_counts": dict(Counter(str(row["method"]) for row in audited)),
        "lane_counts": dict(Counter(str(row["lane"]) for row in audited)),
        "fresh_reset_count": sum(int(row["fresh_reset"]) for row in audited),
        "ai_send_count": sum(int(row["ai_sent"]) for row in audited),
        "reference_send_count": sum(int(row["reference_sent"]) for row in audited),
        "negative_send_count": sum(int(row["negative_sent"]) for row in audited),
        "evidence_hash_count": sum(int(bool(row["evidence_hash"])) for row in audited),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "child_reports": child_reports,
        "records": audited,
    }
    audit["audit_sha256"] = _digest(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["collection_audit"] = {"audit_id": audit["audit_id"], "audit_file": str(AUDIT.relative_to(ROOT)), "audit_sha256": audit["audit_sha256"], "all_required_fields_complete": audit["all_required_fields_complete"], "audited_record_count": len(audited), "training_eligible": False}
    report["promotion"] = {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_id": audit["audit_id"], "records": len(audited), "complete": audit["all_required_fields_complete"], "families": audit["family_counts"], "methods": audit["method_counts"], "missing": missing, "audit_sha256": audit["audit_sha256"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["all_required_fields_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

