# -*- coding: utf-8 -*-
"""Independent structural audit for the PG-267 capacity-training report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg267_payload_grounding_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg267_payload_grounding_capacity_training_dataset_v1.json"
FRESH_DATASET = RESEARCH / "pg267_payload_grounding_augmented_dataset_v1.json"
AUDIT = RESEARCH / "pg267_payload_grounding_capacity_training_audit_v1.json"


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
    report = _read(REPORT)
    dataset = _read(DATASET)
    fresh = _read(FRESH_DATASET)
    selected = dict(report.get("selected") or {})
    artifact = ROOT / str(selected.get("artifact", ""))
    judge = dict(report.get("independent_final_judge") or {})
    audit_checks = {
        "status_complete": report.get("status") == "completed_pg267_payload_grounding_capacity_training",
        "capacity_variants_exact": report.get("capacity_variants") == [8192, 12288, 16384],
        "train_steps_recorded": int(report.get("train_steps", 0) or 0) == 170,
        "combined_records_322": int((report.get("growth_counts") or {}).get("combined_records", 0) or 0) == 322,
        "pg267_records_12": int((report.get("growth_counts") or {}).get("pg267_records", 0) or 0) == 12,
        "pg267_even_holdout_6": int((report.get("growth_counts") or {}).get("pg267_even_seed_holdout", 0) or 0) == 6,
        "fresh_dataset_records_12": len(list(fresh.get("records") or [])) == 12,
        "fresh_dataset_raw_excluded": all(
            not bool(row.get("raw_payload_strings_stored")) and not bool(row.get("raw_response_bodies_stored"))
            for row in list(fresh.get("records") or [])
            if isinstance(row, dict)
        ),
        "model_input_excludes_payload_and_oracle": bool((report.get("evaluation_audit") or {}).get("payload_strings_in_model_input") is False)
        and bool((report.get("evaluation_audit") or {}).get("oracle_target_in_model_input") is False),
        "report_hash_valid": _valid_hash(report, "report_sha256"),
        "dataset_hash_valid": _valid_hash(dataset, "dataset_sha256"),
        "artifact_present": artifact.is_file(),
        "artifact_hash_valid": artifact.is_file()
        and hashlib.sha256(artifact.read_bytes()).hexdigest() == str(selected.get("artifact_sha256", "")),
        "judge_is_candidate_only": bool(judge.get("model_output_is_candidate_only"))
        and bool(judge.get("oracle_or_reference_is_not_model_input")),
        "promotion_blocked": all(not bool(value) for value in dict(report.get("promotion") or {}).values() if isinstance(value, bool)),
    }
    all_pass = all(audit_checks.values())
    report["evaluation_audit"] = dict(
        report.get("evaluation_audit") or {},
        independent_audit_id="pg267-payload-grounding-capacity-independent-audit-v1",
        independent_audit_pass=all_pass,
        structural_checks=audit_checks,
    )
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "audit_id": "pg267-payload-grounding-capacity-independent-audit-v1",
        "status": "passed" if all_pass else "failed",
        "all_required_fields_complete": all_pass,
        "audit_checks": audit_checks,
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": report["report_sha256"],
        "artifact": str(artifact.relative_to(ROOT)),
    }
    result["audit_sha256"] = _digest(result)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
