"""Independent audit for PG-282 evaluator-only binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg282_evaluator_binding_report_v1.json"
AUDIT = RESEARCH / "pg282_evaluator_binding_audit_v1.json"


def _sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = list(report.get("results") or [])
    checks = {
        "report_status_is_offline_contract": report.get("status") == "completed_offline_pg282_binding_contract",
        "remote_docker_status_recorded": report.get("source", {}).get("remote_docker_status") in {"available", "unavailable", "unreachable", "ssh_unavailable"},
        "no_live_replay": report.get("source", {}).get("live_replay") is False,
        "remote_unavailable_has_no_positive": report.get("source", {}).get("remote_docker_status") == "available" or not any(row.get("status") == "confirmed_positive" for row in rows),
        "all_hard_negatives_abstain": all(row.get("status") == "abstain" for row in rows if row.get("hard_negative")),
        "literal_payload_excluded": all(row.get("literal_payload_stored") is False for row in rows) and report.get("binding_contract", {}).get("literal_payload_generation") is False,
        "raw_response_excluded": all(row.get("raw_response_stored") is False for row in rows) and report.get("binding_contract", {}).get("raw_response_storage") is False,
        "promotion_blocked": report.get("promotion", {}).get("training_eligible") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False,
        "confirmation_contract_complete": set(report.get("binding_contract", {}).get("confirmation_requirements", [])) == {"typed_effect", "negative_control_clean", "fresh_reset", "reference_agreement", "replay_consistent", "evidence_hash", "non_destructive"},
        "row_evidence_hashes_present": all(isinstance(row.get("binding_evidence_sha256"), str) and len(row["binding_evidence_sha256"]) == 64 for row in rows),
    }
    audit = {
        "schema_version": "pg282-evaluator-binding-audit-v1",
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": report.get("report_sha256") if report.get("report_sha256") == _sha(_without(report, "report_sha256")) else _sha(report),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "blocked",
        "interpretation": "PG-282 只验证抽象计划绑定契约；未接通 evaluator 时不得把 candidate 或 abstain 当成真实漏洞结果。",
    }
    audit["audit_sha256"] = _sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
