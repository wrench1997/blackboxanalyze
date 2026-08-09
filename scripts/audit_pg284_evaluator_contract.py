"""Independent audit for the PG-284 offline evaluator contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg284_evaluator_contract_report_v1.json"
TRACE = RESEARCH / "pg284_evaluator_contract_trace_v1.json"
PROTOCOL = RESEARCH / "pg284_evaluator_contract_protocol_v1.json"
AUDIT = RESEARCH / "pg284_evaluator_contract_audit_v1.json"


def _sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = list(report.get("results") or [])
    checks = {
        "report_hash": report.get("report_sha256") == _sha(_without(report, "report_sha256")),
        "trace_hash": trace.get("trace_sha256") == _sha(_without(trace, "trace_sha256")),
        "protocol_hash": protocol.get("protocol_sha256") == _sha(_without(protocol, "protocol_sha256")),
        "trace_report_binding": trace.get("report_sha256") == report.get("report_sha256"),
        "protocol_report_binding": protocol.get("report_sha256") == report.get("report_sha256"),
        "offline_status": report.get("status") == "completed_offline_pg284_evaluator_contract",
        "no_live_requests": report.get("source", {}).get("network_calls") == 0 and report.get("source", {}).get("live_replay") is False,
        "remote_unavailable_no_positive": report.get("source", {}).get("remote_docker_status") != "available" and not any(row.get("status") == "confirmed_effect" for row in rows),
        "hard_negative_blocked": all(row.get("status") == "blocked" for row in rows if row.get("hard_negative")),
        "raw_payload_excluded": all(row.get("literal_payload_stored") is False for row in rows) and report.get("contract", {}).get("literal_payload_generation") is False,
        "raw_response_excluded": all(row.get("raw_response_stored") is False for row in rows),
        "confirmation_contract_complete": set(report.get("contract", {})) >= {"typed_effect_required", "negative_control_required", "fresh_reset_required", "reference_agreement_required", "replay_consistency_required", "evidence_hash_required", "non_destructive_required"},
        "scientific_gate_blocked": report.get("scientific_gate", {}).get("status") == "blocked" and report.get("scientific_gate", {}).get("claim_allowed") is False,
        "promotion_blocked": report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False,
    }
    audit = {
        "audit_id": "pg284-evaluator-contract-independent-audit-v1",
        "schema_version": "pg284-evaluator-contract-audit-v1",
        "status": "passed" if all(checks.values()) else "blocked",
        "audit_checks": checks,
        "report": str(REPORT.relative_to(ROOT).as_posix()),
        "interpretation": "PG-284 证明 evaluator 接口的 fail-closed 合同可验证；远程 Docker 不可用时没有真实 effect 或 payload 结论。",
    }
    audit["audit_sha256"] = _sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
