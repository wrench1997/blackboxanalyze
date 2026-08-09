"""Independent audit for PG-271 fresh-seed candidate replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg271_teacher_candidate_replay_report_v1.json"
TRACE = ROOT / "research" / "pg271_teacher_candidate_replay_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg271_teacher_candidate_replay_protocol_v1.json"
SOURCE_AUDIT = ROOT / "research" / "pg271_independent_seed_failure_guided_replay_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, value: bool) -> None:
        if not value:
            failures.append(name)

    check("report_status", report.get("status") == "candidate_replay_completed")
    check("report_hash", report.get("report_sha256") == sha({key: value for key, value in report.items() if key != "report_sha256"}))
    check("trace_hash", trace.get("trace_sha256") == sha({key: value for key, value in trace.items() if key != "trace_sha256"}))
    check("protocol_hash", protocol.get("protocol_sha256") == sha({key: value for key, value in protocol.items() if key != "protocol_sha256"}))
    check("source_audit_pass", source_audit.get("status") == "passed" and source_audit.get("all_required_fields_complete") is True)
    details = list(report.get("evaluations", {}).get("fresh_seed_all", {}).get("details", []))
    check("40_unique_details", len(details) == 40 and len({item.get("record_id") for item in details}) == 40)
    check("fresh_seed_not_training_seed", int(report.get("source", {}).get("fresh_seed", 0) or 0) != 27001)
    check("family_holdout_present", int(report.get("evaluations", {}).get("fresh_seed_family_holdout", {}).get("count", 0) or 0) > 0)
    check("context_only", all(int(item.get("context_token_count", 0) or 0) > 0 for item in details))
    check("no_unsupported_positive", all(int(metrics.get("unsupported_positive_count", 1)) == 0 for metrics in report.get("evaluations", {}).values()))
    check("model_claim_blocked", report.get("capability_gate", {}).get("claim_allowed") is False)
    check("promotion_blocked", report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False and report.get("promotion", {}).get("vulnerability_claim_allowed") is False)
    checks = report.get("capability_gate", {}).get("checks", {})
    check("reported_checks_true", bool(checks) and all(bool(value) for value in checks.values()))

    audit = {
        "audit_id": "pg271-teacher-candidate-replay-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "all_required_fields_complete": not failures,
        "audit_checks": {
            "report_trace_protocol_hashes": not any(name.endswith("_hash") for name in failures),
            "fresh_source_audit": "source_audit_pass" not in failures,
            "independent_seed": "fresh_seed_not_training_seed" not in failures,
            "family_holdout": "family_holdout_present" not in failures,
            "context_only": "context_only" not in failures,
            "unsupported_positive_zero": "no_unsupported_positive" not in failures,
            "promotion_blocked": "promotion_blocked" not in failures and "model_claim_blocked" not in failures,
        },
        "report": str(REPORT.relative_to(ROOT)),
        "trace": str(TRACE.relative_to(ROOT)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "source_audit": str(SOURCE_AUDIT.relative_to(ROOT)),
        "failures": failures,
    }
    audit["audit_sha256"] = sha(audit)
    out = ROOT / "research" / "pg271_teacher_candidate_replay_audit_v1.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
