"""Independent audit for PG-272 independent implementation probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg272_independent_surface_probe_report_v1.json"
TRACE = ROOT / "research" / "pg272_independent_surface_probe_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg272_independent_surface_probe_protocol_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, value: bool) -> None:
        if not value:
            failures.append(name)

    check("status", report.get("status") == "completed_independent_implementation_evaluation")
    check("report_hash", report.get("report_sha256") == sha({key: value for key, value in report.items() if key != "report_sha256"}))
    check("trace_hash", trace.get("trace_sha256") == sha({key: value for key, value in trace.items() if key != "trace_sha256"}))
    check("protocol_hash", protocol.get("protocol_sha256") == sha({key: value for key, value in protocol.items() if key != "protocol_sha256"}))
    rows = list(report.get("rows", []))
    check("nine_rows", len(rows) == 9 and len({row.get("record_id") for row in rows}) == 9)
    check("positive_rows", sum(bool(row.get("expected_positive")) for row in rows) == 2)
    check("fresh_source", bool(report.get("target", {}).get("source_hash")) and report.get("target", {}).get("fresh_target") is True)
    check("context_raw_free", all(not any(term in token.casefold() for term in ("payload", "response", "oracle", "body_sha")) for row in rows for token in row.get("context_tokens", [])))
    check("oracle_off_context", report.get("training_boundary", {}).get("surface_fixture_seen_during_training") is False and trace.get("oracle_in_context") is False)
    check("promotion_blocked", report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False and report.get("promotion", {}).get("vulnerability_claim_allowed") is False)
    # The important result is allowed to be a failure: the audit verifies that
    # the gate correctly exposed missing compositional recall rather than
    # hiding it behind a high abstain/precision score.
    metrics = dict(report.get("metrics") or {})
    check("metrics_consistent", int(metrics.get("false_negative_candidate_count", -1)) == 2 and int(metrics.get("false_positive_candidate_count", -1)) == 0 and float(metrics.get("positive_recall_candidate", -1)) == 0.0)

    audit = {
        "audit_id": "pg272-independent-surface-probe-audit-v1",
        "status": "passed" if not failures else "failed",
        "all_required_fields_complete": not failures,
        "audit_checks": {
            "hashes": not any(name.endswith("_hash") for name in failures),
            "rows_and_positive_support": "nine_rows" not in failures and "positive_rows" not in failures,
            "fresh_independent_source": "fresh_source" not in failures,
            "context_firewall": "context_raw_free" not in failures and "oracle_off_context" not in failures,
            "promotion_blocked": "promotion_blocked" not in failures,
            "failure_signal_not_hidden": "metrics_consistent" not in failures,
        },
        "report": str(REPORT.relative_to(ROOT)),
        "trace": str(TRACE.relative_to(ROOT)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "gate_status_under_test": report.get("gate", {}).get("status"),
        "failures": failures,
    }
    audit["audit_sha256"] = sha(audit)
    output = ROOT / "research" / "pg272_independent_surface_probe_audit_v1.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
