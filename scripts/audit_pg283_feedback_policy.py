"""Independent audit for the remote PG-283 feedback-policy run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg283_feedback_policy_report_v1.json"
TRACE = RESEARCH / "pg283_feedback_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg283_feedback_policy_protocol_v1.json"
DATASET = RESEARCH / "pg283_feedback_policy_dataset_v1.json"
DATASET_AUDIT = RESEARCH / "pg283_feedback_policy_dataset_audit_v1.json"
HARD = RESEARCH / "pg283_feedback_policy_hard_negative_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg283-feedback-policy" / "pg283_feedback_policy.pt"
AUDIT = RESEARCH / "pg283_feedback_policy_audit_v1.json"


def _sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset_audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    hard = json.loads(HARD.read_text(encoding="utf-8"))
    summary = dict(report.get("risk_weight_sweep") or {})
    variants = dict(summary.get("variants") or {})
    expected_variants = {"plain_sft", "guarded_sft", "risk_4_0"}
    checks = {
        "report_hash": report.get("report_sha256") == _sha(_without(report, "report_sha256")),
        "trace_hash": trace.get("trace_sha256") == _sha(_without(trace, "trace_sha256")),
        "protocol_hash": protocol.get("protocol_sha256") == _sha(_without(protocol, "protocol_sha256")),
        "trace_report_binding": trace.get("report_sha256") == report.get("report_sha256"),
        "protocol_report_binding": protocol.get("report_sha256") == report.get("report_sha256"),
        "dataset_audit_pass": dataset_audit.get("status") == "passed" and dataset_audit.get("dataset_sha256") == dataset.get("dataset_sha256"),
        "dataset_count_binding": int(report.get("split", {}).get("train", -1)) == int(dataset.get("counts", {}).get("train", -2)) and int(report.get("split", {}).get("hard_negative", -1)) == int(dataset.get("counts", {}).get("hard_negative", -2)),
        "hard_negative_hash_binding": report.get("source", {}).get("hard_negative_sha256") == hard.get("dataset_sha256"),
        "a800_gpu0_only": report.get("device", {}).get("cuda_visible_devices") == "0" and report.get("device", {}).get("visible_device_count") == 1 and report.get("device", {}).get("current_device") == 0 and "A800" in str(report.get("device", {}).get("device_name", "")),
        "risk_sweep_complete": set(variants) == expected_variants and summary.get("selected_variant") in expected_variants,
        "risk_sweep_safe": all(float(item.get("hard_negative_false_allow_max", 1)) == 0 and float(item.get("hard_negative_safe_reject_min", 0)) >= 0.99 for item in variants.values()),
        "hard_negative_dataset_eval_only": dataset.get("training_contract", {}).get("hard_negative_training_eligible") is False and hard.get("training_eligible") is False and hard.get("memory_promotion_allowed") is False,
        "engineering_gate_recorded": report.get("hypothesis_gate", {}).get("status") == "passed",
        "scientific_gate_blocked": report.get("scientific_gate", {}).get("status") == "blocked" and report.get("scientific_gate", {}).get("claim_allowed") is False,
        "no_live_claim": report.get("source", {}).get("live_send") is False and report.get("source", {}).get("real_application_gold_rows") == 0 and report.get("promotion", {}).get("vulnerability_claim_allowed") is False,
        "literal_payload_excluded": report.get("source", {}).get("literal_payload_in_context") is False and report.get("policy_scope", {}).get("literal_payload_generation") is False,
        "checkpoint_present": CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 0,
        "promotion_blocked": report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False,
    }
    audit = {
        "audit_id": "pg283-feedback-policy-independent-audit-v1",
        "schema_version": "pg283-feedback-policy-audit-v1",
        "status": "passed" if all(checks.values()) else "blocked",
        "audit_checks": checks,
        "report": str(REPORT.relative_to(ROOT).as_posix()),
        "dataset": str(DATASET.relative_to(ROOT).as_posix()),
        "interpretation": "工程训练/风险扫描门通过；科学门因模板化过程数据、远程 Docker/evaluator 不可用而阻断，不能声称真实 payload 能力。",
    }
    audit["audit_sha256"] = _sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
