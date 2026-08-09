"""Independent audit for PG-276 third-implementation replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg276_third_implementation_dataset_v1.json"
REPORT = ROOT / "research" / "pg276_third_implementation_report_v1.json"
TRACE = ROOT / "research" / "pg276_third_implementation_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg276_third_implementation_protocol_v1.json"
AUDIT = ROOT / "research" / "pg276_third_implementation_audit_v1.json"


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def strip_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    check("dataset_hash", data.get("dataset_sha256") == sha(strip_hash(data, "dataset_sha256")))
    check("report_hash", report.get("report_sha256") == sha(strip_hash(report, "report_sha256")))
    check("trace_hash", trace.get("trace_sha256") == sha(strip_hash(trace, "trace_sha256")))
    check("protocol_hash", protocol.get("protocol_sha256") == sha(strip_hash(protocol, "protocol_sha256")))
    source, gpu = report.get("source", {}), report.get("source", {}).get("cuda_assignment", {})
    check("a800_gpu0_only", source.get("device") == "cuda" and gpu.get("cuda_visible_devices") == "0" and gpu.get("visible_device_count") == 1 and gpu.get("current_device") == 0 and "A800" in str(gpu.get("device_name")))
    check("dataset_binding", source.get("dataset_sha256") == data.get("dataset_sha256") and trace.get("source_dataset_sha256") == data.get("dataset_sha256"))
    check("split_contract", report.get("split", {}).get("disjoint") is True and report.get("split", {}).get("holdout_implementation") == "heterogeneous_surface_v3" and report.get("split", {}).get("old_canary_implementation") == "heterogeneous_surface_v2")
    check("context_firewall", source.get("raw_payload_in_context") is False and source.get("oracle_in_context") is False and trace.get("raw_payload_strings_stored") is False and trace.get("raw_response_bodies_stored") is False and trace.get("oracle_in_context") is False)
    check("offline_only", trace.get("evaluation_only") is True and trace.get("training_eligible") is False and trace.get("memory_write") is False)
    evaluations = report.get("evaluations", {})
    check("policy_variants", set(evaluations) == {"weighted_sft_atomic", "conservative_offline_update", "dpo_preference_update"})
    check("v3_gates", report.get("gates", {}).get("status") == "passed" and report.get("gates", {}).get("checks", {}).get("v3_positive_recall_min") is True and report.get("gates", {}).get("checks", {}).get("v3_negative_reject_min") is True and report.get("gates", {}).get("checks", {}).get("v3_false_positive_zero") is True)
    check("old_canary_gate", report.get("gates", {}).get("checks", {}).get("old_canary_positive_recall_min") is True)
    check("promotion_blocked", report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False and report.get("promotion", {}).get("vulnerability_claim_allowed") is False and report.get("gates", {}).get("checks", {}).get("promotion_blocked") is True)
    audit = {"audit_id": "pg276-third-implementation-independent-audit-v1", "status": "passed" if not failures else "failed", "audit_checks": {"hash_chain": not any(x.endswith("hash") for x in failures), "a800_gpu0_only": "a800_gpu0_only" not in failures, "dataset_binding": "dataset_binding" not in failures, "split_contract": "split_contract" not in failures, "context_firewall": "context_firewall" not in failures, "offline_only": "offline_only" not in failures, "policy_variants": "policy_variants" not in failures, "v3_gates": "v3_gates" not in failures, "old_canary_gate": "old_canary_gate" not in failures, "promotion_blocked": "promotion_blocked" not in failures}, "report": str(REPORT.relative_to(ROOT)), "failures": failures, "interpretation": "Atomic observation representation and conservative/DPO updates pass this third implementation and preserve the v2 canary, but this is still a small one-family fixture and does not authorize a vulnerability claim or memory promotion."}
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
