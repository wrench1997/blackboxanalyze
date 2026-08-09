"""Independent audit for PG-275 falsifiable hypothesis ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg273_composition_dataset_v1.json"
REPORT = ROOT / "research" / "pg275_hypothesis_ablation_report_v1.json"
TRACE = ROOT / "research" / "pg275_hypothesis_ablation_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg275_hypothesis_ablation_protocol_v1.json"
AUDIT = ROOT / "research" / "pg275_hypothesis_ablation_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def no_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
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

    check("dataset_hash", data.get("dataset_sha256") == sha(no_hash(data, "dataset_sha256")))
    check("report_hash", report.get("report_sha256") == sha(no_hash(report, "report_sha256")))
    check("trace_hash", trace.get("trace_sha256") == sha(no_hash(trace, "trace_sha256")))
    check("protocol_hash", protocol.get("protocol_sha256") == sha(no_hash(protocol, "protocol_sha256")))
    source = report.get("source", {})
    gpu = source.get("cuda_assignment", {})
    check("a800_gpu0_only", source.get("device") == "cuda" and gpu.get("cuda_visible_devices") == "0" and gpu.get("visible_device_count") == 1 and gpu.get("current_device") == 0 and "A800" in str(gpu.get("device_name")))
    check("dataset_binding", source.get("dataset_sha256") == data.get("dataset_sha256") and trace.get("source_dataset_sha256") == data.get("dataset_sha256"))
    check("context_firewall", source.get("raw_payload_in_context") is False and source.get("oracle_in_context") is False and trace.get("raw_payload_strings_stored") is False and trace.get("raw_response_bodies_stored") is False and trace.get("oracle_in_context") is False)
    check("offline_only", trace.get("evaluation_only") is True and trace.get("training_eligible") is False and trace.get("memory_write") is False)
    evaluations = report.get("evaluations", {})
    required = {"weighted_sft_minimal", "weighted_sft_atomic", "weighted_sft_collapsed", "conservative_offline_update", "dpo_preference_update"}
    check("all_variants", required.issubset(evaluations))
    atomic = evaluations.get("weighted_sft_atomic", {}).get("v2_holdout", {})
    minimal = evaluations.get("weighted_sft_minimal", {}).get("v2_holdout", {})
    collapsed = evaluations.get("weighted_sft_collapsed", {}).get("v2_holdout", {})
    conservative = evaluations.get("conservative_offline_update", {}).get("v2_holdout", {})
    dpo = evaluations.get("dpo_preference_update", {}).get("v2_holdout", {})
    check("representation_result", atomic.get("positive_recall") == 1.0 and minimal.get("positive_recall") == 0.0 and collapsed.get("positive_recall") == 0.0 and atomic.get("false_positive_count") == 0)
    check("conservative_no_regression", conservative.get("positive_recall") == atomic.get("positive_recall") and conservative.get("negative_reject") == atomic.get("negative_reject") and conservative.get("false_positive_count") == 0)
    check("dpo_no_regression", dpo.get("positive_recall") == atomic.get("positive_recall") and dpo.get("negative_reject") == atomic.get("negative_reject") and dpo.get("false_positive_count") == 0)
    check("promotion_blocked", report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False and report.get("promotion", {}).get("claim_allowed") is False)
    audit = {"audit_id": "pg275-hypothesis-ablation-independent-audit-v1", "status": "passed" if not failures else "failed", "audit_checks": {"hash_chain": not any(x.endswith("hash") for x in failures), "a800_gpu0_only": "a800_gpu0_only" not in failures, "dataset_binding": "dataset_binding" not in failures, "context_firewall": "context_firewall" not in failures, "offline_only": "offline_only" not in failures, "all_variants": "all_variants" not in failures, "representation_result": "representation_result" not in failures, "conservative_no_regression": "conservative_no_regression" not in failures, "dpo_no_regression": "dpo_no_regression" not in failures, "promotion_blocked": "promotion_blocked" not in failures}, "report": str(REPORT.relative_to(ROOT)), "failures": failures, "interpretation": "Atomic observation tokens carry compositional signal; collapsed/minimal representations lose positive recall. Conservative/DPO updates preserve the atomic SFT result in this split, but third implementation and fresh-seed validation remain mandatory."}
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
