"""Independent structural audit for PG-274 score/SFT/offline-RL ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg273_composition_dataset_v1.json"
REPORT = ROOT / "research" / "pg274_score_rl_report_v1.json"
TRACE = ROOT / "research" / "pg274_score_rl_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg274_score_rl_protocol_v1.json"
AUDIT = ROOT / "research" / "pg274_score_rl_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def without_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, value: bool) -> None:
        if not value:
            failures.append(name)

    train = [row for row in data.get("records", []) if row.get("split") == "implementation_v1_train"]
    holdout = [row for row in data.get("records", []) if row.get("split") == "implementation_v2_holdout"]
    check("dataset_hash", data.get("dataset_sha256") == sha(without_hash(data, "dataset_sha256")))
    check("report_hash", report.get("report_sha256") == sha(without_hash(report, "report_sha256")))
    check("trace_hash", trace.get("trace_sha256") == sha(without_hash(trace, "trace_sha256")))
    check("protocol_hash", protocol.get("protocol_sha256") == sha(without_hash(protocol, "protocol_sha256")))
    check("split_contract", len(train) == 36 and len(holdout) == 36 and {x.get("implementation") for x in train} == {"heterogeneous_surface_v1"} and {x.get("implementation") for x in holdout} == {"heterogeneous_surface_v2"})
    source = report.get("source", {})
    assignment = source.get("cuda_assignment", {})
    check("a800_gpu0_only", source.get("device") == "cuda" and assignment.get("cuda_visible_devices") == "0" and assignment.get("visible_device_count") == 1 and assignment.get("current_device") == 0 and "A800" in str(assignment.get("device_name")))
    check("dataset_binding", source.get("dataset_sha256") == data.get("dataset_sha256") and trace.get("source_dataset_sha256") == data.get("dataset_sha256"))
    check("context_firewall", source.get("raw_payload_in_context") is False and source.get("oracle_in_context") is False and trace.get("raw_payload_strings_stored") is False and trace.get("raw_response_bodies_stored") is False and trace.get("oracle_in_context") is False)
    check("offline_only", report.get("training", {}).get("online_target_requests") is False and report.get("training", {}).get("memory_write") is False and trace.get("evaluation_only") is True and trace.get("training_eligible") is False)
    evaluations = report.get("evaluations", {})
    check("all_ablation_variants", set(evaluations) == {"plain_sft", "score_weighted_sft", "score_rl"})
    rl = evaluations.get("score_rl", {}).get("v2_holdout", {})
    weighted = evaluations.get("score_weighted_sft", {}).get("v2_holdout", {})
    check("rl_result_bound", trace.get("evaluations", {}).get("score_rl", {}).get("v2_holdout", {}).get("positive_recall") == rl.get("positive_recall"))
    check("rl_regression_surfaced", rl.get("positive_recall", 0) < weighted.get("positive_recall", 0) and report.get("capability_gate", {}).get("status") == "blocked" and report.get("capability_gate", {}).get("claim_allowed") is False)
    check("promotion_blocked", report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False and report.get("promotion", {}).get("vulnerability_claim_allowed") is False and report.get("capability_gate", {}).get("checks", {}).get("promotion_blocked") is True)
    audit = {
        "audit_id": "pg274-score-rl-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {
            "hash_chain": not any(x.endswith("hash") for x in failures),
            "split_contract": "split_contract" not in failures,
            "a800_gpu0_only": "a800_gpu0_only" not in failures,
            "dataset_binding": "dataset_binding" not in failures,
            "context_firewall": "context_firewall" not in failures,
            "offline_only": "offline_only" not in failures,
            "ablation_variants": "all_ablation_variants" not in failures,
            "rl_regression_surfaced": "rl_regression_surfaced" not in failures,
            "promotion_blocked": "promotion_blocked" not in failures,
        },
        "dataset": str(DATASET.relative_to(ROOT)),
        "report": str(REPORT.relative_to(ROOT)),
        "failures": failures,
        "capability_gate": report.get("capability_gate"),
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
