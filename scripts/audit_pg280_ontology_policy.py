"""Independent final audit for PG-280 ontology/process supervision study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg280_shared_ontology_dataset_v1.json"
DATASET_AUDIT = ROOT / "research" / "pg280_shared_ontology_dataset_audit_v1.json"
REPORT = ROOT / "research" / "pg280_ontology_policy_report_v1.json"
TRACE = ROOT / "research" / "pg280_ontology_policy_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg280_ontology_policy_protocol_v1.json"
AUDIT = ROOT / "research" / "pg280_ontology_policy_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset_audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    check("dataset_hash", dataset.get("dataset_sha256") == sha(without(dataset, "dataset_sha256")))
    check("dataset_audit_hash", dataset_audit.get("audit_sha256") == sha(without(dataset_audit, "audit_sha256")))
    check("report_hash", report.get("report_sha256") == sha(without(report, "report_sha256")))
    check("trace_hash", trace.get("trace_sha256") == sha(without(trace, "trace_sha256")))
    check("protocol_hash", protocol.get("protocol_sha256") == sha(without(protocol, "protocol_sha256")))
    check("dataset_audit_pass", dataset_audit.get("status") == "passed" and not dataset_audit.get("failures"))
    source = dict(report.get("source") or {})
    gpu = dict(source.get("cuda_assignment") or {})
    check("remote_scope", source.get("remote_host") == "112.111.7.91:60228" and source.get("remote_docker_available") is False and source.get("loopback_only") is True)
    check("a800_gpu0_only", gpu.get("cuda_visible_devices") == "0" and gpu.get("visible_device_count") == 1 and gpu.get("current_device") == 0 and "A800" in str(gpu.get("device_name")))
    check("report_dataset_binding", source.get("dataset_sha256") == dataset.get("dataset_sha256") and source.get("dataset_audit_sha256") == dataset_audit.get("audit_sha256"))
    check("trace_protocol_binding", trace.get("source_dataset_sha256") == dataset.get("dataset_sha256") and trace.get("report_sha256") == report.get("report_sha256") and protocol.get("report_sha256") == report.get("report_sha256"))
    check("status_namespace", report.get("status") == "completed_remote_pg280_ontology_policy_study" and report.get("schema_version") == "pg280-shared-ontology-policy-report-v1")
    ident = dict(report.get("identifiability") or {})
    check("identifiability_gate", float(ident.get("conditional_entropy_bits", 0.0) or 0.0) > 0.0 and float(ident.get("bayes_error_lower_bound", 0.0) or 0.0) >= 0.49)
    comparison = dict(report.get("comparison") or {})
    final_only = dict(comparison.get("final_only") or {})
    process = dict(comparison.get("process") or {})
    check("final_only_no_pre_supervision", int(final_only.get("pre_supervision_rows", -1)) == 0)
    check("process_pre_supervision_present", int(process.get("pre_supervision_rows", 0)) == 288)
    check("process_ask_safe", float(process.get("missing_ask_rate_min", 0.0) or 0.0) >= 0.90 and float(process.get("missing_safe_non_supported_min", 0.0) or 0.0) >= 0.95)
    check("final_only_classification_visible", float(final_only.get("post_transition_accuracy_min", 0.0) or 0.0) >= 0.80)
    check("hard_negative_lane", int((report.get("family_ood_hard_negative") or {}).get("rows", 0) or 0) == 48 and (report.get("family_ood_hard_negative") or {}).get("training_eligible") is False)
    check("promotion_blocked", (report.get("promotion") or {}).get("training_allowed") is False and (report.get("promotion") or {}).get("memory_promotion_allowed") is False and (report.get("promotion") or {}).get("vulnerability_claim_allowed") is False)
    check("docker_probe_honest", (report.get("docker_probe") or {}).get("status") == "unavailable" and (report.get("docker_probe") or {}).get("docker_binary") is False)
    gate = dict(report.get("hypothesis_gate") or {})
    check("gate_recorded", gate.get("status") in {"passed", "blocked"} and gate.get("checks", {}).get("identifiability_audit_pass") is True)

    findings = [
        {"id": "mathematical-non-identifiability", "status": "supported", "evidence": f"coarse context conditional entropy={ident.get('conditional_entropy_bits')} bits; Bayes error lower bound={ident.get('bayes_error_lower_bound')}.", "implication": "缺失观测时不能用更多 epoch/RL 生成不可见事实；必须 ASK 或保持 unresolved。"},
        {"id": "final-only-shortcut", "status": "supported" if int(final_only.get("pre_supervision_rows", -1)) == 0 else "blocked", "evidence": f"final-only post transition min={final_only.get('post_transition_accuracy_min')}，但 pre supervision rows={final_only.get('pre_supervision_rows')}。", "implication": "最终分类分数不能作为主动提问/排错能力证据。"},
        {"id": "process-ask", "status": "supported" if float(process.get("missing_ask_rate_min", 0.0) or 0.0) >= 0.90 else "blocked", "evidence": f"process ASK min={process.get('missing_ask_rate_min')}，safe unresolved min={process.get('missing_safe_non_supported_min')}。", "implication": "后续训练必须保留 pre-state→question→observation→repair 轨迹。"},
        {"id": "family-ood", "status": "blocked", "evidence": "PG-280 family-OOD hard-negative 仅作为 evaluation-only，真实 Docker 不可用。", "implication": "Docker 可用后再接真实应用 gold；当前不晋级长期记忆。"},
    ]
    audit = {
        "audit_id": "pg280-ontology-policy-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in ["dataset_hash", "dataset_audit_hash", "report_hash", "trace_hash", "protocol_hash", "dataset_audit_pass", "remote_scope", "a800_gpu0_only", "report_dataset_binding", "trace_protocol_binding", "status_namespace", "identifiability_gate", "final_only_no_pre_supervision", "process_pre_supervision_present", "process_ask_safe", "final_only_classification_visible", "hard_negative_lane", "promotion_blocked", "docker_probe_honest", "gate_recorded"]},
        "report": REPORT.relative_to(ROOT).as_posix(),
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "findings": findings,
        "failures": failures,
        "interpretation": "PG-280 的工程/数学审计通过才可进入下一实验；它证明安全 ASK 与 final-only 分类是不同能力，但不证明真实应用漏洞发现。",
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
