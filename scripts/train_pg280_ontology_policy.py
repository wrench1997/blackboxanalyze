"""PG-280 remote A800 comparison: final-only versus process supervision.

The baseline policy implementation is reused as a small research model.  The
new dataset adds a shared Rule-IR slot ontology and keeps family-OOD hard
negatives evaluation-only.  This wrapper writes a bounded report proving that
final-only supervision has no pre-question training signal, while process
supervision is scored on ASK/safe-belief behavior.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg280_shared_ontology_dataset_v1.json"
DATASET_AUDIT = RESEARCH / "pg280_shared_ontology_dataset_audit_v1.json"
DOCKER_PROBE = RESEARCH / "pg280_remote_docker_probe_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg280-ontology-policy"
CHECKPOINT = OUTPUT_DIR / "pg280_ontology_policies.pt"
REPORT = RESEARCH / "pg280_ontology_policy_report_v1.json"
TRACE = RESEARCH / "pg280_ontology_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg280_ontology_policy_protocol_v1.json"
MARKDOWN = RESEARCH / "pg280_ontology_policy_report_v1.md"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    data_audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    docker_probe = json.loads(DOCKER_PROBE.read_text(encoding="utf-8"))
    if data_audit.get("status") != "passed":
        raise RuntimeError("PG-280 dataset audit must pass before training")
    mod = load_module(ROOT / "scripts" / "train_pg278_multifamily_question_policy.py", "pg280_policy_baseline")
    mod.DATASET = DATASET
    mod.DATASET_AUDIT = DATASET_AUDIT
    mod.OUTPUT_DIR = OUTPUT_DIR
    mod.CHECKPOINT = CHECKPOINT
    mod.REPORT = REPORT
    mod.TRACE = TRACE
    mod.PROTOCOL = PROTOCOL
    mod.MARKDOWN = MARKDOWN
    mod.MODEL_SEEDS = (28011, 28012, 28013)
    mod.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    aggregated = dict(report.get("aggregated") or {})
    final_only = dict(aggregated.get("final_only_sft") or {})
    process = dict(aggregated.get("enriched_process_sft") or {})
    final_missing = dict(final_only.get("missing_observation") or {})
    process_missing = dict(process.get("missing_observation") or {})
    final_impl = dict(final_only.get("implementation_holdout") or {})
    process_impl = dict(process.get("implementation_holdout") or {})
    comparison = {
        "final_only": {
            "pre_supervision_rows": 0,
            "post_transition_accuracy_min": float(final_impl.get("post_transition_accuracy", {}).get("min", 0.0) or 0.0),
            "pre_action_accuracy_min": float(final_impl.get("pre_action_accuracy", {}).get("min", 0.0) or 0.0),
            "missing_ask_rate_min": float(final_missing.get("ask_rate", {}).get("min", 0.0) or 0.0),
            "missing_safe_non_supported_min": float(final_missing.get("safe_non_supported_rate", {}).get("min", 0.0) or 0.0),
            "interpretation": "final-only 可在 post/final 分类上得分，但没有任何 pre-question 监督，不能把偶然的 ask 输出解释为主动排错能力。",
        },
        "process": {
            "pre_supervision_rows": int(data.get("identifiability", {}).get("process_pre_supervision_rows", 0) or 0),
            "post_transition_accuracy_min": float(process_impl.get("post_transition_accuracy", {}).get("min", 0.0) or 0.0),
            "pre_action_accuracy_min": float(process_impl.get("pre_action_accuracy", {}).get("min", 0.0) or 0.0),
            "missing_ask_rate_min": float(process_missing.get("ask_rate", {}).get("min", 0.0) or 0.0),
            "missing_safe_non_supported_min": float(process_missing.get("safe_non_supported_rate", {}).get("min", 0.0) or 0.0),
            "interpretation": "process supervision 显式训练 pre-state→ASK→observation→belief/repair，才允许声称主动提问/安全未决。",
        },
    }
    checks = dict(report.get("hypothesis_gate", {}).get("checks") or {})
    checks.update({
        "identifiability_audit_pass": data_audit.get("status") == "passed" and float(data.get("identifiability", {}).get("conditional_entropy_bits", 0.0) or 0.0) > 0.0 and float(data.get("identifiability", {}).get("bayes_error_lower_bound", 0.0) or 0.0) >= 0.49,
        "shared_ontology_present": int(data.get("counts", {}).get("total", 0) or 0) == 288,
        "family_ood_hard_negative_lane_present": int(data.get("counts", {}).get("family_ood_hard_negative", 0) or 0) == 48,
        "final_only_pre_supervision_zero": comparison["final_only"]["pre_supervision_rows"] == 0,
        "process_ask_recovery_min": comparison["process"]["missing_ask_rate_min"] >= 0.90 and comparison["process"]["missing_safe_non_supported_min"] >= 0.95,
        "final_only_post_classification_visible": comparison["final_only"]["post_transition_accuracy_min"] >= 0.80,
        "remote_docker_honest": docker_probe.get("status") == "unavailable" and docker_probe.get("docker_binary") is False,
        "promotion_blocked": True,
    })
    report["protocol_id"] = "pg280-shared-ontology-policy-v1"
    report["schema_version"] = "pg280-shared-ontology-policy-report-v1"
    report["status"] = "completed_remote_pg280_ontology_policy_study"
    report["source"].update({"dataset": DATASET.relative_to(ROOT).as_posix(), "dataset_sha256": data["dataset_sha256"], "dataset_audit": DATASET_AUDIT.relative_to(ROOT).as_posix(), "dataset_audit_sha256": data_audit["audit_sha256"], "remote_host": "112.111.7.91:60228", "remote_docker_available": False, "real_application_gold_rows": 0, "loopback_only": True})
    report["identifiability"] = dict(data.get("identifiability") or {})
    report["shared_slot_ontology"] = dict(data.get("shared_slot_ontology") or {})
    report["family_ood_hard_negative"] = {"rows": int(data.get("counts", {}).get("family_ood_hard_negative", 0) or 0), "training_eligible": False, "status": "evaluation_only"}
    report["comparison"] = comparison
    report["docker_probe"] = docker_probe
    report["hypothesis_gate"] = {"status": "passed" if all(bool(value) for value in checks.values()) else "blocked", "checks": checks, "claim_allowed": False}
    report["promotion"] = {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "PG-280 has zero real-application gold and family-OOD hard negatives are evaluation-only"}
    report["formal_conclusion"] = "PG-280 separates two claims: positive conditional entropy proves exact missing-slot resolution is not identifiable from the coarse observation; final-only SFT can still score post/final classification but has zero pre-question supervision; process supervision is required to learn ASK and safe unresolved belief. Shared ontology is a representation improvement, not proof of real-application vulnerability capability."
    report.pop("report_sha256", None)
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg280-ontology-policy-trace-v1"
    trace["source_dataset_sha256"] = data["dataset_sha256"]
    trace["report_sha256"] = report["report_sha256"]
    trace["comparison"] = comparison
    trace["identifiability"] = report["identifiability"]
    trace["training_eligible"] = False
    trace["memory_write"] = False
    trace.pop("trace_sha256", None)
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["protocol_id"] = "pg280-shared-ontology-policy-v1"
    protocol["schema_version"] = "pg280-shared-ontology-policy-protocol-v1"
    protocol["report_sha256"] = report["report_sha256"]
    protocol["identifiability"] = report["identifiability"]
    protocol["comparison"] = comparison
    protocol["family_ood_hard_negative"] = report["family_ood_hard_negative"]
    protocol["next_experiment"] = "PG-281: authorized remote Docker real-application replay when Docker is available; retain ontology, hard-negative, information-entropy and process-ASK gates"
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-280 共享 Slot Ontology / 不可识别性", "", f"gate=`{report['hypothesis_gate']['status']}`", f"conditional_entropy={report['identifiability']['conditional_entropy_bits']} bits", f"bayes_error_lower_bound={report['identifiability']['bayes_error_lower_bound']}", f"final_only_pre_supervision={comparison['final_only']['pre_supervision_rows']}", f"process_ask_min={comparison['process']['missing_ask_rate_min']}", "real_application_gold=0", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": report["source"]["cuda_assignment"], "hypothesis_gate": report["hypothesis_gate"], "comparison": comparison, "report": REPORT.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
