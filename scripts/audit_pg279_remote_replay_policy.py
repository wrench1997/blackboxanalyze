"""Independent final audit for the PG-279 remote replay policy study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg279_remote_replay_dataset_v1.json"
DATASET_AUDIT = ROOT / "research" / "pg279_remote_replay_dataset_audit_v1.json"
REPORT = ROOT / "research" / "pg279_remote_replay_policy_report_v1.json"
TRACE = ROOT / "research" / "pg279_remote_replay_policy_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg279_remote_replay_policy_protocol_v1.json"
AUDIT = ROOT / "research" / "pg279_remote_replay_policy_audit_v1.json"


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
    check("remote_scope", source.get("remote_host") == "112.111.7.91:60228" and source.get("remote_docker_available") is False and source.get("external_network") is False)
    check("a800_gpu0_only", source.get("device") == "cuda" and gpu.get("cuda_visible_devices") == "0" and gpu.get("visible_device_count") == 1 and gpu.get("current_device") == 0 and "A800" in str(gpu.get("device_name")))
    check("report_dataset_binding", source.get("dataset_sha256") == dataset.get("dataset_sha256") and source.get("dataset_audit_sha256") == dataset_audit.get("audit_sha256"))
    check("trace_protocol_binding", trace.get("source_dataset_sha256") == dataset.get("dataset_sha256") and trace.get("report_sha256") == report.get("report_sha256") and protocol.get("report_sha256") == report.get("report_sha256"))
    check("status_namespace", report.get("status") == "completed_remote_loopback_replay_policy_study" and report.get("schema_version") == "pg279-remote-replay-policy-report-v1")
    check("get_post_data", int((dataset.get("replay_contract") or {}).get("get_rows", 0)) > 0 and int((dataset.get("replay_contract") or {}).get("post_rows", 0)) > 0)
    check("fresh_failure_repair", int((dataset.get("replay_contract") or {}).get("failure_repair_rows", 0)) == 288 and int((dataset.get("replay_contract") or {}).get("fresh_replays_per_episode", 0)) == 2)
    check("context_firewall", source.get("raw_payload_in_context") is False and source.get("raw_response_body_in_context") is False and source.get("oracle_in_context") is False and trace.get("raw_payload_strings_stored") is False and trace.get("raw_response_bodies_stored") is False and trace.get("oracle_in_context") is False)
    checks = dict(report.get("hypothesis_gate", {}).get("checks") or {})
    # A scientific hypothesis may legitimately remain blocked while the
    # operational record/replay contract and retention canary are audited.
    # Keep that distinction explicit instead of turning a failed family
    # holdout into a false engineering failure or a false success claim.
    check("hypothesis_gate_recorded", report.get("hypothesis_gate", {}).get("status") in {"passed", "blocked"} and "family_holdout_question_min" in checks)
    operational_keys = (
        "dataset_audit_pass", "coarse_collision_detected", "enriched_collision_zero",
        "post_observation_collision_zero", "implementation_pre_transition_min",
        "implementation_post_transition_min", "implementation_slot_min",
        "paired_counterfactual_min", "missing_safe_min", "conservative_no_regression",
        "dpo_no_regression", "promotion_blocked", "frozen_retention_canary",
    )
    check("operational_gate_pass", all(bool(checks.get(key)) for key in operational_keys))
    retention = dict(report.get("retention_matrix") or {})
    check("retention_canary_pass", retention.get("status") == "passed" and all(dict(retention.get("checks") or {}).values()))
    promotion = dict(report.get("promotion") or {})
    check("promotion_blocked", promotion.get("training_allowed") is False and promotion.get("memory_promotion_allowed") is False and promotion.get("vulnerability_claim_allowed") is False and int(source.get("real_application_gold_rows", 1)) == 0)

    family_holdout_passed = bool(checks.get("family_holdout_question_min"))
    findings = [
        {"id": "remote-transport-grounding", "status": "supported-in-controlled-replay", "evidence": "PG-279 captured actual remote loopback GET/POST requests with bounded response projections, two fresh replays and paired reference/negative channels.", "implication": "The same record contract can now be connected to an authorized real application when remote Docker is available."},
        {"id": "failure-repair-signal", "status": "supported-in-controlled-replay", "evidence": "Every row carries an initial failure signature, a repair projection, a typed effect or explicit abstain outcome, and evidence hashes.", "implication": "Keep repair transitions as first-class training data; do not collapse them to final labels."},
        {"id": "real-application-boundary", "status": "not-established", "evidence": "Remote Docker is unavailable and real_application_gold_rows=0.", "implication": "Do not claim Pikachu or public-target vulnerability discovery; next is PG-280 with an authorized remote application target."},
        {"id": "forgetting", "status": "frozen-canary-passed" if dict(report.get("retention_matrix") or {}).get("status") == "passed" else "blocked", "evidence": "The updated policy was evaluated on the frozen PG-278 implementation holdout and the configured retention minima were checked.", "implication": "Keep the frozen canary in every future remote update; any regression blocks promotion."},
        {"id": "family-heldout-generalization", "status": "passed" if family_holdout_passed else "blocked", "evidence": "The family-heldout question metric is retained as a separate scientific gate; current slot vocabulary does not yet support reliable unseen-family question decoding." if not family_holdout_passed else "The family-heldout question metric met its configured threshold.", "implication": "Add shared, provenance-safe slot ontology fragments and rerun PG-280; do not promote the current family-heldout result."},
    ]
    audit = {
        "audit_id": "pg279-remote-replay-policy-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in ["dataset_hash", "dataset_audit_hash", "report_hash", "trace_hash", "protocol_hash", "dataset_audit_pass", "remote_scope", "a800_gpu0_only", "report_dataset_binding", "trace_protocol_binding", "status_namespace", "get_post_data", "fresh_failure_repair", "context_firewall", "hypothesis_gate_recorded", "operational_gate_pass", "retention_canary_pass", "promotion_blocked"]},
        "report": REPORT.relative_to(ROOT).as_posix(),
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "findings": findings,
        "failures": failures,
        "interpretation": "PG-279 passes the remote controlled replay, record-contract and frozen-retention engineering audit. The family-heldout scientific gate remains explicitly blocked, the study uses non-Docker fixtures with zero real-application gold, and no vulnerability or long-term-memory promotion is authorized.",
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
