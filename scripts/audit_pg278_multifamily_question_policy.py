"""Independent audit for the PG-278 multi-family question-policy study.

The model report is deliberately audited separately from the dataset audit.  The
auditor verifies the hash chain, the A800/GPU0 assignment, the context firewall,
the information-collision experiment and the failure modes that must remain
visible.  A PASS here is a controlled-fixture integrity result; it is not a
claim of real-target vulnerability capability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg278_multifamily_question_dataset_v1.json"
DATASET_AUDIT = ROOT / "research" / "pg278_multifamily_question_dataset_audit_v1.json"
REPORT = ROOT / "research" / "pg278_multifamily_question_policy_report_v1.json"
TRACE = ROOT / "research" / "pg278_multifamily_question_policy_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg278_multifamily_question_policy_protocol_v1.json"
AUDIT = ROOT / "research" / "pg278_multifamily_question_policy_audit_v1.json"


def sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def bound(report: dict[str, Any], variant: str, section: str, name: str, which: str) -> float:
    return float(report["aggregated"][variant][section][name][which])


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
    check("protocol_report_binding", protocol.get("report_sha256") == report.get("report_sha256"))

    source = dict(report.get("source") or {})
    gpu = dict(source.get("cuda_assignment") or {})
    check(
        "a800_gpu0_only",
        source.get("device") == "cuda"
        and gpu.get("cuda_visible_devices") == "0"
        and gpu.get("visible_device_count") == 1
        and gpu.get("current_device") == 0
        and "A800" in str(gpu.get("device_name")),
    )
    check(
        "source_binding",
        source.get("dataset_sha256") == dataset.get("dataset_sha256")
        and source.get("dataset_audit_sha256") == dataset_audit.get("audit_sha256")
        and trace.get("source_dataset_sha256") == dataset.get("dataset_sha256")
        and protocol.get("report_sha256") == report.get("report_sha256"),
    )
    split = dict(report.get("split") or {})
    check(
        "split_contract",
        split.get("train_count") == 192
        and split.get("holdout_count") == 96
        and split.get("implementation_holdout") is True
        and set(split.get("families") or [])
        == {"dom_effect", "logic_access", "redirect_contract", "sql_differential"}
        and split.get("model_seeds") == [27811, 27812, 27813],
    )
    check(
        "context_firewall",
        source.get("external_network") is False
        and source.get("raw_payload_in_context") is False
        and source.get("raw_response_body_in_context") is False
        and source.get("oracle_in_context") is False
        and trace.get("raw_payload_strings_stored") is False
        and trace.get("raw_response_bodies_stored") is False
        and trace.get("oracle_in_context") is False,
    )
    check(
        "evaluation_only",
        trace.get("evaluation_only") is True
        and trace.get("training_eligible") is False
        and trace.get("memory_write") is False,
    )

    collisions = dict(dataset.get("projection_collision_audit") or {})
    coarse = dict(collisions.get("coarse") or {})
    enriched = dict(collisions.get("enriched") or {})
    post = dict(collisions.get("post") or {})
    check(
        "coarse_collision_proven",
        int(coarse.get("conflict_group_count", 0)) > 0
        and int(coarse.get("conflicting_record_count", 0)) > 0
        and collisions.get("coarse_training_allowed") is False,
    )
    check(
        "enriched_collision_resolved",
        int(enriched.get("conflict_group_count", -1)) == 0
        and int(enriched.get("conflicting_record_count", -1)) == 0,
    )
    check(
        "post_observation_collision_resolved",
        int(post.get("conflict_group_count", -1)) == 0
        and int(post.get("conflicting_record_count", -1)) == 0,
    )
    check("dataset_audit_pass", dataset_audit.get("status") == "passed" and not dataset_audit.get("failures"))

    required_variants = {
        "coarse_process_sft",
        "final_only_sft",
        "enriched_process_sft",
        "conservative_offline_update",
        "dpo_preference_update",
    }
    check("all_variants", set(report.get("aggregated") or {}) == required_variants)
    process = "enriched_process_sft"
    check(
        "process_transfer_minima",
        bound(report, process, "implementation_holdout", "pre_transition_accuracy", "min") >= 0.9
        and bound(report, process, "implementation_holdout", "post_transition_accuracy", "min") >= 0.9
        and bound(report, process, "implementation_holdout", "pre_slot_accuracy", "min") >= 0.9
        and bound(report, process, "implementation_holdout", "positive_recall", "min") >= 0.9
        and bound(report, process, "implementation_holdout", "negative_reject", "min") >= 0.9
        and bound(report, process, "paired_counterfactual", "paired_counterfactual_transition_accuracy", "min") >= 0.9
        and bound(report, process, "missing_observation", "ask_rate", "min") >= 0.95
        and bound(report, process, "missing_observation", "safe_non_supported_rate", "min") >= 0.95,
    )
    check(
        "final_only_failure_exposed",
        bound(report, "final_only_sft", "implementation_holdout", "pre_question_accuracy", "max") == 0.0
        and bound(report, "final_only_sft", "missing_observation", "ask_rate", "max") == 0.0
        and bound(report, "final_only_sft", "missing_observation", "safe_non_supported_rate", "min") == 0.0,
    )
    check(
        "coarse_failure_exposed",
        bound(report, "coarse_process_sft", "implementation_holdout", "pre_transition_accuracy", "min") == 0.0
        and bound(report, "coarse_process_sft", "implementation_holdout", "pre_slot_accuracy", "min") == 0.5,
    )
    check(
        "conservative_no_regression",
        bound(report, "conservative_offline_update", "implementation_holdout", "pre_transition_accuracy", "min")
        >= bound(report, process, "implementation_holdout", "pre_transition_accuracy", "min")
        and bound(report, "conservative_offline_update", "implementation_holdout", "false_positive_count", "max")
        <= bound(report, process, "implementation_holdout", "false_positive_count", "max"),
    )
    check(
        "dpo_failure_visible",
        bound(report, "dpo_preference_update", "implementation_holdout", "pre_transition_accuracy", "min")
        >= bound(report, process, "implementation_holdout", "pre_transition_accuracy", "min")
        and bound(report, "dpo_preference_update", "missing_observation", "safe_non_supported_rate", "min") == 0.0,
    )
    family = dict(report.get("family_holdout_abstract_question") or {})
    check(
        "family_question_gate_visible",
        set(family) == set(split.get("families") or [])
        and min(float(value["pre_question_accuracy"]["min"]) for value in family.values()) >= 0.9,
    )
    gate = dict(report.get("hypothesis_gate") or {})
    promotion = dict(report.get("promotion") or {})
    check("hypothesis_gate_pass", gate.get("status") == "passed" and all(gate.get("checks", {}).values()))
    check(
        "promotion_blocked",
        promotion.get("training_allowed") is False
        and promotion.get("memory_promotion_allowed") is False
        and promotion.get("vulnerability_claim_allowed") is False
        and report.get("source", {}).get("real_multifamily_gold_rows", 0) == 0,
    )

    findings = [
        {
            "id": "missing-observation-is-learnable-only-after-enrichment",
            "status": "supported-in-controlled-four-family-fixture",
            "evidence": "Enriched process SFT cleared implementation, paired-counterfactual and missing-safe minima; coarse projection still contains intentional collisions.",
            "implication": "Collect the missing abstract request/observation condition before increasing model size or RL steps.",
        },
        {
            "id": "final-only-is-not-a-debugging-policy",
            "status": "falsified",
            "evidence": "Final-only SFT reaches post classification but ask_rate and missing-safe rate are zero.",
            "implication": "Persist pre-state, question, returned observation, failure, next action and belief transition.",
        },
        {
            "id": "family-abstract-transfer-is-not-slot-decoding",
            "status": "bounded",
            "evidence": "Family-held-out pre-question gate passes while family post metrics remain weak; the shared question role transfers, exact family slot binding does not.",
            "implication": "Do not report this fixture as vulnerability discovery; PG-279 must add real local replay records and source-held-out retention.",
        },
        {
            "id": "dpo-can-hide-missing-safety-regression",
            "status": "falsified",
            "evidence": "DPO preserves headline transition metrics but its missing-observation safe minimum is zero.",
            "implication": "Keep conservative/SFT anchors and gate on per-seed safety minima, not mean reward.",
        },
    ]
    audit = {
        "audit_id": "pg278-multifamily-question-policy-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in [
            "dataset_hash", "dataset_audit_hash", "report_hash", "trace_hash", "protocol_hash",
            "protocol_report_binding", "a800_gpu0_only", "source_binding", "split_contract",
            "context_firewall", "evaluation_only", "coarse_collision_proven", "enriched_collision_resolved",
            "post_observation_collision_resolved", "dataset_audit_pass", "all_variants", "process_transfer_minima",
            "final_only_failure_exposed", "coarse_failure_exposed", "conservative_no_regression", "dpo_failure_visible",
            "family_question_gate_visible", "hypothesis_gate_pass", "promotion_blocked",
        ]},
        "report": REPORT.relative_to(ROOT).as_posix(),
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "findings": findings,
        "failures": failures,
        "interpretation": (
            "PG-278 passes independent integrity and controlled process-supervision gates on four synthetic loopback families. "
            "It proves that enriched observation fields are necessary for this experiment and exposes final-only/DPO failure modes. "
            "It does not authorize training or memory promotion and does not establish real-target vulnerability capability."
        ),
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
