"""Independent audit for PG-277 question-driven composition ablation.

This auditor intentionally separates artifact integrity from capability claims.
Expected model failures stay visible as observations instead of being hidden by
an overall structural PASS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg277_counterfactual_question_dataset_v1.json"
DATASET_AUDIT = ROOT / "research" / "pg277_counterfactual_question_dataset_audit_v1.json"
REPORT = ROOT / "research" / "pg277_question_composition_report_v1.json"
TRACE = ROOT / "research" / "pg277_question_composition_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg277_question_composition_protocol_v1.json"
AUDIT = ROOT / "research" / "pg277_question_composition_audit_v1.json"


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


def metric(report: dict[str, Any], variant: str, section: str, name: str, bound: str) -> float:
    return float(report["aggregated"][variant][section][name][bound])


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
    check("report_protocol_binding", protocol.get("report_sha256") == report.get("report_sha256"))

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
        "dataset_binding",
        source.get("dataset_sha256") == dataset.get("dataset_sha256")
        and trace.get("source_dataset_sha256") == dataset.get("dataset_sha256"),
    )
    check(
        "split_contract",
        report.get("split", {}).get("variant_and_seed_disjoint") is True
        and report.get("split", {}).get("train_variants") == ["alpha", "beta"]
        and report.get("split", {}).get("holdout_variant") == "gamma"
        and report.get("split", {}).get("model_seeds") == [27711, 27712, 27713]
        and dataset.get("split_contract", {}).get("variant_and_seed_disjoint") is True,
    )
    check(
        "context_firewall",
        source.get("raw_payload_in_context") is False
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
    check(
        "coarse_collision_proven",
        coarse.get("conflict_group_count") == 4
        and coarse.get("conflicting_record_count") == 36
        and collisions.get("coarse_training_allowed") is False,
    )
    check(
        "enriched_collision_resolved",
        enriched.get("conflict_group_count") == 0
        and enriched.get("conflicting_record_count") == 0
        and collisions.get("enriched_training_allowed") is True,
    )
    check("dataset_audit_pass", dataset_audit.get("status") == "passed" and not dataset_audit.get("failures"))

    required_variants = {
        "coarse_process_sft",
        "enriched_final_only_sft",
        "enriched_process_sft",
        "conservative_offline_update",
        "dpo_preference_update",
    }
    check("all_variants", set(report.get("aggregated") or {}) == required_variants)
    check(
        "coarse_representation_failure_exposed",
        metric(report, "coarse_process_sft", "holdout", "positive_recall", "max") == 0.0
        and metric(report, "coarse_process_sft", "holdout", "negative_reject", "min") == 1.0
        and metric(report, "coarse_process_sft", "holdout", "false_negative_count", "min") == 4.0,
    )
    check(
        "final_only_failure_exposed",
        metric(report, "enriched_final_only_sft", "holdout", "pre_question_accuracy", "max") == 0.0
        and metric(report, "enriched_final_only_sft", "missing_observation", "ask_recovery_rate", "max") == 0.0
        and metric(report, "enriched_final_only_sft", "missing_observation", "safe_non_positive_rate", "min") == 0.0,
    )
    check(
        "process_supervision_transfer",
        metric(report, "enriched_process_sft", "holdout", "positive_recall", "min") == 1.0
        and metric(report, "enriched_process_sft", "holdout", "negative_reject", "min") == 1.0
        and metric(report, "enriched_process_sft", "counterfactual", "belief_flip_accuracy", "min") == 1.0
        and metric(report, "enriched_process_sft", "missing_observation", "ask_recovery_rate", "min") == 1.0,
    )
    check(
        "conservative_update_stable",
        metric(report, "conservative_offline_update", "holdout", "positive_recall", "min") == 1.0
        and metric(report, "conservative_offline_update", "holdout", "false_positive_count", "max") == 0.0
        and metric(report, "conservative_offline_update", "missing_observation", "question_recovery_rate", "min") == 1.0,
    )
    check(
        "dpo_instability_not_hidden",
        metric(report, "dpo_preference_update", "holdout", "positive_recall", "min") == 1.0
        and metric(report, "dpo_preference_update", "missing_observation", "question_recovery_rate", "min") == 0.0
        and metric(report, "dpo_preference_update", "missing_observation", "question_recovery_rate", "max") == 1.0,
    )
    promotion = dict(report.get("promotion") or {})
    check(
        "promotion_blocked",
        promotion.get("training_allowed") is False
        and promotion.get("memory_promotion_allowed") is False
        and promotion.get("vulnerability_claim_allowed") is False
        and report.get("hypothesis_gate", {}).get("claim_allowed") is False,
    )

    findings = [
        {
            "id": "representation-collision",
            "status": "falsified-coarse-sufficiency",
            "evidence": "36 records in 4 identical-input/conflicting-label groups; coarse positive recall stayed 0%.",
            "implication": "Do not add epochs or RL to an information-colliding projection; collect the missing observation channel first.",
        },
        {
            "id": "final-label-only",
            "status": "falsified-active-recovery",
            "evidence": "Final-only SFT classified complete holdout inputs at 100% but pre-question and ask-recovery were 0%; missing-safety minimum was 0%.",
            "implication": "Store pre-question, observation, failure, next-question, next-action and belief transitions, not only the final label.",
        },
        {
            "id": "process-supervision",
            "status": "supported-in-controlled-fixture",
            "evidence": "Process SFT reached 100% holdout recall/reject and recovered ASK on all missing-observation cases, but exact question recovery varied by model seed.",
            "implication": "Question identity needs more counterfactual and missing-field support before promotion.",
        },
        {
            "id": "reward-update",
            "status": "conservative-supported-dpo-unstable",
            "evidence": "Conservative update kept exact question recovery at 100% across three seeds; DPO ranged from 0% to 100% on that metric.",
            "implication": "Keep SFT/KL anchors and evaluate per-seed minima; a mean score can hide a failed policy seed.",
        },
    ]
    audit = {
        "audit_id": "pg277-question-composition-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {
            "hash_chain": not any(name.endswith("hash") for name in failures),
            "report_protocol_binding": "report_protocol_binding" not in failures,
            "a800_gpu0_only": "a800_gpu0_only" not in failures,
            "dataset_binding": "dataset_binding" not in failures,
            "split_contract": "split_contract" not in failures,
            "context_firewall": "context_firewall" not in failures,
            "evaluation_only": "evaluation_only" not in failures,
            "coarse_collision_proven": "coarse_collision_proven" not in failures,
            "enriched_collision_resolved": "enriched_collision_resolved" not in failures,
            "dataset_audit_pass": "dataset_audit_pass" not in failures,
            "all_variants": "all_variants" not in failures,
            "coarse_representation_failure_exposed": "coarse_representation_failure_exposed" not in failures,
            "final_only_failure_exposed": "final_only_failure_exposed" not in failures,
            "process_supervision_transfer": "process_supervision_transfer" not in failures,
            "conservative_update_stable": "conservative_update_stable" not in failures,
            "dpo_instability_not_hidden": "dpo_instability_not_hidden" not in failures,
            "promotion_blocked": "promotion_blocked" not in failures,
        },
        "report": str(REPORT.relative_to(ROOT)),
        "findings": findings,
        "failures": failures,
        "interpretation": (
            "PG-277 structurally supports the representation and process-supervision hypotheses in one controlled family. "
            "It also exposes final-only and DPO failure modes. The result does not prove real multi-family vulnerability capability; "
            "the next dataset must contain independent implementations, missing-observation recovery and real replay-backed typed oracles."
        ),
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
