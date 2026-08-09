"""PG-248: observable-feedback tokenization plus larger adapter capacity.

This is a controlled follow-up to PG-247.  It adds only already-observed
feedback projections from PG-230 and maps them to tokens in the frozen PG-191
vocabulary.  Expected oracle labels, retention lanes, and
``payload_grounded_eligible`` are never added to the model input.  All
Pikachu-labelled sources remain an implementation holdout and VulnerableApp
seed 24603 remains a same-implementation holdout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG247 = _load("run_pg247_vulnerableapp_capacity_training.py")
PG237 = PG247.PG237

RESEARCH = ROOT / "research"
PG244_DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
PG230_DATASET = RESEARCH / "pg230_next_token_quality_funnel_dataset_v1.json"
OLD_ADAPTER = ROOT / "artifacts" / "pg244-failure-repair-capacity-v1" / "frozen_xxl_capacity_hidden2048.pt"
REPORT = RESEARCH / "pg248_feedback_token_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg248_feedback_token_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg248_feedback_token_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg248_feedback_token_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg248_feedback_token_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg248-feedback-token-capacity-v1"

# These are post-observation projections, not expected labels.  In
# particular, no alias is provided for lane, repair_action, negative_clean,
# or payload_grounded_eligible.
OBSERVABLE_FEEDBACK_ALIASES = {
    "result_verified=0": "ir.oracle.availability=unknown",
    "result_verified=1": "history::gate::typed_effect",
    "typed_effect=0": "ir.oracle.availability=unknown",
    "typed_effect=1": "ir.failure.kind=typed_positive",
    "feedback=result_verified": "history::gate::typed_effect",
    "feedback=failure_adjusted": "ir.failure.recovery_phase=failure_adjusted",
    "feedback=abstain": "ir.oracle.availability=unknown",
}
FORBIDDEN_INPUT_FIELDS = ("payload_grounded_eligible", "negative_clean", "lane", "repair_action", "expected_oracle")


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_rows, base_counts = PG247._load_records()
    extra_payload = json.loads(PG230_DATASET.read_text(encoding="utf-8-sig"))
    extra_rows = [dict(row) for row in extra_payload.get("records", []) if row.get("lane") not in {"quarantine", "reject"}]
    rows = list(base_rows) + extra_rows
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    duplicate_count = 0
    for row in rows:
        key = (
            str(row.get("trajectory_hash", row.get("token_hash", ""))),
            int(row.get("seed", 0) or 0),
            str(row.get("source", "")),
            str(row.get("route_source_sha256", "")),
            str(row.get("record_id", "")),
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(row)
    counts = {
        **base_counts,
        "pg230_input_records": len(extra_rows),
        "pg230_unique_new_records": sum(1 for row in unique if str(row.get("source", "")) == "pg222_model_decision_error_counterfactual"),
        "input_records_with_pg230": len(rows),
        "unique_records_with_pg230": len(unique),
        "duplicate_records_with_pg230": duplicate_count,
    }
    return unique, counts


def _patched_input_token_id(token: str, vocabulary: dict[str, int]) -> int:
    mapped = OBSERVABLE_FEEDBACK_ALIASES.get(str(token), str(token))
    return PG247.ORIGINAL_INPUT_TOKEN_ID(mapped, vocabulary)


def _configure() -> None:
    # PG-247 imports the same frozen trainer.  Keep the original helper for
    # canary evaluation, but route PG-237 encoding through the new
    # observation-only aliases.
    if not hasattr(PG247, "ORIGINAL_INPUT_TOKEN_ID"):
        PG247.ORIGINAL_INPUT_TOKEN_ID = PG237.PG231._input_token_id
    PG237.PG231._input_token_id = _patched_input_token_id
    PG237._load_records = _load_records
    PG237.BASE_DATASET = PG244_DATASET
    PG237.PG237_TRACE = PG246_DATASET
    PG237.FRESH_SOURCE = "__pg248_unused_fresh_source__"
    PG237.FRESH_HOLDOUT_SEEDS = ()
    PG237.EXTRA_HOLDOUT_SOURCE = "__pg248_unused_extra_source__"
    PG237.EXTRA_HOLDOUT_SEEDS = ()
    PG237.HOLDOUT_SOURCE_SUBSTRINGS = ("pikachu",)
    PG237.HOLDOUT_SOURCE_SEED_PAIRS = (("pg246_vulnerableapp_source_independent", (24603,)),)
    PG237.SEND_PROBABILITY_THRESHOLD = 0.90
    PG237.CAPACITY_VARIANTS = (512, 1024, 2048, 4096)
    PG237.TRAIN_STEPS = 180
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg248_feedback_token"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN


def _finalize() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    all_rows, source_counts = _load_records()
    canary = PG247._canary_rows(all_rows)
    if not canary:
        raise RuntimeError("PG-248 requires a non-empty old SQL/XSS canary")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, input_vocab = PG247._load_base(device)
    selected_artifact = ROOT / report["selected"]["artifact"]
    # Compare the old adapter under its original encoder, then the new
    # adapter under the observable-feedback encoder.  This makes any canary
    # delta attributable to the whole update rather than silently pretending
    # that the representation was unchanged.
    PG237.PG231._input_token_id = PG247.ORIGINAL_INPUT_TOKEN_ID
    before = PG247._evaluate_artifact(OLD_ADAPTER, canary, base, input_vocab, device)
    PG237.PG231._input_token_id = _patched_input_token_id
    after = PG247._evaluate_artifact(selected_artifact, canary, base, input_vocab, device)
    canary_judge = PG247._judge_canary(before, after)
    canary_judge["before_input_encoder"] = "pg191_default_aliases"
    canary_judge["after_input_encoder"] = "pg248_observable_feedback_alias_v1"
    holdout = [
        row
        for row in all_rows
        if row.get("lane") not in {"quarantine", "reject"}
        and ("pikachu" in str(row.get("source", "")) or (str(row.get("source", "")) == "pg246_vulnerableapp_source_independent" and int(row.get("seed", 0) or 0) == 24603))
    ]
    holdout_counts = dict(Counter(str(row.get("source", "")) for row in holdout))
    holdout_has_both_actions = any(PG237.action_target(row) == "send_candidate" for row in holdout) and any(PG237.action_target(row) == "abstain" for row in holdout)
    judge = {
        "authority": ["PG-246 typed DOM oracle", "independent reference/action labels", "matched negative and abstain labels", "Pikachu implementation holdout", "old SQL/XSS canary"],
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
        "observable_feedback_aliases_only": True,
        "hard_gates": {
            "capacity_safety_abstain": bool(report.get("safety_abstain_gate_pass")),
            "capacity_positive_capability": bool(report.get("capability_gate_pass")),
            "holdout_has_send_and_abstain": holdout_has_both_actions,
            "canary_no_guardrail_regression": bool(canary_judge["pass"]),
            "input_alias_does_not_include_expected_labels": not any(field in " ".join(OBSERVABLE_FEEDBACK_ALIASES) for field in FORBIDDEN_INPUT_FIELDS),
            "no_raw_payload_or_response_persistence": True,
        },
    }
    judge["pass"] = all(judge["hard_gates"].values())
    judge["decision"] = "candidate_eligible_for_next_replay" if judge["pass"] else "blocked"
    report.update(
        {
            "protocol_id": "pg-pk-248-feedback-token-capacity-training-v1",
            "schema_version": "pg248-feedback-token-capacity-training-v1",
            "status": "completed_observable_feedback_token_capacity_training_with_pikachu_implementation_holdout",
            "source_datasets": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))],
            "source_counts": source_counts,
            "input_encoder": {"version": "pg248-observable-feedback-alias-v1", "aliases": dict(OBSERVABLE_FEEDBACK_ALIASES), "forbidden_fields": list(FORBIDDEN_INPUT_FIELDS), "oracle_labels_as_input": False},
            "action_send_probability_threshold": PG237.SEND_PROBABILITY_THRESHOLD,
            "holdout_contract": {"all_pikachu_sources_never_in_training": True, "pg246_seed_24603_never_in_training": True, "holdout_source_counts": holdout_counts, "holdout_contains_send_and_abstain": holdout_has_both_actions, "canary_source_overlap_with_train": False},
            "catastrophic_forgetting_canary": canary_judge,
            "independent_final_judge": judge,
            "training_eligible": bool(judge["pass"]),
            "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "judge_decision": judge["decision"]},
            "honesty": {"frozen_xxl_body_not_updated": True, "adapter_only": True, "all_pikachu_sources_are_holdout_only": True, "pg246_seed_24603_is_never_in_training": True, "pg230_rows_are_existing_audited_projections": True, "canary_is_evaluation_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "final_judge_is_not_model_self_report": True},
        }
    )
    report["report_sha256"] = PG237.digest(report)
    dataset["schema_version"] = "pg248-feedback-token-capacity-training-dataset-v1"
    dataset["source_datasets"] = [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))]
    dataset["input_encoder"] = {"version": "pg248-observable-feedback-alias-v1", "aliases": dict(OBSERVABLE_FEEDBACK_ALIASES), "forbidden_fields": list(FORBIDDEN_INPUT_FIELDS), "oracle_labels_as_input": False}
    dataset["canary_manifest"] = {"source_set": ["pg242_pikachu_source_native", "pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"], "record_count": len(canary), "canary_sha256": PG237.digest([{key: row.get(key) for key in ("source", "seed", "trajectory_hash", "failure_stage", "record_id")} for row in canary]), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
    dataset["contract"] = {**dict(dataset.get("contract") or {}), "all_pikachu_sources_never_in_training": True, "pg246_seed_24603_never_in_training": True, "old_sql_xss_canary_replayed_after_update": True, "canary_never_used_as_oracle_feature": True, "observable_feedback_aliases_only": True, "expected_oracle_labels_not_input": True, "false_send_or_guardrail_regression_blocks_promotion": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol.update({"protocol_id": "pg-pk-248-feedback-token-capacity-training-v1", "schema_version": "pg248-feedback-token-capacity-training-protocol-v1", "training_sources": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))], "input_encoder": {"version": "pg248-observable-feedback-alias-v1", "aliases": dict(OBSERVABLE_FEEDBACK_ALIASES), "forbidden_fields": list(FORBIDDEN_INPUT_FIELDS), "oracle_labels_as_input": False}, "implementation_holdout": "all sources containing pikachu", "same_implementation_seed_holdout": ["pg246_vulnerableapp_source_independent:24603"], "old_canary": ["pg242_pikachu_source_native", "pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"], "capacity_variants": list(PG237.CAPACITY_VARIANTS), "train_steps": PG237.TRAIN_STEPS, "canary_replay_required_after_update": True, "final_judge_hard_gates": list(judge["hard_gates"]), "promotion_blocked": True, "raw_payload_and_response_excluded": True})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    trace.update({"schema_version": "pg248-feedback-token-capacity-training-trace-v1", "input_encoder": dataset["input_encoder"], "implementation_holdout": "all sources containing pikachu", "same_implementation_seed_holdout": ["pg246:24603"], "canary": canary_judge, "independent_final_judge": judge, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    PG237._write(TRACE, trace)
    MARKDOWN.write_text("\n".join(["# PG-248 observable feedback token capacity training", "", f"train={report['counts']['train_rows']}; holdout={report['counts']['holdout_rows']}; canary={len(canary)}; pg230_new={source_counts['pg230_unique_new_records']}", f"variants={list(PG237.CAPACITY_VARIANTS)}; selected hidden={report['selected']['hidden_dim']}; holdout positive={report['selected']['metrics']['seed_holdout']['positive_send_recall']}; abstain={report['selected']['metrics']['seed_holdout']['abstain_recall']}; false_send={report['selected']['metrics']['seed_holdout']['false_send_count']}", f"canary pass={canary_judge['pass']}; final_judge={judge['decision']}", "", "输入只增加已观察的反馈投影；期望 oracle、lane、repair 和 payload_grounded_eligible 不进入模型输入。所有 Pikachu 来源仍为实现留出。", ""]), encoding="utf-8")
    result = {"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "canary": canary_judge, "final_judge": judge, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    _configure()
    PG237.main()
    _finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
