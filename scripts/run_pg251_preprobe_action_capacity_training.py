"""PG-251: train the action head on causal pre-probe prefixes.

PG-250 proved that a post-result action policy cannot be called before the
first candidate: it abstained on every live route.  PG-251 derives a causal
prefix from each audited trajectory, keeps the eventual reference outcome as
an off-input action target, and evaluates the same prefix online.
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


PG249 = _load("run_pg249_pikachu_route_seed_capacity_training.py")
PG248 = PG249.PG248
PG237 = PG249.PG237
from app.pg251_preprobe_action import SCHEMA_VERSION as PREPROBE_SCHEMA, build_preprobe_rows  # noqa: E402


RESEARCH = ROOT / "research"
PG244_DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
PG230_DATASET = RESEARCH / "pg230_next_token_quality_funnel_dataset_v1.json"
OLD_ADAPTER = ROOT / "artifacts" / "pg244-failure-repair-capacity-v1" / "frozen_xxl_capacity_hidden2048.pt"
REPORT = RESEARCH / "pg251_preprobe_action_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg251_preprobe_action_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg251_preprobe_action_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg251_preprobe_action_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg251_preprobe_action_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg251-preprobe-action-capacity-v1"


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_rows, base_counts = PG249._load_records()
    preprobe = build_preprobe_rows(base_rows)
    rows = list(base_rows) + preprobe
    return rows, {**base_counts, "preprobe_schema": PREPROBE_SCHEMA, "preprobe_records": len(preprobe), "combined_input_records": len(rows)}


def _pika_holdout(row: dict[str, Any]) -> bool:
    source = str(row.get("split_source", row.get("source", "")))
    seed = int(row.get("seed", 0) or 0)
    return (source == "pg242_pikachu_source_native" and seed == 24202) or (source in {"pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"} and seed == 24402)


def _vapp(row: dict[str, Any]) -> bool:
    return str(row.get("split_source", row.get("source", ""))) == "pg246_vulnerableapp_source_independent"


def _preprobe(row: dict[str, Any]) -> bool:
    return str(row.get("record_role", "")) == "preprobe_action"


def _eligible(rows: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("lane") not in {"quarantine", "reject"} and predicate(row)]


def _configure() -> None:
    if not hasattr(PG248.PG247, "ORIGINAL_INPUT_TOKEN_ID"):
        PG248.PG247.ORIGINAL_INPUT_TOKEN_ID = PG237.PG231._input_token_id
    PG237.PG231._input_token_id = PG248._patched_input_token_id
    PG237._load_records = _load_records
    PG237.BASE_DATASET = PG244_DATASET
    PG237.PG237_TRACE = PG246_DATASET
    PG237.FRESH_SOURCE = "__pg251_unused_fresh_source__"
    PG237.FRESH_HOLDOUT_SEEDS = ()
    PG237.EXTRA_HOLDOUT_SOURCE = "__pg251_unused_extra_source__"
    PG237.EXTRA_HOLDOUT_SEEDS = ()
    PG237.HOLDOUT_SOURCE_SUBSTRINGS = ()
    PG237.HOLDOUT_SOURCE_SEED_PAIRS = (
        ("pg242_pikachu_source_native", (24202,)),
        ("pg244_pikachu_sql_repair", (24402,)),
        ("pg244_pikachu_xss_repair", (24402,)),
    )
    PG237.EXCLUDED_SOURCE_SUBSTRINGS = ("pg246_vulnerableapp_source_independent",)
    PG237.SEND_PROBABILITY_THRESHOLD = 0.90
    PG237.CAPACITY_VARIANTS = (1024, 2048, 4096, 8192)
    PG237.TRAIN_STEPS = 180
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg251_preprobe_action"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN


def _eval(artifact: Path, rows: list[dict[str, Any]], base: Any, input_vocab: dict[str, int], device: torch.device, encoder: Any) -> dict[str, Any]:
    PG237.PG231._input_token_id = encoder
    return PG248.PG247._evaluate_artifact(artifact, rows, base, input_vocab, device)


def _finalize() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    rows, counts = _load_records()
    pika_pre = _eligible(rows, lambda row: _pika_holdout(row) and _preprobe(row))
    pika_post = _eligible(rows, lambda row: _pika_holdout(row) and not _preprobe(row))
    vapp_pre = _eligible(rows, lambda row: _vapp(row) and _preprobe(row))
    vapp_post = _eligible(rows, lambda row: _vapp(row) and not _preprobe(row))
    if not pika_pre or not vapp_pre:
        raise RuntimeError("PG-251 requires preprobe Pikachu and VulnerableApp splits")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, input_vocab = PG248.PG247._load_base(device)
    selected_artifact = ROOT / report["selected"]["artifact"]
    original_encoder = PG248.PG247.ORIGINAL_INPUT_TOKEN_ID
    feedback_encoder = PG248._patched_input_token_id
    pika_metrics = _eval(selected_artifact, pika_pre, base, input_vocab, device, feedback_encoder)
    pika_post_metrics = _eval(selected_artifact, pika_post, base, input_vocab, device, feedback_encoder)
    vapp_metrics = _eval(selected_artifact, vapp_pre, base, input_vocab, device, feedback_encoder)
    vapp_post_metrics = _eval(selected_artifact, vapp_post, base, input_vocab, device, feedback_encoder)
    before = _eval(OLD_ADAPTER, pika_post, base, input_vocab, device, original_encoder)
    after = pika_post_metrics
    canary = PG249.PG248.PG247._judge_canary(before, after)
    canary.update({"before_input_encoder": "pg191_default_aliases", "after_input_encoder": "pg248_observable_feedback_alias_v1", "canary_is_post_result_only": True})
    pika_pass = pika_metrics["false_send_count"] == 0 and pika_metrics["missed_send_count"] == 0 and pika_metrics["abstain_recall"] >= 0.80 and pika_metrics["positive_send_recall"] >= 0.80
    vapp_safety = vapp_metrics["false_send_count"] == 0 and vapp_metrics["abstain_recall"] >= 0.80
    vapp_capability = vapp_safety and vapp_metrics["positive_send_recall"] >= 0.80 and vapp_metrics["missed_send_count"] == 0
    judge = {"authority": ["preprobe prefixes derived from audited typed/reference traces", "Pikachu route/seed holdout", "VulnerableApp implementation OOD", "old post-result canary"], "model_output_is_candidate_only": True, "oracle_or_reference_is_not_model_input": True, "preprobe_target_is_off_input": True, "hard_gates": {"pikachu_preprobe_capability": pika_pass, "post_result_canary_no_regression": bool(canary["pass"]), "no_raw_payload_or_response_persistence": True}, "implementation_ood": {"safety_pass": vapp_safety, "capability_pass": vapp_capability, "metrics": vapp_metrics, "post_result_metrics": vapp_post_metrics, "promotion_must_not_merge_into_pikachu_score": True}}
    judge["pass"] = all(judge["hard_gates"].values())
    judge["decision"] = "candidate_eligible_for_next_replay" if judge["pass"] and vapp_capability else "candidate_eligible_for_pikachu_preprobe_replay_only" if judge["pass"] else "blocked"
    splits = {"pikachu_preprobe_holdout": {"record_count": len(pika_pre), "metrics": pika_metrics, "pass": pika_pass}, "pikachu_post_result_holdout": {"record_count": len(pika_post), "metrics": pika_post_metrics}, "vulnerableapp_preprobe_ood": {"record_count": len(vapp_pre), "metrics": vapp_metrics, "safety_pass": vapp_safety, "capability_pass": vapp_capability}, "vulnerableapp_post_result_ood": {"record_count": len(vapp_post), "metrics": vapp_post_metrics}}
    report.update({"protocol_id": "pg-pk-251-preprobe-action-capacity-training-v1", "schema_version": "pg251-preprobe-action-capacity-training-v1", "status": "completed_causal_preprobe_action_capacity_training", "source_datasets": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))], "source_counts": counts, "preprobe_schema": PREPROBE_SCHEMA, "evaluation_splits": splits, "catastrophic_forgetting_canary": canary, "independent_final_judge": judge, "training_eligible": bool(judge["pass"]), "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False, "judge_decision": judge["decision"]}, "honesty": {"frozen_xxl_body_not_updated": True, "adapter_only": True, "preprobe_target_derived_from_authorized_reference_but_not_input": True, "route_seed_holdout_disjoint": True, "vulnerableapp_separate_ood": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "final_judge_is_not_model_self_report": True}})
    report["report_sha256"] = PG237.digest(report)
    dataset["schema_version"] = "pg251-preprobe-action-capacity-training-dataset-v1"
    dataset["source_datasets"] = [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))]
    dataset["preprobe_schema"] = PREPROBE_SCHEMA
    dataset["evaluation_splits"] = splits
    dataset["contract"] = {**dict(dataset.get("contract") or {}), "preprobe_prefix_excludes_result_and_expected_oracle": True, "preprobe_target_off_input": True, "pikachu_route_seed_holdout_never_in_training": True, "vulnerableapp_implementation_ood_separate": True, "post_result_canary_replayed_after_update": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol.update({"protocol_id": "pg-pk-251-preprobe-action-capacity-training-v1", "schema_version": "pg251-preprobe-action-capacity-training-protocol-v1", "preprobe_schema": PREPROBE_SCHEMA, "training_sources": [str(PG244_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))], "preprobe_target": "send_candidate iff audited positive; otherwise abstain", "pika_holdout": ["pg242_pikachu_source_native:24202", "pg244_pikachu_sql_repair:24402", "pg244_pikachu_xss_repair:24402"], "implementation_ood": "pg246_vulnerableapp_source_independent:all_seeds", "excluded_source_substrings": ["pg246_vulnerableapp_source_independent"], "capacity_variants": list(PG237.CAPACITY_VARIANTS), "train_steps": PG237.TRAIN_STEPS, "causal_classification_position": "phase=diagnose immediately after observe prefix", "post_result_canary_required": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    trace.update({"schema_version": "pg251-preprobe-action-capacity-training-trace-v1", "evaluation_splits": splits, "canary": canary, "independent_final_judge": judge, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    PG237._write(TRACE, trace)
    MARKDOWN.write_text("\n".join(["# PG-251 causal pre-probe action capacity training", "", f"train={report['counts']['train_rows']}; preprobe train={counts['preprobe_records']}; Pikachu preprobe holdout={len(pika_pre)}; VulnerableApp preprobe OOD={len(vapp_pre)}", f"Pikachu preprobe send={pika_metrics['positive_send_recall']}; abstain={pika_metrics['abstain_recall']}; false_send={pika_metrics['false_send_count']}; missed={pika_metrics['missed_send_count']}", f"VulnerableApp preprobe send={vapp_metrics['positive_send_recall']}; abstain={vapp_metrics['abstain_recall']}; false_send={vapp_metrics['false_send_count']}; missed={vapp_metrics['missed_send_count']}", f"canary={canary['pass']}; final_judge={judge['decision']}", "", "pre-probe prefix 在 phase=diagnose 之前截断；目标只作为 action head 标签，不进入输入 token。", ""]), encoding="utf-8")
    result = {"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "evaluation_splits": splits, "canary": canary, "final_judge": judge, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    _configure()
    PG237.main()
    _finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
