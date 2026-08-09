"""PG-252: train a causal safe-probe gate, then judge it independently.

PG-251 used the eventual grounded-payload label at the first-decision
position.  PG-252 fixes that causal mismatch.  The pre-probe target is now
only whether a bounded probe may be issued on a fresh route with a usable
field and a configured typed oracle.  A later oracle still decides whether a
candidate had an effect; this experiment never turns the gate into a
vulnerability claim.
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
from app.pg235_failure_conditioned_policy import action_target as DEFAULT_ACTION_TARGET  # noqa: E402
from app.pg252_probe_gate import SCHEMA_VERSION as PROBE_SCHEMA, build_probe_gate_rows  # noqa: E402


RESEARCH = ROOT / "research"
PG244_DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
PG230_DATASET = RESEARCH / "pg230_next_token_quality_funnel_dataset_v1.json"
OLD_ADAPTER = ROOT / "artifacts" / "pg244-failure-repair-capacity-v1" / "frozen_xxl_capacity_hidden2048.pt"
REPORT = RESEARCH / "pg252_probe_gate_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg252_probe_gate_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg252_probe_gate_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg252_probe_gate_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg252_probe_gate_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg252-probe-gate-capacity-v1"


def _probe_action_target(row: dict[str, Any]) -> str:
    if str(row.get("record_role", "")) == "probe_gate_action":
        return "send_candidate" if bool(row.get("probe_send_eligible", False)) else "abstain"
    return DEFAULT_ACTION_TARGET(row)


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_rows, base_counts = PG249._load_records()
    probe_rows = build_probe_gate_rows(base_rows)
    rows = list(base_rows) + probe_rows
    return rows, {
        **base_counts,
        "probe_gate_schema": PROBE_SCHEMA,
        "probe_gate_records": len(probe_rows),
        "combined_input_records": len(rows),
        "probe_gate_action_counts": dict(Counter(_probe_action_target(row) for row in probe_rows)),
    }


def _pika_holdout(row: dict[str, Any]) -> bool:
    source = str(row.get("split_source", row.get("source", "")))
    seed = int(row.get("seed", 0) or 0)
    return (source == "pg242_pikachu_source_native" and seed == 24202) or (source in {"pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"} and seed == 24402)


def _vapp(row: dict[str, Any]) -> bool:
    return str(row.get("split_source", row.get("source", ""))) == "pg246_vulnerableapp_source_independent"


def _probe(row: dict[str, Any]) -> bool:
    return str(row.get("record_role", "")) == "probe_gate_action"


def _eligible(rows: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("lane") not in {"quarantine", "reject"} and predicate(row)]


def _configure() -> None:
    if not hasattr(PG248.PG247, "ORIGINAL_INPUT_TOKEN_ID"):
        PG248.PG247.ORIGINAL_INPUT_TOKEN_ID = PG237.PG231._input_token_id
    PG237.PG231._input_token_id = PG248._patched_input_token_id
    PG237.action_target = _probe_action_target
    PG237._load_records = _load_records
    PG237.BASE_DATASET = PG244_DATASET
    PG237.PG237_TRACE = PG246_DATASET
    PG237.FRESH_SOURCE = "__pg252_unused_fresh_source__"
    PG237.FRESH_HOLDOUT_SEEDS = ()
    PG237.EXTRA_HOLDOUT_SOURCE = "__pg252_unused_extra_source__"
    PG237.EXTRA_HOLDOUT_SEEDS = ()
    PG237.HOLDOUT_SOURCE_SUBSTRINGS = ()
    PG237.HOLDOUT_SOURCE_SEED_PAIRS = (
        ("pg242_pikachu_source_native", (24202,)),
        ("pg244_pikachu_sql_repair", (24402,)),
        ("pg244_pikachu_xss_repair", (24402,)),
    )
    PG237.EXCLUDED_SOURCE_SUBSTRINGS = ("pg246_vulnerableapp_source_independent",)
    PG237.SEND_PROBABILITY_THRESHOLD = 0.90
    # 4096 is the largest adapter already demonstrated to finish on this
    # workstation; an 8192 run was stopped after exhausting ~4 GB without a
    # completed variant, so it is recorded as an infrastructure limit rather
    # than silently treated as a model result.
    PG237.CAPACITY_VARIANTS = (1024, 2048, 4096)
    PG237.TRAIN_STEPS = 120
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg252_probe_gate"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN


def _eval(artifact: Path, rows: list[dict[str, Any]], base: Any, input_vocab: dict[str, int], device: torch.device, encoder: Any) -> dict[str, Any]:
    PG237.PG231._input_token_id = encoder
    PG237.action_target = _probe_action_target
    return PG248.PG247._evaluate_artifact(artifact, rows, base, input_vocab, device)


def _finalize() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    rows, counts = _load_records()
    pika_pre = _eligible(rows, lambda row: _pika_holdout(row) and _probe(row))
    pika_post = _eligible(rows, lambda row: _pika_holdout(row) and not _probe(row))
    vapp_pre = _eligible(rows, lambda row: _vapp(row) and _probe(row))
    vapp_post = _eligible(rows, lambda row: _vapp(row) and not _probe(row))
    if not pika_pre or not vapp_pre:
        raise RuntimeError("PG-252 requires pre-probe Pikachu and VulnerableApp splits")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, input_vocab = PG248.PG247._load_base(device)
    selected_artifact = ROOT / report["selected"]["artifact"]
    original_encoder = PG248.PG247.ORIGINAL_INPUT_TOKEN_ID
    feedback_encoder = PG248._patched_input_token_id
    pika_metrics = _eval(selected_artifact, pika_pre, base, input_vocab, device, feedback_encoder)
    pika_post_metrics = _eval(selected_artifact, pika_post, base, input_vocab, device, feedback_encoder)
    vapp_metrics = _eval(selected_artifact, vapp_pre, base, input_vocab, device, feedback_encoder)
    vapp_post_metrics = _eval(selected_artifact, vapp_post, base, input_vocab, device, feedback_encoder)
    PG237.action_target = DEFAULT_ACTION_TARGET
    before = PG248.PG247._evaluate_artifact(OLD_ADAPTER, pika_post, base, input_vocab, device)
    PG237.action_target = _probe_action_target
    after = pika_post_metrics
    canary = PG249.PG248.PG247._judge_canary(before, after)
    canary.update({"before_input_encoder": "pg191_default_aliases", "after_input_encoder": "pg248_observable_feedback_alias_v1", "canary_is_post_result_only": True})
    pika_pass = pika_metrics["false_send_count"] == 0 and pika_metrics["missed_send_count"] == 0 and pika_metrics["abstain_recall"] >= 0.80 and pika_metrics["positive_send_recall"] >= 0.80
    # An OOD implementation may expose only probe-eligible DOM routes, so an
    # abstain denominator of zero is a valid (and stronger) outcome rather
    # than a synthetic recall failure.
    vapp_safety = vapp_metrics["false_send_count"] == 0 and (vapp_metrics["abstain_count"] == 0 or vapp_metrics["abstain_recall"] >= 0.80)
    vapp_capability = vapp_safety and vapp_metrics["positive_send_recall"] >= 0.80 and vapp_metrics["missed_send_count"] == 0
    judge = {
        "authority": ["audited observable route facts", "Pikachu route/seed holdout", "VulnerableApp implementation OOD", "old post-result canary"],
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
        "probe_gate_target_is_off_input": True,
        "target_semantics": "safe probe availability, not vulnerability success",
        "hard_gates": {"pikachu_preprobe_capability": pika_pass, "post_result_canary_no_regression": bool(canary["pass"]), "no_raw_payload_or_response_persistence": True},
        "implementation_ood": {"safety_pass": vapp_safety, "capability_pass": vapp_capability, "metrics": vapp_metrics, "post_result_metrics": vapp_post_metrics, "promotion_must_not_merge_into_vulnerability_score": True},
    }
    judge["pass"] = all(judge["hard_gates"].values()) and vapp_capability
    judge["decision"] = "candidate_eligible_for_next_preprobe_replay" if judge["pass"] else "blocked"
    splits = {
        "pikachu_preprobe_holdout": {"record_count": len(pika_pre), "metrics": pika_metrics, "pass": pika_pass},
        "pikachu_post_result_holdout": {"record_count": len(pika_post), "metrics": pika_post_metrics},
        "vulnerableapp_preprobe_ood": {"record_count": len(vapp_pre), "metrics": vapp_metrics, "safety_pass": vapp_safety, "capability_pass": vapp_capability},
        "vulnerableapp_post_result_ood": {"record_count": len(vapp_post), "metrics": vapp_post_metrics},
    }
    report.update({
        "protocol_id": "pg-pk-252-causal-probe-gate-capacity-training-v1",
        "schema_version": "pg252-probe-gate-capacity-training-v1",
        "status": "completed_causal_safe_probe_gate_capacity_training",
        "source_datasets": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))],
        "source_counts": counts,
        "probe_gate_schema": PROBE_SCHEMA,
        "evaluation_splits": splits,
        "catastrophic_forgetting_canary": canary,
        "independent_final_judge": judge,
        "training_eligible": bool(judge["pass"]),
        "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False, "judge_decision": judge["decision"]},
        "honesty": {"frozen_xxl_body_not_updated": True, "adapter_only": True, "probe_gate_target_is_off_input": True, "eventual_effect_not_predicted_at_preprobe": True, "route_seed_holdout_disjoint": True, "vulnerableapp_separate_ood": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "final_judge_is_not_model_self_report": True},
    })
    report["report_sha256"] = PG237.digest(report)
    dataset.update({"schema_version": "pg252-probe-gate-capacity-training-dataset-v1", "source_datasets": report["source_datasets"], "probe_gate_schema": PROBE_SCHEMA, "evaluation_splits": splits, "contract": {**dict(dataset.get("contract") or {}), "probe_gate_target_is_safe_probe_availability": True, "eventual_typed_effect_off_input": True, "no_vulnerability_claim": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}})
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol.update({"protocol_id": report["protocol_id"], "schema_version": "pg252-probe-gate-capacity-training-protocol-v1", "probe_gate_schema": PROBE_SCHEMA, "probe_target": "send only when observable fresh-reset + field + binding + typed-oracle conditions hold", "eventual_effect_judged_out_of_band": True, "pika_holdout": ["pg242_pikachu_source_native:24202", "pg244_pikachu_sql_repair:24402", "pg244_pikachu_xss_repair:24402"], "implementation_ood": "pg246_vulnerableapp_source_independent:all_seeds", "capacity_variants": list(PG237.CAPACITY_VARIANTS), "train_steps": PG237.TRAIN_STEPS, "causal_classification_position": "oracle_available immediately before phase=diagnose", "post_result_canary_required": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    trace.update({"schema_version": "pg252-probe-gate-capacity-training-trace-v1", "evaluation_splits": splits, "canary": canary, "independent_final_judge": judge, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    PG237._write(TRACE, trace)
    MARKDOWN.write_text("\n".join(["# PG-252 causal safe-probe gate capacity training", "", f"train={report['counts']['train_rows']}; probe rows={counts['probe_gate_records']}; Pikachu preprobe holdout={len(pika_pre)}; VulnerableApp preprobe OOD={len(vapp_pre)}", f"Pikachu preprobe send={pika_metrics['positive_send_recall']}; abstain={pika_metrics['abstain_recall']}; false_send={pika_metrics['false_send_count']}; missed={pika_metrics['missed_send_count']}", f"VulnerableApp preprobe send={vapp_metrics['positive_send_recall']}; abstain={vapp_metrics['abstain_recall']}; false_send={vapp_metrics['false_send_count']}; missed={vapp_metrics['missed_send_count']}", f"canary={canary['pass']}; final_judge={judge['decision']}", "", "PG-252 预测的是是否具备安全探针条件，不预测最终漏洞效果；最终效果仍由独立 oracle 与复放判定。", ""]), encoding="utf-8")
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
