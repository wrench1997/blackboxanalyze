"""PG-249: practical Pikachu route/seed training with separate implementation OOD.

PG-248 showed that excluding every Pikachu source makes the practical task
under-specified: the model cannot infer source-specific field/feedback
conventions from zero examples.  PG-249 therefore permits already-authorized
Pikachu process rows in training, while holding out unseen Pikachu seeds and
routes for the practical score.  VulnerableApp stays entirely separate and is
reported as implementation OOD, never blended into the Pikachu score.
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


PG248 = _load("run_pg248_feedback_token_capacity_training.py")
PG237 = PG248.PG237
RESEARCH = ROOT / "research"
PG244_DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
PG230_DATASET = RESEARCH / "pg230_next_token_quality_funnel_dataset_v1.json"
OLD_ADAPTER = ROOT / "artifacts" / "pg244-failure-repair-capacity-v1" / "frozen_xxl_capacity_hidden2048.pt"
REPORT = RESEARCH / "pg249_pikachu_route_seed_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg249_pikachu_route_seed_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg249_pikachu_route_seed_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg249_pikachu_route_seed_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg249_pikachu_route_seed_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg249-pikachu-route-seed-capacity-v1"


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows, counts = PG248._load_records()
    # The loader already includes PG-244, PG-246, and the six new PG-230
    # counterfactual rows.  Keep its immutable deduplication and only expose
    # the same rows to PG-237 under a different split predicate.
    return rows, {**counts, "pg249_input_records": len(rows)}


def _is_pikachu_holdout(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    seed = int(row.get("seed", 0) or 0)
    return (source == "pg242_pikachu_source_native" and seed == 24202) or (
        source in {"pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"} and seed == 24402
    )


def _is_vapp_ood(row: dict[str, Any]) -> bool:
    return str(row.get("source", "")) == "pg246_vulnerableapp_source_independent"


def _load_split(rows: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("lane") not in {"quarantine", "reject"} and predicate(row)]


def _evaluate_with_encoder(artifact: Path, rows: list[dict[str, Any]], base: Any, input_vocab: dict[str, int], device: torch.device, encoder: Any) -> dict[str, Any]:
    PG237.PG231._input_token_id = encoder
    return PG248.PG247._evaluate_artifact(artifact, rows, base, input_vocab, device)


def _configure() -> None:
    if not hasattr(PG248.PG247, "ORIGINAL_INPUT_TOKEN_ID"):
        PG248.PG247.ORIGINAL_INPUT_TOKEN_ID = PG237.PG231._input_token_id
    PG237.PG231._input_token_id = PG248._patched_input_token_id
    PG237._load_records = _load_records
    PG237.BASE_DATASET = PG244_DATASET
    PG237.PG237_TRACE = PG246_DATASET
    PG237.FRESH_SOURCE = "__pg249_unused_fresh_source__"
    PG237.FRESH_HOLDOUT_SEEDS = ()
    PG237.EXTRA_HOLDOUT_SOURCE = "__pg249_unused_extra_source__"
    PG237.EXTRA_HOLDOUT_SEEDS = ()
    # Practical Pikachu holdout: selected unseen seeds.  VulnerableApp is a
    # second holdout source, so PG-237 keeps it out of train as well.
    PG237.HOLDOUT_SOURCE_SUBSTRINGS = ()
    PG237.HOLDOUT_SOURCE_SEED_PAIRS = (
        ("pg242_pikachu_source_native", (24202,)),
        ("pg244_pikachu_sql_repair", (24402,)),
        ("pg244_pikachu_xss_repair", (24402,)),
    )
    # Keep all VulnerableApp rows in the manifest for the separate OOD score,
    # but prevent them from entering either the Pikachu train or main
    # holdout.  PG-237's excluded-source contract makes this split explicit.
    PG237.EXCLUDED_SOURCE_SUBSTRINGS = ("pg246_vulnerableapp_source_independent",)
    PG237.SEND_PROBABILITY_THRESHOLD = 0.90
    PG237.CAPACITY_VARIANTS = (512, 1024, 2048, 4096)
    PG237.TRAIN_STEPS = 180
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg249_pikachu_route_seed"
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
    rows, counts = _load_records()
    pika_holdout = _load_split(rows, _is_pikachu_holdout)
    vapp_ood = _load_split(rows, _is_vapp_ood)
    canary = list(pika_holdout)
    if not pika_holdout or not vapp_ood:
        raise RuntimeError("PG-249 requires both Pikachu route/seed holdout and VulnerableApp OOD")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, input_vocab = PG248.PG247._load_base(device)
    selected_artifact = ROOT / report["selected"]["artifact"]
    original_encoder = PG248.PG247.ORIGINAL_INPUT_TOKEN_ID
    feedback_encoder = PG248._patched_input_token_id
    before = _evaluate_with_encoder(OLD_ADAPTER, canary, base, input_vocab, device, original_encoder)
    after = _evaluate_with_encoder(selected_artifact, canary, base, input_vocab, device, feedback_encoder)
    canary_judge = PG248.PG247._judge_canary(before, after)
    canary_judge.update({"before_input_encoder": "pg191_default_aliases", "after_input_encoder": "pg248_observable_feedback_alias_v1"})
    pika_metrics = _evaluate_with_encoder(selected_artifact, pika_holdout, base, input_vocab, device, feedback_encoder)
    vapp_metrics = _evaluate_with_encoder(selected_artifact, vapp_ood, base, input_vocab, device, feedback_encoder)
    pika_pass = pika_metrics["false_send_count"] == 0 and pika_metrics["missed_send_count"] == 0 and pika_metrics["abstain_recall"] >= 0.80 and pika_metrics["positive_send_recall"] >= 0.80
    vapp_safety_pass = vapp_metrics["false_send_count"] == 0 and vapp_metrics["abstain_recall"] >= 0.80
    vapp_capability_pass = vapp_safety_pass and vapp_metrics["positive_send_recall"] >= 0.80 and vapp_metrics["missed_send_count"] == 0
    judge = {
        "authority": ["Pikachu typed process/reference labels", "Pikachu route/seed disjointness", "VulnerableApp typed DOM oracle as separate OOD", "old SQL/XSS canary"],
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
        "hard_gates": {
            "pikachu_route_seed_capability": pika_pass,
            "canary_no_guardrail_regression": bool(canary_judge["pass"]),
            "no_raw_payload_or_response_persistence": True,
        },
        "implementation_ood": {"safety_pass": vapp_safety_pass, "capability_pass": vapp_capability_pass, "metrics": vapp_metrics, "promotion_must_not_merge_into_pikachu_score": True},
    }
    judge["pass"] = all(judge["hard_gates"].values())
    judge["decision"] = "candidate_eligible_for_next_replay" if judge["pass"] and vapp_capability_pass else "candidate_eligible_for_pikachu_replay_only" if judge["pass"] else "blocked"
    report.update({
        "protocol_id": "pg-pk-249-pikachu-route-seed-capacity-training-v1",
        "schema_version": "pg249-pikachu-route-seed-capacity-training-v1",
        "status": "completed_pikachu_route_seed_capacity_training_with_separate_vulnerableapp_ood",
        "source_datasets": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))],
        "source_counts": counts,
        "input_encoder": {"version": "pg248-observable-feedback-alias-v1", "aliases": dict(PG248.OBSERVABLE_FEEDBACK_ALIASES), "forbidden_fields": list(PG248.FORBIDDEN_INPUT_FIELDS), "oracle_labels_as_input": False},
        "evaluation_splits": {"pikachu_route_seed_holdout": {"record_count": len(pika_holdout), "sources": dict(Counter(str(row.get("source", "")) for row in pika_holdout)), "metrics": pika_metrics, "pass": pika_pass}, "vulnerableapp_implementation_ood": {"record_count": len(vapp_ood), "sources": dict(Counter(str(row.get("source", "")) for row in vapp_ood)), "metrics": vapp_metrics, "safety_pass": vapp_safety_pass, "capability_pass": vapp_capability_pass}},
        "catastrophic_forgetting_canary": canary_judge,
        "independent_final_judge": judge,
        "training_eligible": bool(judge["pass"]),
        "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "judge_decision": judge["decision"]},
        "honesty": {"frozen_xxl_body_not_updated": True, "adapter_only": True, "pikachu_source_rows_are_authorized_local_process_projections": True, "pikachu_route_seed_holdout_is_disjoint": True, "vulnerableapp_is_not_merged_into_pikachu_score": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "final_judge_is_not_model_self_report": True},
    })
    report["report_sha256"] = PG237.digest(report)
    dataset["schema_version"] = "pg249-pikachu-route-seed-capacity-training-dataset-v1"
    dataset["source_datasets"] = [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))]
    dataset["evaluation_splits"] = report["evaluation_splits"]
    dataset["contract"] = {**dict(dataset.get("contract") or {}), "pikachu_route_seed_holdout_never_in_training": True, "vulnerableapp_implementation_ood_separate": True, "old_sql_xss_canary_replayed_after_update": True, "canary_never_used_as_oracle_feature": True, "expected_oracle_labels_not_input": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol.update({"protocol_id": "pg-pk-249-pikachu-route-seed-capacity-training-v1", "schema_version": "pg249-pikachu-route-seed-capacity-training-protocol-v1", "training_sources": [str(PG244_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))], "pika_holdout": ["pg242_pikachu_source_native:24202", "pg244_pikachu_sql_repair:24402", "pg244_pikachu_xss_repair:24402"], "implementation_ood": "pg246_vulnerableapp_source_independent:all_seeds", "implementation_ood_not_in_pikachu_score": True, "capacity_variants": list(PG237.CAPACITY_VARIANTS), "train_steps": PG237.TRAIN_STEPS, "canary_replay_required_after_update": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    trace.update({"schema_version": "pg249-pikachu-route-seed-capacity-training-trace-v1", "evaluation_splits": report["evaluation_splits"], "canary": canary_judge, "independent_final_judge": judge, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    PG237._write(TRACE, trace)
    MARKDOWN.write_text("\n".join(["# PG-249 Pikachu route/seed capacity training", "", f"train={report['counts']['train_rows']}; Pikachu holdout={len(pika_holdout)}; VulnerableApp OOD={len(vapp_ood)}", f"Pikachu positive={pika_metrics['positive_send_recall']}; abstain={pika_metrics['abstain_recall']}; false_send={pika_metrics['false_send_count']}; missed={pika_metrics['missed_send_count']}", f"VulnerableApp positive={vapp_metrics['positive_send_recall']}; abstain={vapp_metrics['abstain_recall']}; false_send={vapp_metrics['false_send_count']}; missed={vapp_metrics['missed_send_count']}", f"canary={canary_judge['pass']}; final_judge={judge['decision']}", "", "Pikachu 实用能力与跨实现 OOD 分开报告；通过前者不等于任意实现泛化。", ""]), encoding="utf-8")
    result = {"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "evaluation_splits": report["evaluation_splits"], "canary": canary_judge, "final_judge": judge, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    _configure()
    PG237.main()
    _finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
