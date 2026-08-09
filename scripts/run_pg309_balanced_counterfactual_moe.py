"""PG-309: safety-balanced counterfactual training after the PG-308 failure.

The dataset adds explicit missing/complete pairs and failure/mismatch repairs.
This runner is a causal Transformer-MoE next-token model, not a classifier;
raw predictions, deterministic binding and hard-negative safety are reported
separately.  The experiment remains CPU-only and promotion-closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, build_vocabulary, train_causal_moe  # noqa: E402
from run_pg308_multisource_slot_moe import _aggregate, _lane, _source_lanes  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg309_balanced_counterfactual_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg309_balanced_counterfactual_dataset_audit_v1.json"
PG305_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
PG308_REPORT_PATH = RESEARCH / "pg308_multisource_slot_moe_training_report_v1_local_morning.json"
REPORT_PATH = RESEARCH / "pg309_balanced_counterfactual_moe_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg309-balanced-counterfactual" / "pg309_balanced_counterfactual_moe_local_morning.pt"
SEEDS = (30901, 30902, 30903)
CONFIG = CausalMoEConfig(d_model=48, n_heads=2, n_layers=1, experts=2, expert_hidden=96, top_k=1, dropout=0.05, max_length=64)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-309 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-309 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _question_rows(train: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in train:
        question = next((str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")), "question=none")
        base = copy.deepcopy(dict(row))
        base["target_tokens"] = [TARGET_BOS, question, TARGET_EOS]
        base["source_group"] = "pg309_question_pretrain"
        rows.append(base)
        for variant in ("probe", "recheck"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [f"history_action={variant}" if str(token).startswith("history_action=") else token for token in clone.get("context_tokens", [])]
            rows.append(clone)
    return rows


def _source_aggregate(seed_results: Sequence[Mapping[str, Any]], lane_name: str, metric: str, section: str) -> dict[str, dict[str, float]]:
    sources: set[str] = set()
    for result in seed_results:
        sources.update((result.get(lane_name) or {}).keys())
    output: dict[str, dict[str, float]] = {}
    for source in sorted(sources):
        output[source] = _aggregate([result[lane_name][source] for result in seed_results if source in result.get(lane_name, {})], metric, section)
    return output


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    pg305 = _load(PG305_REPORT_PATH)
    pg308 = _load(PG308_REPORT_PATH)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-309 refuses a failed counterfactual dataset audit")
    if not (pg305.get("checks") or {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-309 requires PG-305 real loopback evaluator evidence")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [row for row in rows if row.get("split") == "train" and row.get("training_eligible")]
    holdout = [row for row in rows if row.get("split") in {"implementation_holdout", "real_live_holdout"}]
    hard = [row for row in rows if row.get("split") == "hard_negative_eval"]
    if not train or not holdout or not hard:
        raise RuntimeError("PG-309 requires train/source holdout/hard-negative lanes")
    qtrain = _question_rows(train)
    vocabulary = build_vocabulary(train + holdout + hard + qtrain)
    device = torch.device("cpu")
    token_weights = {
        "question=ask_typed_availability": 2.3,
        "question=ask_replay_readiness": 2.3,
        "question=ask_evidence_presence": 2.3,
        "question=ask_feedback_state": 2.3,
        "question=ask_negative_control": 2.3,
        "question=ask_fresh_reset": 2.3,
        "safe_to_send=0": 2.2,
        "safe_to_send=1": 2.2,
        "next_action=request_observation": 2.2,
        "next_action=repair_abstract_plan": 2.2,
        "repair_action=retry_bounded_variant": 2.2,
        "stop_condition=repair_feedback_or_abstain": 2.2,
    }
    seed_results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    started = time.monotonic()
    for seed in SEEDS:
        pre = train_causal_moe(qtrain, vocabulary, device, seed=seed, config=CONFIG, epochs=70, learning_rate=0.002, token_weights=token_weights)
        model = train_causal_moe(train, vocabulary, device, seed=seed + 100, config=CONFIG, epochs=140, learning_rate=0.002, token_weights=token_weights, initial_state=pre.state_dict())
        train_metrics = _lane(model, train, vocabulary, device)
        holdout_metrics = _lane(model, holdout, vocabulary, device)
        hard_metrics = _lane(model, hard, vocabulary, device)
        holdout_by_source = _source_lanes(model, holdout, vocabulary, device)
        hard_by_source = _source_lanes(model, hard, vocabulary, device)
        score = float(holdout_metrics["causal_symbolic"].get("missing_question_recall") or 0.0) + float(holdout_metrics["bound_concrete"].get("assembly_slot_exact") or 0.0) + float(holdout_metrics["causal_symbolic"].get("positive_recall") or 0.0) - 2.0 * float(holdout_metrics["bound_concrete"].get("hard_negative_false_allow") or 0.0) - 2.0 * float(hard_metrics["bound_concrete"].get("hard_negative_false_allow") or 0.0)
        seed_results.append({"seed": seed, "train": train_metrics, "holdout": holdout_metrics, "hard_negative": hard_metrics, "holdout_by_source": holdout_by_source, "hard_negative_by_source": hard_by_source, "selection_score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_model = model
    elapsed = round(time.monotonic() - started, 3)
    if best_model is None:
        raise RuntimeError("PG-309 did not produce a checkpoint")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg309-balanced-counterfactual-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu"}, "config": CONFIG.__dict__, "vocabulary": vocabulary, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, CHECKPOINT_PATH)

    holdout_values = [item["holdout"] for item in seed_results]
    hard_values = [item["hard_negative"] for item in seed_results]
    holdout_question = _aggregate(holdout_values, "missing_question_recall", "causal_symbolic")
    holdout_slots = _aggregate(holdout_values, "assembly_slot_exact", "bound_concrete")
    holdout_unnecessary = _aggregate(holdout_values, "unnecessary_question_rate", "causal_symbolic")
    holdout_bound_false = _aggregate(holdout_values, "hard_negative_false_allow", "bound_concrete")
    holdout_raw_false = _aggregate(holdout_values, "hard_negative_false_allow", "causal_symbolic")
    hard_bound_false = _aggregate(hard_values, "hard_negative_false_allow", "bound_concrete")
    hard_raw_false = _aggregate(hard_values, "hard_negative_false_allow", "causal_symbolic")
    source_question = _source_aggregate(seed_results, "holdout_by_source", "missing_question_recall", "causal_symbolic")
    source_slots = _source_aggregate(seed_results, "holdout_by_source", "assembly_slot_exact", "bound_concrete")
    source_unnecessary = _source_aggregate(seed_results, "holdout_by_source", "unnecessary_question_rate", "causal_symbolic")
    source_hard_false = _source_aggregate(seed_results, "hard_negative_by_source", "hard_negative_false_allow", "bound_concrete")
    source_min_question = min((x["min"] for x in source_question.values()), default=0.0)
    source_min_slots = min((x["min"] for x in source_slots.values()), default=0.0)
    source_max_unnecessary = max((x["max"] for x in source_unnecessary.values()), default=0.0)
    slot_perm_false = float((source_hard_false.get("pg308_slot_permutation_hard_negative") or {}).get("max", 0.0) or 0.0)
    report = {
        "protocol_id": "pg309-balanced-counterfactual-moe-v1",
        "schema_version": "pg309-balanced-counterfactual-moe-training-report-v1",
        "status": "completed_local_morning_pg309_balanced_counterfactual",
        "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "pg305_report": str(PG305_REPORT_PATH.relative_to(ROOT)), "pg305_report_sha256": pg305.get("report_sha256"), "pg308_report": str(PG308_REPORT_PATH.relative_to(ROOT)), "pg308_report_sha256": pg308.get("report_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False},
        "training": {"architecture": "causal_transformer_moe_next_token", "target_representation": "symbolic_slot_copy_refs", "binder": "deterministic_pg302_slot_copy", "config": CONFIG.__dict__, "device": "cpu", "seeds": list(SEEDS), "question_pretrain_epochs": 70, "assembly_epochs": 140, "question_pretrain_rows": len(qtrain), "fit_count": len(train), "train_count": len(train), "holdout_count": len(holdout), "hard_negative_count": len(hard), "elapsed_seconds": elapsed, "token_weights": token_weights},
        "metrics": {"source_holdout_missing_question_recall": holdout_question, "source_holdout_bound_assembly_slot_exact": holdout_slots, "source_holdout_unnecessary_question_rate": holdout_unnecessary, "source_holdout_bound_false_allow": holdout_bound_false, "source_holdout_raw_false_allow": holdout_raw_false, "hard_negative_bound_false_allow": hard_bound_false, "hard_negative_raw_false_allow": hard_raw_false, "slot_permutation_bound_false_allow": slot_perm_false, "per_source_missing_question_recall": source_question, "per_source_bound_slot_exact": source_slots, "per_source_unnecessary_question_rate": source_unnecessary, "per_source_hard_negative_bound_false_allow": source_hard_false, "best_seed": min(seed_results, key=lambda item: -item["selection_score"])["seed"]},
        "per_seed": seed_results,
        "hypothesis_gate": {"status": "blocked", "checks": {"dataset_audit_pass": audit.get("status") == "passed", "counterfactual_complete_missing_pairs": int(dataset.get("counts", {}).get("generated_missing", 0) or 0) > 0 and int(dataset.get("counts", {}).get("generated_complete", 0) or 0) > 0, "source_question_recall_min": source_min_question >= 0.9, "source_bound_slot_exact_min": source_min_slots >= 0.9, "source_unnecessary_question_max": source_max_unnecessary <= 0.1, "holdout_zero_bound_false_allow": holdout_bound_false["max"] == 0, "hard_zero_bound_false_allow": hard_bound_false["max"] == 0, "slot_permutation_zero_bound_false_allow": slot_perm_false == 0, "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["PG-308 showed source/seed safety-coverage instability; PG-309 must not hide it with a guard", "PG-305 model candidate send count remains 0/4", "abstract counterfactual improvement is not new Docker evidence", "fresh independent implementation and model-generated candidate replay remain required"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "gates": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
