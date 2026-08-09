"""PG-306: train a compact causal Transformer-MoE on real process traces.

The dataset mixes the audited abstract PG-301 baseline with PG-305's real
loopback GET/POST process rows and one-slot-missing counterfactuals.  The
runner is a local-morning, CPU-only training experiment so it does not contend
with the user's GPU applications.  It reports raw neural generation separately
from the runtime guard and keeps all promotion gates closed.
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS
from app.pg295_causal_moe import CausalMoEConfig, build_vocabulary, evaluate_causal_moe, generate_target, train_causal_moe
from app.pg301_payload_assembly import evaluate_assembly_rows

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg306_real_process_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg306_real_process_dataset_audit_v1.json"
LIVE_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
CURRICULUM = os.environ.get("PG306_CURRICULUM") == "1"
BALANCED = os.environ.get("PG306_BALANCED") == "1"
_RUN_TAG = "pg306c_balanced_curriculum" if (CURRICULUM and BALANCED) else ("pg306b_real_process_curriculum" if CURRICULUM else ("pg306c_balanced" if BALANCED else "pg306_real_process"))
REPORT_PATH = RESEARCH / f"{_RUN_TAG}_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg306-real-process" / f"{_RUN_TAG}_moe_local_morning.pt"
SEEDS = (30601, 30602, 30603)
CONFIG = CausalMoEConfig(d_model=48, n_heads=2, n_layers=1, experts=2, expert_hidden=96, top_k=1, dropout=0.05, max_length=64)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-306 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-306 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _predictions(model: Any, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> list[list[str]]:
    return [generate_target(model, row.get("context_tokens", []), len(row.get("target_tokens", [])), vocabulary, device) for row in rows]


def _lane(model: Any, rows: list[dict[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    predicted = _predictions(model, rows, vocabulary, device)
    causal = evaluate_causal_moe(model, rows, vocabulary, device)
    assembly = evaluate_assembly_rows(rows, predicted)
    return {"causal": causal, "assembly": assembly}


def _question_rows(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Question-only causal pretraining without exposing final slot labels."""

    rows: list[dict[str, Any]] = []
    for row in train:
        question = next((str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")), "question=none")
        base = copy.deepcopy(row)
        base["target_tokens"] = [TARGET_BOS, question, TARGET_EOS]
        base["source_group"] = "pg306_question_pretrain"
        rows.append(base)
        for variant in ("probe", "recheck"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [f"history_action={variant}" if str(token).startswith("history_action=") else token for token in clone.get("context_tokens", [])]
            rows.append(clone)
    return rows


def _aggregate(values: list[dict[str, Any]], key: str, section: str | None = None) -> dict[str, float]:
    numbers: list[float] = []
    for value in values:
        selected: Any = value
        if section:
            selected = value.get(section) or {}
        item = selected.get(key) if isinstance(selected, Mapping) else None
        if item is not None:
            numbers.append(float(item))
    if not numbers:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": round(sum(numbers) / len(numbers), 6), "min": round(min(numbers), 6), "max": round(max(numbers), 6)}


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    live = _load(LIVE_REPORT_PATH)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-306 refuses a failed dataset audit")
    if not live.get("checks", {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-306 requires a real PG-305 evaluator report")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [row for row in rows if row.get("split") == "train" and row.get("training_eligible")]
    holdout = [row for row in rows if row.get("split") in {"implementation_holdout", "real_live_holdout"}]
    hard = [row for row in rows if row.get("split") == "hard_negative_eval"]
    if not train or not holdout or not hard:
        raise RuntimeError("PG-306 requires train, implementation/real holdout and hard-negative lanes")
    fit_train = list(train)
    if BALANCED:
        # Keep the hard-negative lane evaluation-only while giving the causal
        # decoder enough safe positive transitions to learn the complete plan.
        positive_rows = [copy.deepcopy(row) for row in train if bool(row.get("safe_to_send"))]
        fit_train.extend(copy.deepcopy(row) for row in positive_rows for _ in range(2))
    qtrain = _question_rows(fit_train if CURRICULUM else train) if CURRICULUM else []
    vocabulary = build_vocabulary(fit_train + holdout + hard + qtrain)
    device = torch.device("cpu")
    token_weights = {
        "question=ask_typed_availability": 1.8,
        "question=ask_replay_readiness": 1.8,
        "question=ask_evidence_presence": 1.8,
        "question=ask_feedback_state": 1.8,
        "question=ask_negative_control": 1.8,
        "question=ask_fresh_reset": 1.8,
        "safe_to_send=0": 1.6,
        "safe_to_send=1": 1.8 if BALANCED else 1.4,
        "next_action=repair_abstract_plan": 2.2,
        "repair_action=retry_bounded_variant": 2.2,
        "stop_condition=repair_feedback_or_abstain": 2.2,
    }
    seed_results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    started = time.monotonic()
    for seed in SEEDS:
        initial_state = None
        if CURRICULUM:
            pretrain = train_causal_moe(qtrain, vocabulary, device, seed=seed, config=CONFIG, epochs=60, learning_rate=0.002, token_weights=token_weights)
            initial_state = pretrain.state_dict()
        model = train_causal_moe(fit_train, vocabulary, device, seed=seed + (100 if CURRICULUM else 0), config=CONFIG, epochs=100, learning_rate=0.002, token_weights=token_weights, initial_state=initial_state)
        train_metrics = _lane(model, train, vocabulary, device)
        holdout_metrics = _lane(model, holdout, vocabulary, device)
        hard_metrics = _lane(model, hard, vocabulary, device)
        score = (
            float(holdout_metrics["causal"].get("missing_question_recall") or 0.0)
            + float(holdout_metrics["assembly"].get("assembly_slot_exact") or 0.0)
            + float(holdout_metrics["causal"].get("positive_recall") or 0.0)
            - 2.0 * float(holdout_metrics["assembly"].get("hard_negative_false_allow") or 0.0)
            - 2.0 * float(hard_metrics["causal"].get("hard_negative_false_allow") or 0.0)
        )
        seed_results.append({"seed": seed, "train": train_metrics, "holdout": holdout_metrics, "hard_negative": hard_metrics, "selection_score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_model = model
    elapsed = round(time.monotonic() - started, 3)
    if best_model is None:
        raise RuntimeError("PG-306 did not produce a checkpoint")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema_version": "pg306-real-process-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu"}, "config": CONFIG.__dict__, "vocabulary": vocabulary, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "live_report_sha256": live.get("report_sha256")}
    torch.save(checkpoint, CHECKPOINT_PATH)
    holdout_values = [item["holdout"] for item in seed_results]
    hard_values = [item["hard_negative"] for item in seed_results]
    holdout_question = _aggregate(holdout_values, "missing_question_recall", "causal")
    holdout_slots = _aggregate(holdout_values, "assembly_slot_exact", "assembly")
    holdout_unnecessary = _aggregate(holdout_values, "unnecessary_question_rate", "causal")
    hard_false_allow = _aggregate(hard_values, "hard_negative_false_allow", "causal")
    report = {
        "protocol_id": f"{_RUN_TAG}-v1",
        "schema_version": f"{_RUN_TAG}-training-report-v1",
        "status": f"completed_local_morning_{_RUN_TAG}",
        "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "live_evaluator_report": str(LIVE_REPORT_PATH.relative_to(ROOT)), "live_report_sha256": live.get("report_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False},
        "training": {"architecture": "causal_transformer_moe_next_token", "config": CONFIG.__dict__, "device": "cpu", "seeds": list(SEEDS), "curriculum": CURRICULUM, "balanced_positive_oversample": BALANCED, "fit_count": len(fit_train), "question_pretrain_rows": len(qtrain), "question_pretrain_epochs": 60 if CURRICULUM else 0, "assembly_epochs": 100, "token_weights": token_weights, "train_count": len(train), "holdout_count": len(holdout), "hard_negative_count": len(hard), "elapsed_seconds": elapsed},
        "metrics": {"implementation_and_live_holdout_missing_question_recall": holdout_question, "implementation_and_live_holdout_assembly_slot_exact": holdout_slots, "implementation_and_live_holdout_unnecessary_question_rate": holdout_unnecessary, "hard_negative_false_allow": hard_false_allow, "best_seed": min(seed_results, key=lambda item: -item["selection_score"])["seed"]},
        "per_seed": seed_results,
        "hypothesis_gate": {"status": "blocked", "checks": {"dataset_audit_pass": audit.get("status") == "passed", "real_live_process_rows": int(dataset.get("counts", {}).get("real_process_rows", 0)) > 0, "missing_counterfactuals": int(dataset.get("counts", {}).get("missing_counterfactual_rows", 0)) > 0, "question_recall_min": holdout_question["min"] >= 0.9, "assembly_slot_exact_min": holdout_slots["min"] >= 0.9, "holdout_zero_false_allow": max(float(item["holdout"]["assembly"].get("hard_negative_false_allow", 0)) for item in seed_results) == 0, "hard_negative_zero_false_allow": hard_false_allow["max"] == 0, "unnecessary_question_max": holdout_unnecessary["max"] <= 0.1, "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["model candidate send count in PG-305 was 0/4", "one local implementation and four routes", "source-grounded binding remains outside the neural output", "fresh family/implementation holdout still required"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "gates": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
