"""PG-307: train a causal Transformer-MoE to copy observable Rule-IR slots.

PG-306 showed that a small decoder can learn some question tokens, but direct
concrete transport/field/encoding labels are easy to memorize and the safety
trade-off is unstable.  PG-307 changes only the target representation: the
decoder emits bounded ``*_ref=surface_*`` symbols and a deterministic binder
resolves them against the current observable context.  The neural model still
does not see a route, family, payload, response body, or oracle label.

This is a local-morning, CPU-only experiment.  It never contacts the remote
A800 and keeps every promotion gate closed until a fresh, family/implementation
holdout passes.
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
from app.pg295_causal_moe import (  # noqa: E402
    CausalMoEConfig,
    build_vocabulary,
    evaluate_causal_moe,
    generate_target,
    train_causal_moe,
)
from app.pg301_payload_assembly import evaluate_assembly_rows, target_map  # noqa: E402
from app.pg302_symbolic_assembly import bind_symbolic_plan  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg307_symbolic_real_process_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg307_symbolic_real_process_dataset_audit_v1.json"
LIVE_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
REPORT_PATH = RESEARCH / "pg307_symbolic_real_process_moe_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg307-symbolic-real-process" / "pg307_symbolic_real_process_moe_local_morning.pt"
SEEDS = (30701, 30702, 30703)
CONFIG = CausalMoEConfig(
    d_model=48,
    n_heads=2,
    n_layers=1,
    experts=2,
    expert_hidden=96,
    top_k=1,
    dropout=0.05,
    max_length=64,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-307 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(
            f"PG-307 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})"
        )
    # Explicitly keep the experiment on CPU.  This protects the user's local
    # graphics workload and makes it impossible for this runner to use A800.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _predictions(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
) -> list[list[str]]:
    return [
        generate_target(
            model,
            row.get("context_tokens", []),
            len(row.get("target_tokens", [])),
            vocabulary,
            device,
        )
        for row in rows
    ]


def _bound_lane(
    rows: Sequence[Mapping[str, Any]],
    predicted: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Bind both sides, then score concrete slots without exposing payloads."""

    expected_rows: list[dict[str, Any]] = []
    bound_predictions: list[list[str]] = []
    invalid_expected = 0
    invalid_predicted = 0
    for row, prediction in zip(rows, predicted):
        context = [str(token) for token in row.get("context_tokens") or []]
        expected = bind_symbolic_plan(row.get("target_tokens") or [], context)
        if expected is None:
            invalid_expected += 1
            expected = []
        expected_row = copy.deepcopy(dict(row))
        expected_row["target_tokens"] = expected
        expected_row["safe_to_send"] = target_map(expected).get("safe_to_send") == "1"
        expected_rows.append(expected_row)
        bound = bind_symbolic_plan(prediction, context)
        if bound is None:
            invalid_predicted += 1
            bound = []
        bound_predictions.append(bound)
    metrics = evaluate_assembly_rows(expected_rows, bound_predictions)
    metrics.update(
        {
            "invalid_expected_symbolic_plan_count": invalid_expected,
            "invalid_predicted_symbolic_plan_count": invalid_predicted,
            "binder": "deterministic_pg302_slot_copy",
        }
    )
    return metrics


def _lane(
    model: Any,
    rows: list[dict[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
) -> dict[str, Any]:
    predicted = _predictions(model, rows, vocabulary, device)
    causal = evaluate_causal_moe(model, rows, vocabulary, device)
    bound = _bound_lane(rows, predicted)
    return {"causal_symbolic": causal, "bound_concrete": bound}


def _question_rows(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pretrain only the missing-observation question; hide final slot values."""

    rows: list[dict[str, Any]] = []
    for row in train:
        question = next(
            (str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")),
            "question=none",
        )
        base = copy.deepcopy(row)
        base["target_tokens"] = [TARGET_BOS, question, TARGET_EOS]
        base["source_group"] = "pg307_question_pretrain"
        rows.append(base)
        # Preserve the process-learning objective while exposing benign
        # history synonyms that should not change the required question.
        for variant in ("probe", "recheck"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [
                f"history_action={variant}" if str(token).startswith("history_action=") else token
                for token in clone.get("context_tokens", [])
            ]
            rows.append(clone)
    return rows


def _aggregate(values: list[dict[str, Any]], key: str, section: str) -> dict[str, float]:
    numbers: list[float] = []
    for value in values:
        selected = value.get(section) or {}
        item = selected.get(key) if isinstance(selected, Mapping) else None
        if item is not None:
            numbers.append(float(item))
    if not numbers:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(numbers) / len(numbers), 6),
        "min": round(min(numbers), 6),
        "max": round(max(numbers), 6),
    }


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    live = _load(LIVE_REPORT_PATH)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-307 refuses a failed symbolic dataset audit")
    if not live.get("checks", {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-307 requires the real PG-305 loopback evaluator report")

    rows = [dict(row) for row in dataset.get("records", [])]
    train = [row for row in rows if row.get("split") == "train" and row.get("training_eligible")]
    holdout = [
        row
        for row in rows
        if row.get("split") in {"implementation_holdout", "real_live_holdout"}
    ]
    hard = [row for row in rows if row.get("split") == "hard_negative_eval"]
    if not train or not holdout or not hard:
        raise RuntimeError("PG-307 requires train, implementation/live holdout and hard-negative lanes")

    question_train = _question_rows(train)
    vocabulary = build_vocabulary(train + holdout + hard + question_train)
    device = torch.device("cpu")
    token_weights = {
        "question=ask_typed_availability": 1.8,
        "question=ask_replay_readiness": 1.8,
        "question=ask_evidence_presence": 1.8,
        "question=ask_feedback_state": 1.8,
        "question=ask_negative_control": 1.8,
        "question=ask_fresh_reset": 1.8,
        "safe_to_send=0": 1.8,
        "safe_to_send=1": 1.5,
        "next_action=request_observation": 2.0,
        "next_action=repair_abstract_plan": 2.0,
        "repair_action=retry_bounded_variant": 2.0,
        "stop_condition=repair_feedback_or_abstain": 2.0,
    }

    seed_results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    started = time.monotonic()
    for seed in SEEDS:
        pretrain = train_causal_moe(
            question_train,
            vocabulary,
            device,
            seed=seed,
            config=CONFIG,
            epochs=60,
            learning_rate=0.002,
            token_weights=token_weights,
        )
        model = train_causal_moe(
            train,
            vocabulary,
            device,
            seed=seed + 100,
            config=CONFIG,
            epochs=100,
            learning_rate=0.002,
            token_weights=token_weights,
            initial_state=pretrain.state_dict(),
        )
        train_metrics = _lane(model, train, vocabulary, device)
        holdout_metrics = _lane(model, holdout, vocabulary, device)
        hard_metrics = _lane(model, hard, vocabulary, device)
        score = (
            float(holdout_metrics["causal_symbolic"].get("missing_question_recall") or 0.0)
            + float(holdout_metrics["bound_concrete"].get("assembly_slot_exact") or 0.0)
            + float(holdout_metrics["causal_symbolic"].get("positive_recall") or 0.0)
            - 2.0 * float(holdout_metrics["bound_concrete"].get("hard_negative_false_allow") or 0.0)
            - 2.0 * float(hard_metrics["bound_concrete"].get("hard_negative_false_allow") or 0.0)
        )
        seed_results.append(
            {
                "seed": seed,
                "train": train_metrics,
                "holdout": holdout_metrics,
                "hard_negative": hard_metrics,
                "selection_score": round(score, 6),
            }
        )
        if score > best_score:
            best_score = score
            best_model = model

    elapsed = round(time.monotonic() - started, 3)
    if best_model is None:
        raise RuntimeError("PG-307 did not produce a checkpoint")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "pg307-symbolic-real-process-moe-checkpoint-v1",
        "assignment": {
            "execution_mode": "local_morning_cpu",
            "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "device": "cpu",
        },
        "config": CONFIG.__dict__,
        "vocabulary": vocabulary,
        "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()},
        "dataset_sha256": dataset.get("dataset_sha256"),
        "audit_sha256": audit.get("audit_sha256"),
        "live_report_sha256": live.get("report_sha256"),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    holdout_values = [item["holdout"] for item in seed_results]
    hard_values = [item["hard_negative"] for item in seed_results]
    holdout_question = _aggregate(holdout_values, "missing_question_recall", "causal_symbolic")
    holdout_raw_slots = _aggregate(holdout_values, "sequence_exact_accuracy", "causal_symbolic")
    holdout_slots = _aggregate(holdout_values, "assembly_slot_exact", "bound_concrete")
    holdout_unnecessary = _aggregate(holdout_values, "unnecessary_question_rate", "causal_symbolic")
    hard_false_allow = _aggregate(hard_values, "hard_negative_false_allow", "bound_concrete")
    holdout_false_allow = _aggregate(holdout_values, "hard_negative_false_allow", "bound_concrete")

    report = {
        "protocol_id": "pg307-symbolic-real-process-moe-v1",
        "schema_version": "pg307-symbolic-real-process-moe-training-report-v1",
        "status": "completed_local_morning_pg307_symbolic_slot_copy",
        "source": {
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "dataset_sha256": dataset.get("dataset_sha256"),
            "audit": str(AUDIT_PATH.relative_to(ROOT)),
            "audit_sha256": audit.get("audit_sha256"),
            "live_evaluator_report": str(LIVE_REPORT_PATH.relative_to(ROOT)),
            "live_report_sha256": live.get("report_sha256"),
            "raw_payload_in_context": False,
            "raw_response_body_in_context": False,
            "wire_emission": False,
        },
        "training": {
            "architecture": "causal_transformer_moe_next_token",
            "target_representation": "symbolic_slot_copy_refs",
            "binder": "deterministic_pg302_slot_copy",
            "config": CONFIG.__dict__,
            "device": "cpu",
            "seeds": list(SEEDS),
            "question_pretrain_epochs": 60,
            "assembly_epochs": 100,
            "question_pretrain_rows": len(question_train),
            "fit_count": len(train),
            "token_weights": token_weights,
            "train_count": len(train),
            "holdout_count": len(holdout),
            "hard_negative_count": len(hard),
            "elapsed_seconds": elapsed,
        },
        "metrics": {
            "implementation_and_live_holdout_missing_question_recall": holdout_question,
            "implementation_and_live_holdout_raw_sequence_exact": holdout_raw_slots,
            "implementation_and_live_holdout_bound_assembly_slot_exact": holdout_slots,
            "implementation_and_live_holdout_unnecessary_question_rate": holdout_unnecessary,
            "implementation_and_live_holdout_bound_false_allow": holdout_false_allow,
            "hard_negative_bound_false_allow": hard_false_allow,
            "best_seed": min(seed_results, key=lambda item: -item["selection_score"])["seed"],
        },
        "per_seed": seed_results,
        "hypothesis_gate": {
            "status": "blocked",
            "checks": {
                "dataset_audit_pass": audit.get("status") == "passed",
                "real_process_rows": int(dataset.get("counts", {}).get("real_process_rows", 0)) > 0,
                "missing_counterfactuals": int(dataset.get("counts", {}).get("missing_counterfactual_rows", 0)) > 0,
                "question_recall_min": holdout_question["min"] >= 0.9,
                "bound_assembly_slot_exact_min": holdout_slots["min"] >= 0.9,
                "holdout_zero_false_allow": holdout_false_allow["max"] == 0,
                "hard_negative_zero_false_allow": hard_false_allow["max"] == 0,
                "unnecessary_question_max": holdout_unnecessary["max"] <= 0.1,
                "promotion_blocked": True,
            },
            "claim_allowed": False,
        },
        "scientific_gate": {
            "status": "blocked",
            "reasons": [
                "PG-305 model candidate send count was 0/4; typed gold effects are evaluator evidence only",
                "PG-307 is a symbolic projection of four real routes, not a new family-level sample",
                "source-grounded binding remains deterministic and outside the neural decoder",
                "fresh family/implementation holdout and multi-target replay are still required",
            ],
            "claim_allowed": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "checkpoint_role": "research_candidate_only",
        },
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": report["metrics"],
                "gates": report["hypothesis_gate"],
                "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
