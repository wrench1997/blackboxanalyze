"""PG-317: train and audit multi-missing-observation question anchors.

This remains a decoder-only next-token experiment.  The model emits abstract
Rule-IR slots; the evaluator checks whether it asks for the correct missing
observation before it is allowed to assemble anything.  No literal payload or
response body is present in model context and this runner never emits wire
traffic.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
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


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG313 = _load_module("pg313_for_pg317", ROOT / "scripts" / "run_pg313_probe_variant_moe.py")
PG316 = _load_module("pg316_for_pg317", ROOT / "scripts" / "run_pg316_failure_repair_moe.py")
from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS  # noqa: E402
from app.pg301_payload_assembly import target_map  # noqa: E402
from app.pg313_probe_variant import bind_probe_variant_plan  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg317_question_anchor_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg317_question_anchor_dataset_audit_v1.json"
PG305_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
REPORT_PATH = RESEARCH / "pg317_question_anchor_moe_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg317-question-anchor" / "pg317_question_anchor_moe_local_morning.pt"
SEED_CHECKPOINT_DIR = ROOT / "artifacts" / "pg317-question-anchor" / "seeds"
SEEDS = (31701, 31702, 31703)
CONFIG = PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-317 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-317 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _anchor_question_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pretrain only the question prefix on anchor rows.

    The context is unchanged; only the target is shortened to the question
    token.  This is a next-token language-model curriculum, not a separate
    classification head.  Two copies with harmless history labels make the
    question invariant to an action-history spelling.
    """

    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("counterfactual_kind") != "ask_complete_pair" or row.get("anchor_role") != "ask":
            continue
        question = next((str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")), "question=none")
        base = copy.deepcopy(dict(row))
        base["target_tokens"] = [TARGET_BOS, question, TARGET_EOS]
        for history in ("ask_anchor", "ask_anchor_retry"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [f"history_action={history}" if str(token).startswith("history_action=") else token for token in clone.get("context_tokens", [])]
            clone["training_eligible"] = True
            clone["question_anchor_pretrain"] = True
            result.append(clone)
    return result


def _anchor_metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[str]]) -> dict[str, Any]:
    ask_total = ask_correct = ask_false_allow = 0
    complete_total = complete_unnecessary = complete_exact = 0
    invalid = 0
    for row, prediction in zip(rows, predictions):
        if row.get("counterfactual_kind") != "ask_complete_pair":
            continue
        context = [str(token) for token in row.get("context_tokens") or []]
        expected = bind_probe_variant_plan(row.get("target_tokens") or [], context) or []
        actual = bind_probe_variant_plan(prediction, context)
        if actual is None:
            invalid += 1
            actual = []
        expected_values = target_map(expected)
        actual_values = target_map(actual)
        if row.get("anchor_role") == "ask":
            ask_total += 1
            ask_correct += int(actual_values.get("question") == expected_values.get("question") and actual_values.get("next_action") == "request_observation")
            ask_false_allow += int(actual_values.get("safe_to_send") == "1")
        elif row.get("anchor_role") == "complete":
            complete_total += 1
            complete_unnecessary += int(actual_values.get("question", "none") != "none")
            complete_exact += int(actual_values == expected_values)
    return {
        "ask_count": ask_total,
        "ask_question_exact": round(ask_correct / max(ask_total, 1), 6),
        "ask_safe_allow": ask_false_allow,
        "complete_count": complete_total,
        "complete_unnecessary_question_rate": round(complete_unnecessary / max(complete_total, 1), 6),
        "complete_bound_exact": round(complete_exact / max(complete_total, 1), 6),
        "invalid_plan_count": invalid,
    }


def _lane(model: Any, rows: list[dict[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    predictions = PG313._predictions(model, rows, vocab, device)
    return {
        "causal_symbolic": PG313.evaluate_causal_moe(model, rows, vocab, device),
        "bound_probe": PG313._bound_metrics(rows, predictions),
        "anchor": _anchor_metrics(rows, predictions),
        "repair": PG316._repair_metrics(rows, predictions),
    }


def _aggregate(values: Sequence[Mapping[str, Any]], key: str, section: str) -> dict[str, float]:
    nums = [float((value.get(section) or {}).get(key)) for value in values if (value.get(section) or {}).get(key) is not None]
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6)} if nums else {"mean": 0.0, "min": 0.0, "max": 0.0}


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    pg305 = _load(PG305_REPORT_PATH)
    if audit.get("status") != "passed" or not (pg305.get("checks") or {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-317 requires passed dataset audit and PG-305 real evaluator")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [row for row in rows if row.get("split") == "train" and row.get("training_eligible")]
    holdout = [row for row in rows if row.get("split") in {"implementation_holdout", "real_live_holdout"}]
    hard = [row for row in rows if row.get("split") == "hard_negative_eval"]
    qtrain = PG313._question_rows(train) + _anchor_question_rows(train)
    vocab = PG313.build_vocabulary(train + holdout + hard + qtrain)
    device = torch.device("cpu")
    weights = {
        "question=ask_typed_availability": 8.0,
        "question=ask_replay_readiness": 8.0,
        "question=ask_evidence_presence": 8.0,
        "question=ask_feedback_state": 8.0,
        "question=ask_negative_control": 8.0,
        "question=ask_fresh_reset": 8.0,
        "safe_to_send=0": 5.0,
        "safe_to_send=1": 2.6,
        "next_action=request_observation": 7.0,
        "next_action=repair_abstract_plan": 6.0,
        "next_action=assemble_abstract_plan": 2.0,
        "repair_action=retry_bounded_variant": 6.0,
        "stop_condition=repair_feedback_or_abstain": 5.0,
        "probe_variant_ref=none": 5.0,
        "probe_variant_ref=source_attested_candidate": 2.0,
        "probe_variant_ref=reference_canary": 2.0,
        "probe_variant_ref=negative_control": 2.0,
    }
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    started = time.monotonic()
    SEED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        pre = PG313.train_causal_moe(qtrain, vocab, device, seed=seed, config=CONFIG, epochs=200, learning_rate=0.001, token_weights=weights)
        model = PG313.train_causal_moe(train, vocab, device, seed=seed + 100, config=CONFIG, epochs=260, learning_rate=0.001, token_weights=weights, initial_state=pre.state_dict())
        train_m = _lane(model, train, vocab, device)
        hold_m = _lane(model, holdout, vocab, device)
        hard_m = _lane(model, hard, vocab, device)
        score = (
            3.0 * float(hold_m["anchor"].get("ask_question_exact") or 0.0)
            + 1.5 * float(hold_m["bound_probe"].get("missing_question_recall") or 0.0)
            + float(hold_m["bound_probe"].get("variant_exact") or 0.0)
            + float(hold_m["repair"].get("repair_exact") or 0.0)
            - 4.0 * float(hold_m["anchor"].get("ask_safe_allow") or 0.0)
            - 2.0 * float(hard_m["bound_probe"].get("hard_negative_false_allow") or 0.0)
        )
        seed_checkpoint = SEED_CHECKPOINT_DIR / f"pg317_question_anchor_moe_seed_{seed}.pt"
        torch.save({"schema_version": "pg317-question-anchor-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "seed": seed}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, seed_checkpoint)
        results.append({"seed": seed, "train": train_m, "holdout": hold_m, "hard_negative": hard_m, "selection_score": round(score, 6), "checkpoint": str(seed_checkpoint.relative_to(ROOT))})
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("PG-317 did not produce a checkpoint")
    elapsed = round(time.monotonic() - started, 3)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg317-question-anchor-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "selected": True}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, CHECKPOINT_PATH)
    hold_values = [row["holdout"] for row in results]
    hard_values = [row["hard_negative"] for row in results]
    metrics = {
        "holdout_anchor_question_exact": _aggregate(hold_values, "ask_question_exact", "anchor"),
        "holdout_anchor_safe_allow_max": _aggregate(hold_values, "ask_safe_allow", "anchor"),
        "holdout_anchor_unnecessary_question": _aggregate(hold_values, "complete_unnecessary_question_rate", "anchor"),
        "holdout_anchor_complete_exact": _aggregate(hold_values, "complete_bound_exact", "anchor"),
        "holdout_missing_question_recall": _aggregate(hold_values, "missing_question_recall", "bound_probe"),
        "holdout_variant_exact": _aggregate(hold_values, "variant_exact", "bound_probe"),
        "holdout_repair_exact": _aggregate(hold_values, "repair_exact", "repair"),
        "hard_bound_false_allow": _aggregate(hard_values, "hard_negative_false_allow", "bound_probe"),
        "hard_repair_exact": _aggregate(hard_values, "repair_exact", "repair"),
        "best_seed": min(results, key=lambda item: -item["selection_score"])["seed"],
    }
    report = {
        "protocol_id": "pg317-question-anchor-moe-v1",
        "schema_version": "pg317-question-anchor-moe-training-report-v1",
        "status": "completed_local_morning_pg317_question_anchor",
        "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "pg305_report": str(PG305_REPORT_PATH.relative_to(ROOT)), "pg305_report_sha256": pg305.get("report_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False},
        "training": {"architecture": "causal_transformer_moe_next_token", "target_representation": "multi_missing_question_anchor_plus_symbolic_slot_copy_plus_failure_repair", "binder": "deterministic_pg313_probe_variant", "config": CONFIG.__dict__, "device": "cpu", "seeds": list(SEEDS), "seed_checkpoint_dir": str(SEED_CHECKPOINT_DIR.relative_to(ROOT)), "question_pretrain_epochs": 200, "assembly_epochs": 260, "question_pretrain_rows": len(qtrain), "fit_count": len(train), "holdout_count": len(holdout), "hard_negative_count": len(hard), "anchor_train_rows": sum(int(row.get("counterfactual_kind") == "ask_complete_pair") for row in train), "elapsed_seconds": elapsed, "token_weights": weights},
        "metrics": metrics,
        "per_seed": results,
        "hypothesis_gate": {"status": "blocked", "checks": {"audit_pass": audit.get("status") == "passed", "anchor_question_min": metrics["holdout_anchor_question_exact"]["min"] >= 0.95, "anchor_zero_safe_allow": metrics["holdout_anchor_safe_allow_max"]["max"] == 0, "anchor_unnecessary_question_max": metrics["holdout_anchor_unnecessary_question"]["max"] <= 0.1, "original_question_min": metrics["holdout_missing_question_recall"]["min"] >= 0.9, "variant_exact_min": metrics["holdout_variant_exact"]["min"] >= 0.9, "repair_exact_min": metrics["holdout_repair_exact"]["min"] >= 0.9, "hard_zero_false_allow": metrics["hard_bound_false_allow"]["max"] == 0, "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["PG-317 is offline anchor training; independent fresh GET/POST replay is still required", "the adapter emits abstract Rule-IR references rather than literal payloads", "all-seed worst-case and family-out tests must remain green before any promotion", "no training or memory promotion"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gates": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
