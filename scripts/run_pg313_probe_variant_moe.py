"""PG-313 causal Transformer-MoE for abstract probe-variant selection."""

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
from app.pg295_causal_moe import CausalMoEConfig, build_vocabulary, evaluate_causal_moe, generate_target, train_causal_moe  # noqa: E402
from app.pg301_payload_assembly import TARGET_KEYS, target_map  # noqa: E402
from app.pg313_probe_variant import bind_probe_variant_plan  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg313_probe_variant_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg313_probe_variant_dataset_audit_v1.json"
PG305_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
REPORT_PATH = RESEARCH / "pg313_probe_variant_moe_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg313-probe-variant" / "pg313_probe_variant_moe_local_morning.pt"
SEED_CHECKPOINT_DIR = ROOT / "artifacts" / "pg313-probe-variant" / "seeds"
SEEDS = (31301, 31302, 31303)
CONFIG = CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-313 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-313 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _question_rows(train: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in train:
        q = next((str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")), "question=none")
        base = copy.deepcopy(dict(row))
        base["target_tokens"] = [TARGET_BOS, q, TARGET_EOS]
        for variant in ("identity", "probe", "recheck"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [f"history_action={variant}" if str(token).startswith("history_action=") else token for token in clone.get("context_tokens", [])]
            rows.append(clone)
    return rows


def _predictions(model: Any, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> list[list[str]]:
    return [generate_target(model, row.get("context_tokens", []), len(row.get("target_tokens", [])), vocab, device) for row in rows]


def _bound_metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[str]]) -> dict[str, Any]:
    base_keys = ("transport", "field_role", "encoding", "oracle", "stop_condition")
    variant_keys = ("probe_variant", "encoding_chain")
    base_correct = {key: 0 for key in base_keys}
    variant_correct = {key: 0 for key in variant_keys}
    exact = 0
    variant_total = 0
    variant_exact = 0
    question_total = 0
    question_correct = 0
    unnecessary = 0
    normal_total = 0
    false_allow = 0
    invalid_predicted = 0
    for row, prediction in zip(rows, predictions):
        context = [str(token) for token in row.get("context_tokens") or []]
        expected = bind_probe_variant_plan(row.get("target_tokens") or [], context) or []
        actual = bind_probe_variant_plan(prediction, context)
        if actual is None:
            invalid_predicted += 1
            actual = []
        expected_values = target_map(expected)
        actual_values = target_map(actual)
        exact += int(expected == actual)
        for key in base_keys:
            base_correct[key] += int(actual_values.get(key) == expected_values.get(key))
        for key in variant_keys:
            variant_correct[key] += int(actual_values.get(key) == expected_values.get(key))
        variant_total += 1
        variant_exact += int(all(actual_values.get(key) == expected_values.get(key) for key in variant_keys))
        expected_question = next((str(token).split("=", 1)[1] for token in row.get("target_tokens", []) if str(token).startswith("question=")), "none")
        actual_question = actual_values.get("question", "none")
        if expected_question != "none":
            question_total += 1
            question_correct += int(actual_question == expected_question)
        else:
            normal_total += 1
            unnecessary += int(actual_question != "none")
        false_allow += int(expected_values.get("safe_to_send") != "1" and actual_values.get("safe_to_send") == "1")
    count = max(len(rows), 1)
    return {"count": len(rows), "sequence_exact_accuracy": round(exact / count, 6), "slot_accuracy": {key: round(value / count, 6) for key, value in base_correct.items()}, "variant_accuracy": {key: round(value / count, 6) for key, value in variant_correct.items()}, "variant_exact": round(variant_exact / max(variant_total, 1), 6), "missing_question_recall": round(question_correct / max(question_total, 1), 6) if question_total else None, "unnecessary_question_rate": round(unnecessary / max(normal_total, 1), 6) if normal_total else 0.0, "hard_negative_false_allow": false_allow, "invalid_predicted_plan_count": invalid_predicted}


def _lane(model: Any, rows: list[dict[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    predictions = _predictions(model, rows, vocab, device)
    return {"causal_symbolic": evaluate_causal_moe(model, rows, vocab, device), "bound_probe": _bound_metrics(rows, predictions)}


def _aggregate(values: Sequence[Mapping[str, Any]], key: str, section: str) -> dict[str, float]:
    nums = [float((value.get(section) or {}).get(key)) for value in values if (value.get(section) or {}).get(key) is not None]
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6)} if nums else {"mean": 0.0, "min": 0.0, "max": 0.0}


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    pg305 = _load(PG305_REPORT_PATH)
    if audit.get("status") != "passed" or not (pg305.get("checks") or {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-313 requires passed dataset audit and PG-305 real evaluator")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [r for r in rows if r.get("split") == "train" and r.get("training_eligible")]
    holdout = [r for r in rows if r.get("split") in {"implementation_holdout", "real_live_holdout"}]
    hard = [r for r in rows if r.get("split") == "hard_negative_eval"]
    qtrain = _question_rows(train)
    vocab = build_vocabulary(train + holdout + hard + qtrain)
    device = torch.device("cpu")
    weights = {"question=ask_typed_availability": 3.0, "question=ask_replay_readiness": 3.0, "question=ask_evidence_presence": 3.0, "question=ask_feedback_state": 3.0, "question=ask_negative_control": 3.0, "question=ask_fresh_reset": 3.0, "safe_to_send=0": 2.2, "safe_to_send=1": 2.8, "next_action=request_observation": 2.3, "next_action=repair_abstract_plan": 2.3, "repair_action=retry_bounded_variant": 2.3, "stop_condition=repair_feedback_or_abstain": 2.3}
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    started = time.monotonic()
    for seed in SEEDS:
        pre = train_causal_moe(qtrain, vocab, device, seed=seed, config=CONFIG, epochs=140, learning_rate=0.001, token_weights=weights)
        model = train_causal_moe(train, vocab, device, seed=seed + 100, config=CONFIG, epochs=220, learning_rate=0.001, token_weights=weights, initial_state=pre.state_dict())
        train_m = _lane(model, train, vocab, device)
        hold_m = _lane(model, holdout, vocab, device)
        hard_m = _lane(model, hard, vocab, device)
        score = float(hold_m["bound_probe"].get("missing_question_recall") or 0.0) + float(hold_m["bound_probe"].get("variant_exact") or 0.0) + float(hold_m["causal_symbolic"].get("positive_recall") or 0.0) - 2.0 * float(hold_m["bound_probe"].get("hard_negative_false_allow") or 0.0) - 2.0 * float(hard_m["bound_probe"].get("hard_negative_false_allow") or 0.0)
        seed_checkpoint = SEED_CHECKPOINT_DIR / f"pg313_probe_variant_moe_seed_{seed}.pt"
        SEED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": "pg313-probe-variant-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "seed": seed}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, seed_checkpoint)
        results.append({"seed": seed, "train": train_m, "holdout": hold_m, "hard_negative": hard_m, "selection_score": round(score, 6), "checkpoint": str(seed_checkpoint.relative_to(ROOT))})
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("PG-313 did not produce a checkpoint")
    elapsed = round(time.monotonic() - started, 3)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg313-probe-variant-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu"}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, CHECKPOINT_PATH)
    hold_values = [r["holdout"] for r in results]
    hard_values = [r["hard_negative"] for r in results]
    metrics = {"holdout_missing_question_recall": _aggregate(hold_values, "missing_question_recall", "bound_probe"), "holdout_bound_base_slot_exact": _aggregate(hold_values, "sequence_exact_accuracy", "bound_probe"), "holdout_variant_exact": _aggregate(hold_values, "variant_exact", "bound_probe"), "holdout_unnecessary_question": _aggregate(hold_values, "unnecessary_question_rate", "bound_probe"), "holdout_bound_false_allow": _aggregate(hold_values, "hard_negative_false_allow", "bound_probe"), "hard_bound_false_allow": _aggregate(hard_values, "hard_negative_false_allow", "bound_probe"), "best_seed": min(results, key=lambda item: -item["selection_score"])["seed"]}
    report = {"protocol_id": "pg313-probe-variant-moe-v1", "schema_version": "pg313-probe-variant-moe-training-report-v1", "status": "completed_local_morning_pg313_probe_variant", "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "pg305_report": str(PG305_REPORT_PATH.relative_to(ROOT)), "pg305_report_sha256": pg305.get("report_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False}, "training": {"architecture": "causal_transformer_moe_next_token", "target_representation": "symbolic_slot_copy_plus_probe_variant_ref", "binder": "deterministic_pg313_probe_variant", "config": CONFIG.__dict__, "device": "cpu", "seeds": list(SEEDS), "seed_checkpoint_dir": str(SEED_CHECKPOINT_DIR.relative_to(ROOT)), "seeds": list(SEEDS), "question_pretrain_epochs": 140, "assembly_epochs": 220, "question_pretrain_rows": len(qtrain), "fit_count": len(train), "holdout_count": len(holdout), "hard_negative_count": len(hard), "elapsed_seconds": elapsed, "token_weights": weights}, "metrics": metrics, "per_seed": results, "hypothesis_gate": {"status": "blocked", "checks": {"audit_pass": audit.get("status") == "passed", "question_recall_min": metrics["holdout_missing_question_recall"]["min"] >= 0.9, "base_slot_exact_min": metrics["holdout_bound_base_slot_exact"]["min"] >= 0.9, "variant_exact_min": metrics["holdout_variant_exact"]["min"] >= 0.9, "unnecessary_question_max": metrics["holdout_unnecessary_question"]["max"] <= 0.1, "holdout_zero_false_allow": metrics["holdout_bound_false_allow"]["max"] == 0, "hard_zero_false_allow": metrics["hard_bound_false_allow"]["max"] == 0, "promotion_blocked": True}, "claim_allowed": False}, "scientific_gate": {"status": "blocked", "reasons": ["probe variant remains abstract and source-grounded adapter binding is not literal payload generation", "PG-312 used one implementation; independent implementation is still required", "no payload catalog or memory promotion until model-selected variant is replayed twice"], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"}, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT))}
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gates": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
