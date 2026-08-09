"""PG-316: next-token repair/variant anchor training and cross-seed audit."""

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


def _load_pg313() -> Any:
    path = ROOT / "scripts" / "run_pg313_probe_variant_moe.py"
    spec = importlib.util.spec_from_file_location("pg313_for_pg316", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-313 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG313 = _load_pg313()
from app.pg313_probe_variant import bind_probe_variant_plan  # noqa: E402
from app.pg301_payload_assembly import target_map  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg316_failure_repair_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg316_failure_repair_dataset_audit_v1.json"
PG305_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
REPORT_PATH = RESEARCH / "pg316_failure_repair_moe_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg316-failure-repair" / "pg316_failure_repair_moe_local_morning.pt"
SEED_CHECKPOINT_DIR = ROOT / "artifacts" / "pg316-failure-repair" / "seeds"
SEEDS = (31601, 31602, 31603)
CONFIG = PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-316 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-316 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _repair_metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[str]]) -> dict[str, Any]:
    total = 0
    correct = 0
    safe_allow = 0
    variant_non_none = 0
    for row, prediction in zip(rows, predictions):
        expected = bind_probe_variant_plan(row.get("target_tokens") or [], row.get("context_tokens") or []) or []
        expected_values = target_map(expected)
        if expected_values.get("next_action") != "repair_abstract_plan":
            continue
        total += 1
        actual = bind_probe_variant_plan(prediction, row.get("context_tokens") or [])
        values = target_map(actual or [])
        is_correct = values.get("next_action") == "repair_abstract_plan" and values.get("repair_action") == "retry_bounded_variant" and values.get("safe_to_send") == "0" and values.get("probe_variant") == "none" and values.get("encoding_chain") == "none"
        correct += int(is_correct)
        safe_allow += int(values.get("safe_to_send") == "1")
        variant_non_none += int(values.get("probe_variant") not in {None, "none", ""})
    return {"count": total, "repair_exact": round(correct / max(total, 1), 6), "safe_allow": safe_allow, "variant_non_none": variant_non_none}


def _lane(model: Any, rows: list[dict[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    predictions = PG313._predictions(model, rows, vocab, device)
    return {"causal_symbolic": PG313.evaluate_causal_moe(model, rows, vocab, device), "bound_probe": PG313._bound_metrics(rows, predictions), "repair": _repair_metrics(rows, predictions)}


def _aggregate(values: Sequence[Mapping[str, Any]], key: str, section: str) -> dict[str, float]:
    nums = [float((value.get(section) or {}).get(key)) for value in values if (value.get(section) or {}).get(key) is not None]
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6)} if nums else {"mean": 0.0, "min": 0.0, "max": 0.0}


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    pg305 = _load(PG305_REPORT_PATH)
    if audit.get("status") != "passed" or not (pg305.get("checks") or {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-316 requires passed dataset audit and PG-305 real evaluator")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [row for row in rows if row.get("split") == "train" and row.get("training_eligible")]
    holdout = [row for row in rows if row.get("split") in {"implementation_holdout", "real_live_holdout"}]
    hard = [row for row in rows if row.get("split") == "hard_negative_eval"]
    qtrain = PG313._question_rows(train)
    vocab = PG313.build_vocabulary(train + holdout + hard + qtrain)
    device = torch.device("cpu")
    weights = {
        "question=ask_typed_availability": 3.0,
        "question=ask_replay_readiness": 3.0,
        "question=ask_evidence_presence": 3.0,
        "question=ask_feedback_state": 3.0,
        "question=ask_negative_control": 3.0,
        "question=ask_fresh_reset": 3.0,
        "safe_to_send=0": 4.0,
        "safe_to_send=1": 2.8,
        "next_action=request_observation": 2.5,
        "next_action=repair_abstract_plan": 6.0,
        "next_action=assemble_abstract_plan": 2.0,
        "repair_action=retry_bounded_variant": 6.0,
        "stop_condition=repair_feedback_or_abstain": 4.0,
        "probe_variant_ref=none": 4.0,
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
        pre = PG313.train_causal_moe(qtrain, vocab, device, seed=seed, config=CONFIG, epochs=120, learning_rate=0.001, token_weights=weights)
        model = PG313.train_causal_moe(train, vocab, device, seed=seed + 100, config=CONFIG, epochs=240, learning_rate=0.001, token_weights=weights, initial_state=pre.state_dict())
        train_m = _lane(model, train, vocab, device)
        hold_m = _lane(model, holdout, vocab, device)
        hard_m = _lane(model, hard, vocab, device)
        score = float(hold_m["repair"].get("repair_exact") or 0.0) + float(hold_m["bound_probe"].get("variant_exact") or 0.0) + float(hold_m["bound_probe"].get("missing_question_recall") or 0.0) - 2.0 * float(hard_m["bound_probe"].get("hard_negative_false_allow") or 0.0) - 4.0 * float(hold_m["repair"].get("safe_allow") or 0.0)
        seed_checkpoint = SEED_CHECKPOINT_DIR / f"pg316_failure_repair_moe_seed_{seed}.pt"
        torch.save({"schema_version": "pg316-failure-repair-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "seed": seed}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, seed_checkpoint)
        results.append({"seed": seed, "train": train_m, "holdout": hold_m, "hard_negative": hard_m, "selection_score": round(score, 6), "checkpoint": str(seed_checkpoint.relative_to(ROOT))})
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("PG-316 did not produce a checkpoint")
    elapsed = round(time.monotonic() - started, 3)
    torch.save({"schema_version": "pg316-failure-repair-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "selected": True}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg305_report_sha256": pg305.get("report_sha256")}, CHECKPOINT_PATH)
    hold_values = [row["holdout"] for row in results]
    hard_values = [row["hard_negative"] for row in results]
    metrics = {
        "holdout_missing_question_recall": _aggregate(hold_values, "missing_question_recall", "bound_probe"),
        "holdout_variant_exact": _aggregate(hold_values, "variant_exact", "bound_probe"),
        "holdout_base_sequence_exact": _aggregate(hold_values, "sequence_exact_accuracy", "bound_probe"),
        "holdout_repair_exact": _aggregate(hold_values, "repair_exact", "repair"),
        "holdout_repair_safe_allow_max": _aggregate(hold_values, "safe_allow", "repair"),
        "hard_bound_false_allow": _aggregate(hard_values, "hard_negative_false_allow", "bound_probe"),
        "hard_repair_exact": _aggregate(hard_values, "repair_exact", "repair"),
        "best_seed": min(results, key=lambda item: -item["selection_score"])["seed"],
    }
    report = {
        "protocol_id": "pg316-failure-repair-moe-v1",
        "schema_version": "pg316-failure-repair-moe-training-report-v1",
        "status": "completed_local_morning_pg316_failure_repair",
        "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "pg305_report": str(PG305_REPORT_PATH.relative_to(ROOT)), "pg305_report_sha256": pg305.get("report_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False},
        "training": {"architecture": "causal_transformer_moe_next_token", "target_representation": "symbolic_slot_copy_plus_probe_variant_ref_plus_failure_repair", "binder": "deterministic_pg313_probe_variant", "config": CONFIG.__dict__, "device": "cpu", "seeds": list(SEEDS), "seed_checkpoint_dir": str(SEED_CHECKPOINT_DIR.relative_to(ROOT)), "question_pretrain_epochs": 120, "assembly_epochs": 240, "question_pretrain_rows": len(qtrain), "fit_count": len(train), "holdout_count": len(holdout), "hard_negative_count": len(hard), "repair_train_rows": sum(int(row.get("counterfactual_kind") == "failure_repair_pair") for row in train), "elapsed_seconds": elapsed, "token_weights": weights},
        "metrics": metrics,
        "per_seed": results,
        "hypothesis_gate": {"status": "blocked", "checks": {"audit_pass": audit.get("status") == "passed", "question_recall_min": metrics["holdout_missing_question_recall"]["min"] >= 0.9, "variant_exact_min": metrics["holdout_variant_exact"]["min"] >= 0.9, "repair_exact_min": metrics["holdout_repair_exact"]["min"] >= 0.9, "repair_safe_allow_max": metrics["holdout_repair_safe_allow_max"]["max"] == 0, "hard_zero_false_allow": metrics["hard_bound_false_allow"]["max"] == 0, "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["offline repair/variant anchor requires fresh independent live replay", "PG-315 showed seed-dependent variant misselection and no-repair; this run does not erase those failures", "abstract adapter output is not literal payload generation", "no training or memory promotion"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gates": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
