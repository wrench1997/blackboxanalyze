"""PG-319 decoder-only Transformer-MoE cross-implementation training.

This is a local morning CPU experiment.  The model predicts the abstract
Rule-IR target token by token; a deterministic binder validates slot assembly.
PG-318 live traces are loaded only as a frozen family-holdout canary and never
enter the optimizer.  A small new-only adaptation is measured against a
replay-mix adaptation so catastrophic forgetting is visible rather than
hidden by a single final score.
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


PG313 = _load_module("pg313_for_pg319", ROOT / "scripts" / "run_pg313_probe_variant_moe.py")
PG316 = _load_module("pg316_for_pg319", ROOT / "scripts" / "run_pg316_failure_repair_moe.py")
PG317 = _load_module("pg317_for_pg319", ROOT / "scripts" / "run_pg317_question_anchor_moe.py")
from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS  # noqa: E402
from app.pg301_payload_assembly import target_map  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg319_cross_impl_rule_ir_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg319_cross_impl_rule_ir_dataset_audit_v1.json"
PG318_TRACE_PATH = RESEARCH / "pg318_family_holdout_trace_v1.json"
REPORT_PATH = RESEARCH / "pg319_cross_impl_moe_training_report_v1_local_morning.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg319-cross-impl" / "pg319_cross_impl_moe_local_morning.pt"
SEED_CHECKPOINT_DIR = ROOT / "artifacts" / "pg319-cross-impl" / "seeds"
SEEDS = (31901, 31902, 31903)
CONFIG = PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-319 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-319 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _question_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = PG313._question_rows(rows)
    result.extend(PG317._anchor_question_rows(rows))
    return result


def _predictions(model: Any, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> list[list[str]]:
    return PG313._predictions(model, rows, vocab, device)


def _lane(model: Any, rows: list[dict[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    predictions = _predictions(model, rows, vocab, device)
    return {
        "causal_symbolic": PG313.evaluate_causal_moe(model, rows, vocab, device),
        "bound_probe": PG313._bound_metrics(rows, predictions),
        "anchor": PG317._anchor_metrics(rows, predictions),
        "repair": PG316._repair_metrics(rows, predictions),
    }


def _score(metrics: Mapping[str, Any]) -> float:
    bound = dict(metrics.get("bound_probe") or {})
    return float(bound.get("missing_question_recall") or 0.0) + float(bound.get("variant_exact") or 0.0) + float((metrics.get("repair") or {}).get("repair_exact") or 0.0) - 2.0 * float(bound.get("hard_negative_false_allow") or 0.0)


def _aggregate(values: Sequence[Mapping[str, Any]], section: str, key: str) -> dict[str, float]:
    nums = [float((value.get(section) or {}).get(key)) for value in values if (value.get(section) or {}).get(key) is not None]
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6)} if nums else {"mean": 0.0, "min": 0.0, "max": 0.0}


def _forgetting(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old_score = _score(before)
    new_score = _score(after)
    return {"old_score_before": round(old_score, 6), "old_score_after": round(new_score, 6), "score_drop": round(max(0.0, old_score - new_score), 6), "question_before": (before.get("bound_probe") or {}).get("missing_question_recall"), "question_after": (after.get("bound_probe") or {}).get("missing_question_recall"), "variant_before": (before.get("bound_probe") or {}).get("variant_exact"), "variant_after": (after.get("bound_probe") or {}).get("variant_exact"), "hard_false_allow_after": (after.get("bound_probe") or {}).get("hard_negative_false_allow")}


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    pg318_trace = _load(PG318_TRACE_PATH)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-319 requires passed cross-implementation dataset audit")
    if pg318_trace.get("training_eligible") is not False or pg318_trace.get("memory_promotion_allowed") is not False:
        raise RuntimeError("PG-319 refuses to train when PG-318 holdout is promotable")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [row for row in rows if row.get("split") == "train" and row.get("training_eligible")]
    vapp_train = [row for row in train if str(row.get("source")) == "pg246_vulnerableapp_independent_dom_holdout"]
    route_holdout = [row for row in rows if row.get("split") == "implementation_holdout"]
    seed_holdout = [row for row in rows if row.get("split") == "seed_holdout"]
    prior_holdout = [row for row in rows if row.get("split") in {"implementation_holdout", "real_live_holdout"} and str(row.get("source")) != "pg246_vulnerableapp_independent_dom_holdout"]
    hard = [row for row in rows if row.get("split") == "hard_negative_eval" or (bool(row.get("hard_negative")) and row.get("split") != "train")]
    pg318_eval = [dict(row) for row in pg318_trace.get("episodes", [])]
    qtrain = _question_rows(train)
    vocab = PG313.build_vocabulary(train + route_holdout + seed_holdout + prior_holdout + hard + pg318_eval + qtrain)
    device = torch.device("cpu")
    weights = {
        "question=ask_typed_availability": 8.0,
        "question=ask_replay_readiness": 8.0,
        "question=ask_evidence_presence": 8.0,
        "question=ask_feedback_state": 8.0,
        "question=ask_negative_control": 8.0,
        "question=ask_fresh_reset": 8.0,
        "safe_to_send=0": 6.0,
        "safe_to_send=1": 2.5,
        "next_action=request_observation": 8.0,
        "next_action=repair_abstract_plan": 7.0,
        "next_action=assemble_abstract_plan": 2.5,
        "repair_action=retry_bounded_variant": 7.0,
        "stop_condition=repair_feedback_or_abstain": 6.0,
        "probe_variant_ref=none": 6.0,
    }
    SEED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    started = time.monotonic()
    for seed in SEEDS:
        # Establish an old-capability anchor, then measure a deliberately
        # unsafe new-only adaptation against the replay-mix final model.
        prior_train = [row for row in train if str(row.get("source")) != "pg246_vulnerableapp_independent_dom_holdout"]
        prior_q = _question_rows(prior_train)
        prior_pre = PG313.train_causal_moe(prior_q, vocab, device, seed=seed, config=CONFIG, epochs=90, learning_rate=0.001, token_weights=weights)
        prior_base = PG313.train_causal_moe(prior_train, vocab, device, seed=seed + 10, config=CONFIG, epochs=150, learning_rate=0.001, token_weights=weights, initial_state=prior_pre.state_dict())
        old_before = _lane(prior_base, prior_holdout, vocab, device)
        new_only = PG313.train_causal_moe(vapp_train, vocab, device, seed=seed + 20, config=CONFIG, epochs=80, learning_rate=0.001, token_weights=weights, initial_state=prior_base.state_dict())
        old_after_new_only = _lane(new_only, prior_holdout, vocab, device)
        pre = PG313.train_causal_moe(qtrain, vocab, device, seed=seed + 30, config=CONFIG, epochs=140, learning_rate=0.001, token_weights=weights)
        model = PG313.train_causal_moe(train, vocab, device, seed=seed + 40, config=CONFIG, epochs=240, learning_rate=0.001, token_weights=weights, initial_state=pre.state_dict())
        lanes = {"train": _lane(model, train, vocab, device), "implementation_route_holdout": _lane(model, route_holdout, vocab, device), "seed_holdout": _lane(model, seed_holdout, vocab, device), "prior_holdout": _lane(model, prior_holdout, vocab, device), "hard_negative": _lane(model, hard, vocab, device), "pg318_family_holdout": _lane(model, pg318_eval, vocab, device)}
        forgetting = {"new_only": _forgetting(old_before, old_after_new_only), "replay_mix": _forgetting(old_before, lanes["prior_holdout"])}
        score = 2.0 * float((lanes["implementation_route_holdout"]["bound_probe"].get("missing_question_recall") or 0.0)) + 2.0 * float((lanes["implementation_route_holdout"]["bound_probe"].get("variant_exact") or 0.0)) + float((lanes["implementation_route_holdout"]["repair"].get("repair_exact") or 0.0)) - 4.0 * float((lanes["hard_negative"]["bound_probe"].get("hard_negative_false_allow") or 0.0))
        checkpoint = SEED_CHECKPOINT_DIR / f"pg319_cross_impl_moe_seed_{seed}.pt"
        torch.save({"schema_version": "pg319-cross-impl-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "seed": seed}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg318_trace_sha256": pg318_trace.get("trace_sha256")}, checkpoint)
        results.append({"seed": seed, "lanes": lanes, "forgetting": forgetting, "selection_score": round(score, 6), "checkpoint": str(checkpoint.relative_to(ROOT))})
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("PG-319 did not produce a checkpoint")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg319-cross-impl-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu", "selected": True}, "config": CONFIG.__dict__, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "pg318_trace_sha256": pg318_trace.get("trace_sha256")}, CHECKPOINT_PATH)
    route_values = [result["lanes"]["implementation_route_holdout"] for result in results]
    seed_values = [result["lanes"]["seed_holdout"] for result in results]
    pg318_values = [result["lanes"]["pg318_family_holdout"] for result in results]
    hard_values = [result["lanes"]["hard_negative"] for result in results]
    replay_drops = [float(result["forgetting"]["replay_mix"]["score_drop"]) for result in results]
    metrics = {
        "implementation_route_question_min": _aggregate(route_values, "bound_probe", "missing_question_recall"),
        "implementation_route_variant_min": _aggregate(route_values, "bound_probe", "variant_exact"),
        "implementation_route_repair_min": _aggregate(route_values, "repair", "repair_exact"),
        "implementation_route_false_allow_max": _aggregate(route_values, "bound_probe", "hard_negative_false_allow"),
        "seed_holdout_question_min": _aggregate(seed_values, "bound_probe", "missing_question_recall"),
        "seed_holdout_variant_min": _aggregate(seed_values, "bound_probe", "variant_exact"),
        "pg318_family_question_min": _aggregate(pg318_values, "bound_probe", "missing_question_recall"),
        "pg318_family_variant_min": _aggregate(pg318_values, "bound_probe", "variant_exact"),
        "pg318_family_false_allow_max": _aggregate(pg318_values, "bound_probe", "hard_negative_false_allow"),
        "hard_negative_false_allow_max": _aggregate(hard_values, "bound_probe", "hard_negative_false_allow"),
        "replay_mix_forgetting_drop_max": round(max(replay_drops) if replay_drops else 0.0, 6),
        "new_only_forgetting_drop_max": round(max(float(result["forgetting"]["new_only"]["score_drop"]) for result in results), 6),
        "best_seed": min(results, key=lambda item: -item["selection_score"])["seed"],
    }
    elapsed = round(time.monotonic() - started, 3)
    report = {
        "protocol_id": "pg-pk-319-cross-implementation-moe-v1",
        "schema_version": "pg319-cross-impl-moe-training-report-v1",
        "status": "completed_local_morning_pg319_cross_impl_moe",
        "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "pg318_trace": str(PG318_TRACE_PATH.relative_to(ROOT)), "pg318_trace_sha256": pg318_trace.get("trace_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False},
        "training": {"architecture": "causal_transformer_moe_next_token", "target_representation": "abstract_rule_ir_slot_assembly_plus_multi_missing_question_plus_failure_repair", "binder": "deterministic_pg313_probe_variant", "config": CONFIG.__dict__, "device": "cpu", "seeds": list(SEEDS), "question_pretrain_epochs": 140, "assembly_epochs": 240, "fit_count": len(train), "vapp_train_count": len(vapp_train), "route_holdout_count": len(route_holdout), "seed_holdout_count": len(seed_holdout), "pg318_family_holdout_count": len(pg318_eval), "hard_negative_count": len(hard), "elapsed_seconds": elapsed, "token_weights": weights},
        "metrics": metrics,
        "per_seed": results,
        "catastrophic_forgetting_canary": {"old_capability": "PG-317/earlier abstract holdout", "new_only_adaptation_drop_max": metrics["new_only_forgetting_drop_max"], "replay_mix_drop_max": metrics["replay_mix_forgetting_drop_max"], "threshold": 0.05, "replay_mix_within_threshold": metrics["replay_mix_forgetting_drop_max"] <= 0.05},
        "hypothesis_gate": {"status": "blocked", "checks": {"audit_pass": audit.get("status") == "passed", "route_question_min": metrics["implementation_route_question_min"]["min"] >= 0.9, "route_variant_min": metrics["implementation_route_variant_min"]["min"] >= 0.9, "route_repair_min": metrics["implementation_route_repair_min"]["min"] >= 0.9, "route_zero_false_allow": metrics["implementation_route_false_allow_max"]["max"] == 0, "seed_question_min": metrics["seed_holdout_question_min"]["min"] >= 0.9, "family_question_min": metrics["pg318_family_question_min"]["min"] >= 0.9, "family_variant_min": metrics["pg318_family_variant_min"]["min"] >= 0.9, "family_zero_false_allow": metrics["pg318_family_false_allow_max"]["max"] == 0, "hard_zero_false_allow": metrics["hard_negative_false_allow_max"]["max"] == 0, "replay_forgetting_within_threshold": metrics["replay_mix_forgetting_drop_max"] <= 0.05, "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["PG-319 only tests one additional VulnerableApp implementation and route/seed holdouts", "PG-318 remains a frozen evaluation-only holdout", "wire values remain source-grounded adapter outputs rather than literal decoder invention", "no training or long-term memory promotion"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gate": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
