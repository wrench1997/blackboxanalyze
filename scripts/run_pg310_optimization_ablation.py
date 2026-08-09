"""PG-310: optimization/capacity ablation on the fixed PG-309 dataset.

This is a controlled experiment, not a promotion path.  It compares a
zero-dropout longer-budget compact MoE with a wider MoE.  Data, source split,
target representation, deterministic binder and evaluator gate are identical
across variants so any change can be attributed to optimization/capacity.
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
from run_pg308_multisource_slot_moe import _aggregate, _lane  # noqa: E402

RESEARCH = ROOT / "research"
DATASET_PATH = RESEARCH / "pg309_balanced_counterfactual_dataset_v1.json"
AUDIT_PATH = RESEARCH / "pg309_balanced_counterfactual_dataset_audit_v1.json"
PG305_REPORT_PATH = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
REPORT_PATH = RESEARCH / "pg310_optimization_ablation_report_v1_local_morning.json"
SEEDS = (31001, 31002, 31003)
VARIANTS = {
    "long_zero_dropout": {"config": CausalMoEConfig(d_model=48, n_heads=2, n_layers=1, experts=2, expert_hidden=96, top_k=1, dropout=0.0, max_length=64), "lr": 0.001, "pretrain_epochs": 100, "assembly_epochs": 220, "safe_positive_weight": 2.5},
    "wide_zero_dropout": {"config": CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=64), "lr": 0.0015, "pretrain_epochs": 80, "assembly_epochs": 180, "safe_positive_weight": 2.6},
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-310 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-310 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _question_rows(train: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in train:
        q = next((str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")), "question=none")
        base = copy.deepcopy(dict(row))
        base["target_tokens"] = [TARGET_BOS, q, TARGET_EOS]
        for variant in ("identity", "recheck"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [f"history_action={variant}" if str(token).startswith("history_action=") else token for token in clone.get("context_tokens", [])]
            rows.append(clone)
    return rows


def _run_variant(name: str, spec: Mapping[str, Any], train: list[dict[str, Any]], holdout: list[dict[str, Any]], hard: list[dict[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    config = spec["config"]
    qtrain = _question_rows(train)
    token_weights = {"question=ask_typed_availability": 2.5, "question=ask_replay_readiness": 2.5, "question=ask_evidence_presence": 2.5, "question=ask_feedback_state": 2.5, "question=ask_negative_control": 2.5, "question=ask_fresh_reset": 2.5, "safe_to_send=0": 2.3, "safe_to_send=1": float(spec["safe_positive_weight"]), "next_action=request_observation": 2.2, "next_action=repair_abstract_plan": 2.2, "repair_action=retry_bounded_variant": 2.2, "stop_condition=repair_feedback_or_abstain": 2.2}
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_score = float("-inf")
    variant_started = time.monotonic()
    for seed in SEEDS:
        pre = train_causal_moe(qtrain, vocabulary, device, seed=seed, config=config, epochs=int(spec["pretrain_epochs"]), learning_rate=float(spec["lr"]), token_weights=token_weights)
        model = train_causal_moe(train, vocabulary, device, seed=seed + 100, config=config, epochs=int(spec["assembly_epochs"]), learning_rate=float(spec["lr"]), token_weights=token_weights, initial_state=pre.state_dict())
        train_m = _lane(model, train, vocabulary, device)
        hold_m = _lane(model, holdout, vocabulary, device)
        hard_m = _lane(model, hard, vocabulary, device)
        score = float(hold_m["causal_symbolic"].get("missing_question_recall") or 0.0) + float(hold_m["bound_concrete"].get("assembly_slot_exact") or 0.0) + float(hold_m["causal_symbolic"].get("positive_recall") or 0.0) - 2.0 * float(hold_m["bound_concrete"].get("hard_negative_false_allow") or 0.0) - 2.0 * float(hard_m["bound_concrete"].get("hard_negative_false_allow") or 0.0)
        results.append({"seed": seed, "train": train_m, "holdout": hold_m, "hard_negative": hard_m, "selection_score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_model = model
    hold_values = [r["holdout"] for r in results]
    hard_values = [r["hard_negative"] for r in results]
    metrics = {"holdout_missing_question_recall": _aggregate(hold_values, "missing_question_recall", "causal_symbolic"), "holdout_bound_slot_exact": _aggregate(hold_values, "assembly_slot_exact", "bound_concrete"), "holdout_unnecessary_question": _aggregate(hold_values, "unnecessary_question_rate", "causal_symbolic"), "holdout_bound_false_allow": _aggregate(hold_values, "hard_negative_false_allow", "bound_concrete"), "holdout_raw_false_allow": _aggregate(hold_values, "hard_negative_false_allow", "causal_symbolic"), "hard_bound_false_allow": _aggregate(hard_values, "hard_negative_false_allow", "bound_concrete"), "hard_raw_false_allow": _aggregate(hard_values, "hard_negative_false_allow", "causal_symbolic"), "best_seed": min(results, key=lambda item: -item["selection_score"])["seed"]}
    if best_model is None:
        raise RuntimeError(f"PG-310 variant {name} did not produce a model")
    checkpoint = ROOT / "artifacts" / "pg310-optimization" / f"pg310_{name}_local_morning.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg310-optimization-checkpoint-v1", "variant": name, "assignment": {"execution_mode": "local_morning_cpu", "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "device": "cpu"}, "config": config.__dict__, "vocabulary": vocabulary, "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}}, checkpoint)
    return {"variant": name, "config": config.__dict__, "optimization": {"lr": spec["lr"], "pretrain_epochs": spec["pretrain_epochs"], "assembly_epochs": spec["assembly_epochs"], "safe_positive_weight": spec["safe_positive_weight"], "elapsed_seconds": round(time.monotonic() - variant_started, 3)}, "metrics": metrics, "per_seed": results, "checkpoint": str(checkpoint.relative_to(ROOT))}


def main() -> int:
    _require_gate()
    dataset = _load(DATASET_PATH)
    audit = _load(AUDIT_PATH)
    pg305 = _load(PG305_REPORT_PATH)
    if audit.get("status") != "passed" or not (pg305.get("checks") or {}).get("real_docker_contacted", False):
        raise RuntimeError("PG-310 requires passed PG-309 audit and PG-305 real evaluator")
    rows = [dict(row) for row in dataset.get("records", [])]
    train = [r for r in rows if r.get("split") == "train" and r.get("training_eligible")]
    holdout = [r for r in rows if r.get("split") in {"implementation_holdout", "real_live_holdout"}]
    hard = [r for r in rows if r.get("split") == "hard_negative_eval"]
    vocabulary = build_vocabulary(train + holdout + hard + _question_rows(train))
    device = torch.device("cpu")
    started = time.monotonic()
    variants = {name: _run_variant(name, spec, train, holdout, hard, vocabulary, device) for name, spec in VARIANTS.items()}
    report = {"protocol_id": "pg310-optimization-ablation-v1", "schema_version": "pg310-optimization-ablation-report-v1", "status": "completed_local_morning_pg310_optimization_ablation", "source": {"dataset": str(DATASET_PATH.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT_PATH.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "pg305_report": str(PG305_REPORT_PATH.relative_to(ROOT)), "pg305_report_sha256": pg305.get("report_sha256"), "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_emission": False}, "variants": variants, "total_elapsed_seconds": round(time.monotonic() - started, 3), "scientific_gate": {"status": "blocked", "reasons": ["optimization/capacity ablation cannot replace missing fresh model-generated replay", "all variants remain compared on the same PG-309 source holdout", "promotion and vulnerability claims remain closed"], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    report["report_sha256"] = _digest(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "variants": {k: v["metrics"] for k, v in variants.items()}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
