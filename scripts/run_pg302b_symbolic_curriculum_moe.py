"""PG-302B: question pretraining followed by symbolic assembly SFT."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
import random
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS, sha256_json  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, build_vocabulary, evaluate_causal_moe, generate_target, train_causal_moe  # noqa: E402
from app.pg301_payload_assembly import evaluate_assembly_rows  # noqa: E402
from app.pg302_symbolic_assembly import bind_symbolic_plan  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg302_symbolic_assembly_dataset_v1.json"
AUDIT = RESEARCH / "pg302_symbolic_assembly_audit_v1.json"
REPORT = RESEARCH / "pg302b_symbolic_curriculum_training_report_v1_local_morning.json"
TRACE = RESEARCH / "pg302b_symbolic_curriculum_training_trace_v1_local_morning.json"
PROTOCOL = RESEARCH / "pg302b_symbolic_curriculum_training_protocol_v1_local_morning.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg302b-symbolic-curriculum"
CHECKPOINT = CHECKPOINT_DIR / "pg302b_symbolic_curriculum_moe_local_morning.pt"
SEEDS = (30211, 30212, 30213)
CONFIG = CausalMoEConfig(d_model=48, n_heads=2, n_layers=1, experts=2, expert_hidden=96, top_k=1, dropout=0.05, max_length=64)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None})
    return {key: {"mean": round(sum(float(row[key]) for row in rows) / len(rows), 6), "min": round(min(float(row[key]) for row in rows), 6), "max": round(max(float(row[key]) for row in rows), 6)} for key in keys}


def verify() -> tuple[dict[str, Any], datetime]:
    if os.environ.get("PG302B_LOCAL_RUN") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-302B requires PG302B_LOCAL_RUN=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-302B local training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("PG-302B requires one visible local CUDA device")
    assignment = {"execution_mode": "local_morning", "timestamp": now.isoformat(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(), "device_name": torch.cuda.get_device_name(0)}
    if assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or assignment["current_device"] != 0 or "A800" in assignment["device_name"]:
        raise RuntimeError(f"PG-302B requires one non-A800 local GPU0, got {assignment}")
    return assignment, now


def question_rows(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in train:
        question = next((str(token) for token in row.get("target_tokens", []) if str(token).startswith("question=")), "question=none")
        base = copy.deepcopy(row)
        base["target_tokens"] = [TARGET_BOS, question, TARGET_EOS]
        base["source_group"] = "question_pretrain"
        rows.append(base)
        # Two surface/history perturbations teach the question policy to
        # ignore presentation noise before full Rule-IR assembly fine-tuning.
        for variant in ("probe", "recheck"):
            clone = copy.deepcopy(base)
            clone["context_tokens"] = [f"history_action={variant}" if str(token).startswith("history_action=") else token for token in clone.get("context_tokens", [])]
            rows.append(clone)
    return rows


def lane_metrics(model: CausalMoELanguageModel, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    predictions = [generate_target(model, row.get("context_tokens", []), len(row.get("target_tokens", [])), vocabulary, device) for row in rows]
    symbolic = evaluate_causal_moe(model, rows, vocabulary, device)
    expected = []
    actual = []
    for row, prediction in zip(rows, predictions):
        clone = copy.deepcopy(row)
        clone["target_tokens"] = bind_symbolic_plan(row.get("target_tokens", []), row.get("context_tokens", [])) or []
        expected.append(clone)
        actual.append(bind_symbolic_plan(prediction, row.get("context_tokens", [])) or [])
    bound = evaluate_assembly_rows(expected, actual)
    return {"causal_symbolic": symbolic, "bound_abstract": bound}


def main() -> None:
    assignment, now = verify()
    dataset = load(DATASET)
    audit = load(AUDIT)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-302B source audit must pass")
    records = list(dataset.get("records") or [])
    train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    holdout = [row for row in records if row.get("split") == "implementation_holdout"]
    hard = [row for row in records if row.get("split") == "hard_negative_eval"]
    qtrain = question_rows(train)
    vocabulary = build_vocabulary(train + qtrain)
    device = torch.device("cuda")
    started = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    for seed in SEEDS:
        random.seed(seed)
        pretrain = train_causal_moe(qtrain, vocabulary, device, seed=seed, config=CONFIG, epochs=80)
        model = train_causal_moe(train, vocabulary, device, seed=seed + 100, config=CONFIG, epochs=120, initial_state=pretrain.state_dict())
        per_seed.append({"seed": seed, "train": lane_metrics(model, train, vocabulary, device), "implementation_holdout": lane_metrics(model, holdout, vocabulary, device), "hard_negative": lane_metrics(model, hard, vocabulary, device)})
        snapshots.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    def lane(name: str) -> list[dict[str, Any]]:
        return [row[name] for row in per_seed]
    def lane_aggregate(name: str, metric: str) -> dict[str, Any]:
        return aggregate([row[metric] for row in lane(name)])
    metrics = {name: {"causal_symbolic": lane_aggregate(name, "causal_symbolic"), "bound_abstract": lane_aggregate(name, "bound_abstract")} for name in ("train", "implementation_holdout", "hard_negative")}
    metrics["per_seed"] = per_seed
    hold_causal = metrics["implementation_holdout"]["causal_symbolic"]
    hold_bound = metrics["implementation_holdout"]["bound_abstract"]
    hard_causal = metrics["hard_negative"]["causal_symbolic"]
    hard_bound = metrics["hard_negative"]["bound_abstract"]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "resource_contract": assignment["visible_device_count"] == 1 and 8 <= now.hour < 18,
        "question_pretrain_present": len(qtrain) == len(train) * 3,
        "symbolic_slot_contract": dataset.get("contract", {}).get("symbolic_slot_references") is True,
        "deterministic_binder": dataset.get("contract", {}).get("deterministic_binder") is True,
        "implementation_holdout_present": bool(holdout),
        "question_recall_min": hold_causal.get("missing_question_recall", {}).get("min", 0.0) >= 0.9,
        "bound_assembly_slot_exact_min": hold_bound.get("assembly_slot_exact", {}).get("min", 0.0) >= 0.9,
        "hard_negative_zero_false_allow": hard_bound.get("hard_negative_false_allow", {}).get("max", 999999) == 0,
        "hard_negative_unnecessary_question_max": hard_causal.get("unnecessary_question_rate", {}).get("max", 999999) <= 0.1,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg302b-symbolic-curriculum-training-v1",
        "schema_version": "pg302b-symbolic-curriculum-training-report-v1",
        "status": "completed_local_morning_pg302b_symbolic_curriculum",
        "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT).as_posix()), "audit_sha256": audit.get("audit_sha256"), "oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False},
        "device": assignment,
        "config": CONFIG.__dict__,
        "curriculum": {"question_pretrain_rows": len(qtrain), "assembly_sft_rows": len(train), "question_pretrain_epochs": 80, "assembly_sft_epochs": 120, "initial_state_transfer": True},
        "split": {"total": len(records), "train": len(train), "implementation_holdout": len(holdout), "hard_negative_eval": len(hard), "seeds": list(SEEDS)},
        "metrics": metrics,
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "implementation_holdout": True}, "reasons": ["question-pretrain + symbolic assembly only", "no fresh typed replay", "no real application gold"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "conclusion": "PG-302B tests question pretraining followed by symbolic abstract assembly SFT; it is not a payload or vulnerability success claim.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg302b-symbolic-curriculum-checkpoint-v1", "assignment": assignment, "config": CONFIG.__dict__, "vocabulary": vocabulary, "state": snapshots[-1]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg302b-symbolic-curriculum-training-trace-v1", "report_sha256": report["report_sha256"], "training_eligible": False, "memory_write": False, "question_pretraining": True, "sft_initial_state_transfer": True, "symbolic_slot_references": True, "deterministic_binder": True, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg302b-symbolic-curriculum-training-protocol-v1", "execution_mode": "local_morning", "window": "08:00-18:00 Asia/Shanghai", "question_pretraining": True, "assembly_sft": True, "initial_state_transfer": True, "symbolic_slot_references": True, "deterministic_binder": True, "typed_oracle_required": True, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-303: evaluator-only loopback adapter and replay evidence binding for the symbolic plan."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": {"implementation_holdout": metrics["implementation_holdout"], "hard_negative": metrics["hard_negative"]}, "engineering_gate": report["engineering_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
