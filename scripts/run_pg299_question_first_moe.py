"""Train/evaluate PG-299 question-first causal Transformer-MoE."""

from __future__ import annotations

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

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, build_vocabulary, evaluate_causal_moe, train_causal_moe  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg299_question_first_dataset_v1.json"
AUDIT = RESEARCH / "pg299_question_first_audit_v1.json"
REPORT = RESEARCH / "pg299_question_first_training_report_v1_local_morning.json"
TRACE = RESEARCH / "pg299_question_first_training_trace_v1_local_morning.json"
PROTOCOL = RESEARCH / "pg299_question_first_training_protocol_v1_local_morning.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg299-question-first"
CHECKPOINT = CHECKPOINT_DIR / "pg299_question_first_moe_local_morning.pt"
SEEDS = (29901, 29902, 29903)
CONFIG = CausalMoEConfig(d_model=96, n_heads=4, n_layers=2, experts=4, expert_hidden=192, top_k=2, max_length=128)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None})
    return {key: {"mean": round(sum(float(row[key]) for row in rows) / len(rows), 6), "min": round(min(float(row[key]) for row in rows), 6), "max": round(max(float(row[key]) for row in rows), 6)} for key in keys}


def compact_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep all missing-question rows, then a bounded process support set."""

    missing = [row for row in rows if row.get("state_id") == "missing_key" or row.get("dropout_pattern") == "phase_unknown"]
    support = [row for row in rows if row not in missing][:240]
    return missing + support


def verify() -> tuple[dict[str, Any], datetime]:
    if os.environ.get("PG299_LOCAL_RUN") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-299 requires PG299_LOCAL_RUN=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-299 local training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("PG-299 requires one visible local CUDA device")
    assignment = {"execution_mode": "local_morning", "timestamp": now.isoformat(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(), "device_name": torch.cuda.get_device_name(0)}
    if assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or assignment["current_device"] != 0 or "A800" in assignment["device_name"]:
        raise RuntimeError(f"PG-299 requires one non-A800 local GPU0, got {assignment}")
    return assignment, now


def main() -> None:
    assignment, now = verify()
    dataset = load(DATASET)
    audit = load(AUDIT)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-299 audit must pass")
    records = list(dataset.get("records") or [])
    train_full = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    train = compact_training_rows(train_full)
    holdout = [row for row in records if row.get("split") == "implementation_holdout"]
    hard = [row for row in records if row.get("split") == "hard_negative_eval"]
    if not train or not holdout or not hard:
        raise RuntimeError("PG-299 requires train, implementation holdout and hard-negative rows")
    vocab = build_vocabulary(train)
    device = torch.device("cuda")
    started = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    for seed in SEEDS:
        random.seed(seed)
        model = train_causal_moe(train, vocab, device, seed=seed, config=CONFIG, epochs=90)
        per_seed.append({"seed": seed, "train": evaluate_causal_moe(model, train, vocab, device), "implementation_holdout": evaluate_causal_moe(model, holdout, vocab, device), "hard_negative": evaluate_causal_moe(model, hard, vocab, device)})
        snapshots.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    metrics = {"train": aggregate([row["train"] for row in per_seed]), "implementation_holdout": aggregate([row["implementation_holdout"] for row in per_seed]), "hard_negative": aggregate([row["hard_negative"] for row in per_seed]), "per_seed": per_seed}
    checks = {"dataset_audit_pass": audit.get("status") == "passed", "resource_contract": assignment["visible_device_count"] == 1 and 8 <= now.hour < 18, "question_first_contract": dataset.get("contract", {}).get("question_token_first") is True, "phase_dropout_present": int(dataset.get("counts", {}).get("phase_dropout_rows", 0) or 0) > 0, "implementation_holdout_present": bool(holdout), "question_recall_min": metrics["implementation_holdout"].get("missing_question_recall", {}).get("min", 0.0) >= 0.8, "hard_negative_zero_false_allow": metrics["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0, "promotion_blocked": True}
    report = {"protocol_id": "pg299-question-first-training-v1", "schema_version": "pg299-question-first-training-report-v1", "status": "completed_local_morning_pg299_question_first", "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT).as_posix()), "audit_sha256": audit.get("audit_sha256"), "oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False}, "device": assignment, "config": CONFIG.__dict__, "split": {"total": len(records), "train_full": len(train_full), "train_compact": len(train), "implementation_holdout": len(holdout), "hard_negative_eval": len(hard), "seeds": list(SEEDS), "epochs": 90}, "metrics": metrics, "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "implementation_holdout": True}, "reasons": ["abstract question-first data only", "no fresh typed replay", "no real application gold"], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False}, "conclusion": "PG-299 tests whether placing the question token first improves OOD active questioning; it is not a payload success claim.", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg299-question-first-checkpoint-v1", "assignment": assignment, "config": CONFIG.__dict__, "vocabulary": vocab, "state": snapshots[-1]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg299-question-first-training-trace-v1", "report_sha256": report["report_sha256"], "training_eligible": False, "memory_write": False, "question_first": True, "causal_next_token": True, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg299-question-first-training-protocol-v1", "execution_mode": "local_morning", "window": "08:00-18:00 Asia/Shanghai", "question_token_first": True, "phase_dropout": True, "causal_next_token_only": True, "same_context_hard_negative_required": True, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-300: hold out question slot values and test multi-step ask→observe→repair composition."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": {"implementation_holdout": metrics["implementation_holdout"], "hard_negative": metrics["hard_negative"]}, "engineering_gate": report["engineering_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
