"""Train/evaluate PG-296B causal MoE with source-isolated missing-pattern augmentation."""

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
DATASET = RESEARCH / "pg296b_missing_augmentation_dataset_v1.json"
AUDIT = RESEARCH / "pg296b_missing_augmentation_audit_v1.json"
REPORT = RESEARCH / "pg296b_missing_augmentation_training_report_v1_local_morning.json"
TRACE = RESEARCH / "pg296b_missing_augmentation_training_trace_v1_local_morning.json"
PROTOCOL = RESEARCH / "pg296b_missing_augmentation_training_protocol_v1_local_morning.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg296b-missing-augmentation"
CHECKPOINT = CHECKPOINT_DIR / "pg296b_causal_moe_local_morning.pt"
SEEDS = (29601, 29602, 29603)
CONFIG = CausalMoEConfig(d_model=96, n_heads=4, n_layers=2, experts=4, expert_hidden=192, top_k=2, max_length=128)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None})
    return {key: {"mean": round(sum(float(row[key]) for row in rows) / len(rows), 6), "min": round(min(float(row[key]) for row in rows), 6), "max": round(max(float(row[key]) for row in rows), 6)} for key in keys}


def verify() -> tuple[dict[str, Any], datetime]:
    if os.environ.get("PG296B_LOCAL_RUN") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-296B requires PG296B_LOCAL_RUN=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-296B local training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("PG-296B requires one visible local CUDA device")
    assignment = {"execution_mode": "local_morning", "timestamp": now.isoformat(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(), "device_name": torch.cuda.get_device_name(0)}
    if assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or assignment["current_device"] != 0 or "A800" in assignment["device_name"]:
        raise RuntimeError(f"PG-296B requires one non-A800 local GPU0, got {assignment}")
    return assignment, now


def main() -> None:
    assignment, now = verify()
    dataset = load(DATASET)
    audit = load(AUDIT)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-296B audit must pass")
    records = list(dataset.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    ood_rows = [row for row in records if row.get("split") == "implementation_holdout"]
    hard_rows = [row for row in records if row.get("split") == "hard_negative_eval"]
    if not train_rows or not ood_rows or not hard_rows:
        raise RuntimeError("PG-296B requires train, implementation holdout and hard-negative rows")
    vocab = build_vocabulary(train_rows)
    device = torch.device("cuda")
    started = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    for seed in SEEDS:
        random.seed(seed)
        model = train_causal_moe(train_rows, vocab, device, seed=seed, config=CONFIG, epochs=140)
        per_seed.append({"seed": seed, "train": evaluate_causal_moe(model, train_rows, vocab, device), "ood": evaluate_causal_moe(model, ood_rows, vocab, device), "hard_negative": evaluate_causal_moe(model, hard_rows, vocab, device)})
        snapshots.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    summary = {"train": aggregate([row["train"] for row in per_seed]), "ood": aggregate([row["ood"] for row in per_seed]), "hard_negative": aggregate([row["hard_negative"] for row in per_seed]), "per_seed": per_seed}
    checks = {"dataset_audit_pass": audit.get("status") == "passed", "resource_contract": assignment["visible_device_count"] == 1 and 8 <= now.hour < 18, "train_pattern_diversity": len(set(dataset.get("counts", {}).get("train_patterns", []))) >= 3, "implementation_holdout_present": bool(ood_rows), "question_recall_min": summary["ood"].get("missing_question_recall", {}).get("min", 0.0) >= 0.8, "hard_negative_zero_false_allow": summary["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0, "promotion_blocked": True}
    report = {"protocol_id": "pg296b-missing-augmentation-training-v1", "schema_version": "pg296b-missing-augmentation-training-report-v1", "status": "completed_local_morning_pg296b_missing_augmentation", "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT).as_posix()), "audit_sha256": audit.get("audit_sha256"), "oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False}, "device": assignment, "config": CONFIG.__dict__, "split": {"total": len(records), "train": len(train_rows), "implementation_holdout": len(ood_rows), "hard_negative_eval": len(hard_rows), "seeds": list(SEEDS)}, "metrics": summary, "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "implementation_holdout": True}, "reasons": ["abstract missing-pattern data only", "no fresh typed replay", "no real application gold"], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False}, "conclusion": "PG-296B tests whether adding two missingness patterns improves a third unseen pattern; it is not a payload success claim.", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg296b-causal-moe-checkpoint-v1", "assignment": assignment, "config": CONFIG.__dict__, "vocabulary": vocab, "state": snapshots[-1]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg296b-missing-augmentation-training-trace-v1", "report_sha256": report["report_sha256"], "training_eligible": False, "memory_write": False, "oracle_blind": True, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg296b-missing-augmentation-training-protocol-v1", "execution_mode": "local_morning", "window": "08:00-18:00 Asia/Shanghai", "train_patterns": dataset["counts"]["train_patterns"], "holdout_patterns": dataset["counts"]["holdout_patterns"], "causal_next_token_only": True, "same_context_hard_negative_required": True, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-297: add a genuinely independent parser/implementation and keep one missingness family fully held out."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": {"ood": summary["ood"], "hard_negative": summary["hard_negative"]}, "engineering_gate": report["engineering_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
