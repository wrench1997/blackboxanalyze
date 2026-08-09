"""Run PG-301 abstract Rule-IR assembly with a causal Transformer-MoE."""

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
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, build_vocabulary, evaluate_causal_moe, generate_target, train_causal_moe  # noqa: E402
from app.pg301_payload_assembly import evaluate_assembly_rows  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg301_payload_assembly_dataset_v1.json"
AUDIT = RESEARCH / "pg301_payload_assembly_audit_v1.json"
REPORT = RESEARCH / "pg301_payload_assembly_training_report_v1_local_morning.json"
TRACE = RESEARCH / "pg301_payload_assembly_training_trace_v1_local_morning.json"
PROTOCOL = RESEARCH / "pg301_payload_assembly_training_protocol_v1_local_morning.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg301-payload-assembly"
CHECKPOINT = CHECKPOINT_DIR / "pg301_payload_assembly_moe_local_morning.pt"
SEEDS = (30101, 30102, 30103)
CONFIG = CausalMoEConfig(d_model=48, n_heads=2, n_layers=1, experts=2, expert_hidden=96, top_k=1, dropout=0.05, max_length=64)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None})
    return {key: {"mean": round(sum(float(row[key]) for row in rows) / len(rows), 6), "min": round(min(float(row[key]) for row in rows), 6), "max": round(max(float(row[key]) for row in rows), 6)} for key in keys}


def verify() -> tuple[dict[str, Any], datetime]:
    if os.environ.get("PG301_LOCAL_RUN") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-301 requires PG301_LOCAL_RUN=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-301 local training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("PG-301 requires one visible local CUDA device")
    assignment = {"execution_mode": "local_morning", "timestamp": now.isoformat(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(), "device_name": torch.cuda.get_device_name(0)}
    if assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or assignment["current_device"] != 0 or "A800" in assignment["device_name"]:
        raise RuntimeError(f"PG-301 requires one non-A800 local GPU0, got {assignment}")
    return assignment, now


def decode(model: CausalMoELanguageModel, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> list[list[str]]:
    return [generate_target(model, row.get("context_tokens", []), len(row.get("target_tokens", [])), vocabulary, device) for row in rows]


def main() -> None:
    assignment, now = verify()
    dataset = load(DATASET)
    audit = load(AUDIT)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-301 audit must pass")
    records = list(dataset.get("records") or [])
    train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    holdout = [row for row in records if row.get("split") == "implementation_holdout"]
    hard = [row for row in records if row.get("split") == "hard_negative_eval"]
    if not train or not holdout or not hard:
        raise RuntimeError("PG-301 requires train, implementation holdout and hard-negative rows")
    vocabulary = build_vocabulary(train)
    device = torch.device("cuda")
    started = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    for seed in SEEDS:
        random.seed(seed)
        model = train_causal_moe(train, vocabulary, device, seed=seed, config=CONFIG, epochs=120)
        decoded_train = decode(model, train, vocabulary, device)
        decoded_holdout = decode(model, holdout, vocabulary, device)
        decoded_hard = decode(model, hard, vocabulary, device)
        per_seed.append({
            "seed": seed,
            "train": {"causal": evaluate_causal_moe(model, train, vocabulary, device), "assembly": evaluate_assembly_rows(train, decoded_train)},
            "implementation_holdout": {"causal": evaluate_causal_moe(model, holdout, vocabulary, device), "assembly": evaluate_assembly_rows(holdout, decoded_holdout)},
            "hard_negative": {"causal": evaluate_causal_moe(model, hard, vocabulary, device), "assembly": evaluate_assembly_rows(hard, decoded_hard)},
        })
        snapshots.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    def lane(name: str) -> list[dict[str, Any]]:
        return [row[name] for row in per_seed]
    metrics = {
        "train": {"causal": aggregate([row["causal"] for row in lane("train")]), "assembly": aggregate([row["assembly"] for row in lane("train")])},
        "implementation_holdout": {"causal": aggregate([row["causal"] for row in lane("implementation_holdout")]), "assembly": aggregate([row["assembly"] for row in lane("implementation_holdout")])},
        "hard_negative": {"causal": aggregate([row["causal"] for row in lane("hard_negative")]), "assembly": aggregate([row["assembly"] for row in lane("hard_negative")])},
        "per_seed": per_seed,
    }
    hold_causal = metrics["implementation_holdout"]["causal"]
    hold_assembly = metrics["implementation_holdout"]["assembly"]
    hard_causal = metrics["hard_negative"]["causal"]
    hard_assembly = metrics["hard_negative"]["assembly"]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "resource_contract": assignment["visible_device_count"] == 1 and 8 <= now.hour < 18,
        "abstract_slot_contract": dataset.get("contract", {}).get("abstract_slots_only") is True,
        "typed_oracle_required": dataset.get("contract", {}).get("typed_oracle_required") is True,
        "implementation_holdout_present": bool(holdout),
        "question_recall_min": hold_causal.get("missing_question_recall", {}).get("min", 0.0) >= 0.8,
        "assembly_slot_exact_min": hold_assembly.get("assembly_slot_exact", {}).get("min", 0.0) >= 0.8,
        "hard_negative_zero_false_allow": hard_assembly.get("hard_negative_false_allow", {}).get("max", 999999) == 0,
        "hard_negative_unnecessary_question_max": hard_causal.get("unnecessary_question_rate", {}).get("max", 999999) <= 0.1,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg301-payload-assembly-training-v1",
        "schema_version": "pg301-payload-assembly-training-report-v1",
        "status": "completed_local_morning_pg301_payload_assembly",
        "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT).as_posix()), "audit_sha256": audit.get("audit_sha256"), "oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False},
        "device": assignment,
        "config": CONFIG.__dict__,
        "split": {"total": len(records), "train": len(train), "implementation_holdout": len(holdout), "hard_negative_eval": len(hard), "seeds": list(SEEDS), "epochs": 120},
        "metrics": metrics,
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "implementation_holdout": True}, "reasons": ["abstract Rule-IR slots only", "no fresh typed replay", "no real application gold"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "conclusion": "PG-301 measures causal assembly of abstract transport/field/encoding/stop slots; it is not a payload or vulnerability success claim.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg301-payload-assembly-checkpoint-v1", "assignment": assignment, "config": CONFIG.__dict__, "vocabulary": vocabulary, "state": snapshots[-1]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg301-payload-assembly-training-trace-v1", "report_sha256": report["report_sha256"], "training_eligible": False, "memory_write": False, "question_first": True, "causal_next_token": True, "abstract_transport_slots": True, "literal_payload": False, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg301-payload-assembly-training-protocol-v1", "execution_mode": "local_morning", "window": "08:00-18:00 Asia/Shanghai", "question_first": True, "abstract_slots_only": True, "typed_oracle_required": True, "fresh_reset_required": True, "negative_control_required": True, "causal_next_token_only": True, "same_context_hard_negative_required": True, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-302: bind abstract assembly to evaluator-only loopback adapter with fresh reset and typed replay."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": {"implementation_holdout": metrics["implementation_holdout"], "hard_negative": metrics["hard_negative"]}, "engineering_gate": report["engineering_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
